/**
 * Minimal static server + Hugging Face Hub reverse proxy for local CSP demos.
 *
 * Serves the package root (so /dist/* works after `npm run build`) and forwards:
 *   GET /api/hf-proxy/<path>  →  https://huggingface.co/<path>
 *
 * Some tools use a different local prefix (commit in the path, not …/resolve/main/…):
 *   GET /api/resolve-cache/models/<org>/<model>/<revision>/<file…>
 *     → https://huggingface.co/<org>/<model>/resolve/<revision>/<file…>
 * Query strings are not forwarded (HF raw files do not need them; bogus cache keys cause 404s).
 *
 * Redirects are followed **on the server** (fetch + redirect: follow). Hub files often 302 to
 * cdn-lfs.huggingface.co; if we passed that through to the browser, the client would leave
 * localhost and hit CSP blocks — endless orange redirects in DevTools.
 *
 * For local development only — do not expose this proxy on the public internet.
 */

import http from "http";
import fs from "fs";
import path from "path";
import { Readable } from "stream";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const PORT = Number(process.env.PORT) || 5173;
const HF_ORIGIN = "https://huggingface.co";

/** Hop-by-hop / headers we must not forward when re-streaming the final response */
const SKIP_RESPONSE_HEADER = new Set([
  "connection",
  "content-encoding",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

/**
 * Fetch Hub URL with redirects resolved on the server, then stream the final body to the client.
 * Prevents the browser from following 302s to huggingface.co / cdn-lfs (CSP / redirect chains).
 */
async function proxyFetchFollow(targetUrl, res, req) {
  const ac = new AbortController();
  res.on("close", () => ac.abort());
  try {
    const r = await fetch(targetUrl, {
      redirect: "follow",
      signal: ac.signal,
      headers: {
        "user-agent": req?.headers?.["user-agent"] || "ai4data-search-demo-proxy",
        accept: req?.headers?.accept || "*/*",
      },
    });

    const outHeaders = {};
    r.headers.forEach((value, key) => {
      const lower = key.toLowerCase();
      if (lower === "content-security-policy") return;
      if (SKIP_RESPONSE_HEADER.has(lower)) return;
      outHeaders[key] = value;
    });
    outHeaders["access-control-allow-origin"] = "*";

    res.writeHead(r.status, outHeaders);

    if (!r.body) {
      res.end();
      return;
    }

    const nodeReadable = Readable.fromWeb(r.body);
    nodeReadable.on("error", () => {
      if (!res.writableEnded) res.destroy();
    });
    nodeReadable.pipe(res);
  } catch (e) {
    if (e?.name === "AbortError") return;
    if (!res.headersSent) {
      res.writeHead(502, { "Content-Type": "text/plain; charset=utf-8" });
      res.end(`Proxy error: ${e?.message ?? e}`);
    }
  }
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".mjs": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

function safeFilePath(root, pathname) {
  const stripped = decodeURIComponent(pathname).replace(/^\//, "");
  const rel = path.normalize(stripped).replace(/^(\.\.(\/|\\|$))+/, "");
  const full = path.join(root, rel);
  const rootWithSep = root.endsWith(path.sep) ? root : root + path.sep;
  if (!full.startsWith(rootWithSep) && full !== root) return null;
  return full;
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url || "/", `http://127.0.0.1:${PORT}`);

  if (url.pathname.startsWith("/api/hf-proxy/")) {
    const hfPath = url.pathname.slice("/api/hf-proxy".length) + url.search;
    const target = HF_ORIGIN + hfPath;
    void proxyFetchFollow(target, res, req);
    return;
  }

  // Alternate path shape used by some hub helpers / tooling (not Xenova's default template).
  // Example: /api/resolve-cache/models/avsolatorio/GIST-small-Embedding-v0/<commit>/config.json
  if (url.pathname.startsWith("/api/resolve-cache/models/")) {
    const rest = url.pathname.slice("/api/resolve-cache/models/".length);
    const segments = rest.split("/").filter(Boolean);
    if (segments.length >= 4) {
      const org = segments[0];
      const model = segments[1];
      const revision = segments[2];
      const fileInRepo = segments.slice(3).join("/");
      const target = `${HF_ORIGIN}/${org}/${model}/resolve/${revision}/${fileInRepo}`;
      void proxyFetchFollow(target, res, req);
      return;
    }
    res.writeHead(400, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(
      "Bad /api/resolve-cache/models/ path. Expected /api/resolve-cache/models/<org>/<name>/<revision>/<file...>",
    );
    return;
  }

  let pathname = url.pathname === "/" ? "/demo/hf-proxy-demo.html" : url.pathname;
  const filePath = safeFilePath(ROOT, pathname);
  if (!filePath) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Not found");
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
});

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Open http://localhost:${PORT}/`);
  // eslint-disable-next-line no-console
  console.log(`HF proxy: /api/hf-proxy/ → ${HF_ORIGIN}/`);
  // eslint-disable-next-line no-console
  console.log(`HF proxy: /api/resolve-cache/models/… → ${HF_ORIGIN}/<org>/<model>/resolve/<rev>/…`);
  // eslint-disable-next-line no-console
  console.log("Run `npm run build` in this package first so /dist/* is available.");
});
