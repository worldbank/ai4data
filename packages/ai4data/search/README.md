# @ai4data/search

Framework-agnostic semantic search client for the **AI for Data – Data for AI** program. Combines HNSW approximate nearest-neighbour search, BM25 lexical search, and hybrid ranking — all running in a Web Worker. Published under the [@ai4data](https://www.npmjs.com/org/ai4data) npm organization.

**Browser only:** requires a browser environment (Web Workers, `fetch`, Cache API).

## Installation

```bash
npm install @ai4data/search
# or
yarn add @ai4data/search
# or
pnpm add @ai4data/search
```

## Usage

```ts
import { SearchClient } from '@ai4data/search'

const client = new SearchClient('https://example.com/data/your-collection/manifest.json')

client.on('index_ready', () => {
  client.search('climate finance', { topK: 10, mode: 'hybrid' })
})

client.on('results', ({ data, stats }) => {
  console.log(data)   // SearchResult[]
  console.log(stats)  // SearchStats | null
})

// Clean up when done
client.destroy()
```

### Curated highlights (optional)

If **`manifest.json`** includes **`index.highlights`** (e.g. `"index/highlights.json"`), host that file next to your index. Use the same field shape as **`titles.json`** entries (`id`, `title`, `idno`, `type`, …). For a fixed display order, use a **JSON array** of objects; an **object keyed by id** (like `titles.json`) is also supported (enumeration order).

```ts
client.on('highlights', ({ data }) => setSpotlightDocs(data))
client.on('index_ready', () => client.getHighlights(8))
```

### Loading from a CDN (any origin)

When you load the package from a CDN (e.g. via an import map) and your page is on another origin, the browser blocks creating a worker from the CDN URL. Use **`SearchClient.fromCDN()`** so the worker is fetched and run from a blob URL (same-origin):

```ts
import { SearchClient } from '@ai4data/search'

const client = await SearchClient.fromCDN(manifestUrl)
client.on('results', ({ data }) => console.log(data))
```

`workerUrl` defaults to unpkg for the **current package version** (injected at build time), so you usually don't need to pass it. To pin a different version, pass `workerUrl: 'https://unpkg.com/@ai4data/search@0.1.0/dist/worker.mjs'`. If you pass an esm.sh worker URL, the client fetches the raw bundle from unpkg (esm.sh returns a wrapper that fails from a blob). This works from any static host with no build step.

### CSP and Hugging Face (embedding proxy)

The search worker loads the embedding model with **`@xenova/transformers`**, which fetches ONNX and tokenizer files from the Hugging Face Hub by default (`https://huggingface.co/...`). If your page’s **Content Security Policy** blocks `connect-src` to `huggingface.co` or to LFS/CDN hosts, those requests will fail.

**Recommended approach:** serve a **same-origin reverse proxy** that forwards to the Hub, and point Transformers.js at it via **`env.remoteHost`**. This package exposes that as **`transformersRemoteHost`** (and optionally **`transformersRemotePathTemplate`** if your proxy does not mirror the Hub path layout).

- **`modelId`** stays a **full Hub repo id** (e.g. `avsolatorio/GIST-small-Embedding-v0`). It is **not** a raw directory URL; the library builds paths like `…/{model}/resolve/{revision}/tokenizer.json`.
- Pass an **absolute** base URL (or a path relative to the page, which `SearchClient` resolves against `location.href` so blob workers still target your app origin):

```ts
const origin = typeof location !== 'undefined' ? location.origin : 'https://example.com'

const client = new SearchClient(manifestUrl, {
  transformersRemoteHost: `${origin}/api/hf-proxy/`,
})

await SearchClient.fromCDN(manifestUrl, {
  workerUrl: `${origin}/dist/worker.mjs`,
  transformersRemoteHost: `${origin}/api/hf-proxy/`,
})
```

Your proxy should forward to the Hub with the **same path suffix** after the prefix, for example:

`GET https://your.app/api/hf-proxy/avsolatorio/GIST-small-Embedding-v0/resolve/main/config.json`  
→ `GET https://huggingface.co/avsolatorio/GIST-small-Embedding-v0/resolve/main/config.json`

**Important:** Hub responses often **302** to `cdn-lfs.huggingface.co`. The proxy must **follow redirects on the server** and return the final 200 response to the browser; if you pass 302 through, the browser will leave your origin and hit CSP again. The demo server below does this with `fetch(..., { redirect: 'follow' })`.

A second path shape (`/api/resolve-cache/models/<org>/<model>/<revision>/<file…>`) is supported by the same demo server for tools that emit that URL pattern; it maps to `https://huggingface.co/<org>/<model>/resolve/<revision>/<file…>`.

### Disable BM25 (semantic-only)

To skip loading and building the BM25 index even when the manifest includes `bm25_corpus` (e.g. to save memory or avoid lexical search):

```ts
new SearchClient(manifestUrl, { skipBm25: true })
// or with fromCDN:
await SearchClient.fromCDN(manifestUrl, { skipBm25: true })
```

Search will run in semantic-only mode; `mode: 'lexical'` and `mode: 'hybrid'` will behave as semantic when BM25 is disabled.

### Custom worker path (bundler)

If your bundler does not resolve the default worker URL, pass a factory:

```ts
new SearchClient(manifestUrl, {
  workerFactory: () => new Worker(
    new URL('@ai4data/search/worker', import.meta.url),
    { type: 'module' }
  )
})
```

### Rerank worker (optional)

For cross-encoder reranking of results, use the separate rank worker:

```ts
const rankWorker = new Worker(
  new URL('@ai4data/search/rank-worker', import.meta.url),
  { type: 'module' }
)
// Send { query, documents, top_k } and receive scored results
```

Vue and React adapters are planned (future: `@ai4data/search/vue`, `@ai4data/search/react`).

## Building an index (Python pipeline)

`SearchClient` expects a **hosted directory** whose entry point is **`manifest.json`**, plus the index files it references (HNSW shards, `titles.json`, optional `bm25_corpus.json`, optional manually curated **`index/highlights.json`** via `index.highlights` in the manifest, or a flat `embeddings.int8.json` for small collections). This package does **not** build those artifacts; they come from the **search pipeline** in the main **ai4data** repository (except `highlights.json`, which you add yourself).

| Step | Script | Role |
|------|--------|------|
| 1 | `01_fetch_and_prepare.py` | Ingest documents → `metadata.json` |
| 2 | `02_generate_embeddings.py` | Encode with sentence-transformers → `raw_embeddings.npy`, flat index pieces |
| 3 | `03_build_index.py` | Build HNSW (or flat) layout → **`manifest.json`**, `index/`, etc. |

**End-to-end:** run `pipeline.py` from the repo (with [uv](https://docs.astral.sh/uv/)) after installing the Python extra **`search`** (see root `pyproject.toml`):

```bash
# From the ai4data repo root
uv pip install -e ".[search]"
# or: uv sync --extra search

uv run python scripts/search/pipeline/pipeline.py --help
```

Full options, outputs (including gzip vs GitHub Pages), and examples: **[Search pipeline README](https://github.com/worldbank/ai4data/blob/main/scripts/search/pipeline/README.md)** (in-repo copy: [`scripts/search/pipeline/README.md`](../../../scripts/search/pipeline/README.md)).

Host the resulting output directory on any static host (or object storage with CORS) and pass the **absolute URL** to `manifest.json` into `new SearchClient(...)`.

## Demo

Demos are included under **`demo/`**:

1. **Static server** (local `dist/`): run **`npm run demo`**, then open **http://localhost:5173/demo/** (uses [serve](https://www.npmjs.com/package/serve)).
2. **Standalone HTML** (loads the package from npm via [esm.sh](https://esm.sh)): open **demo/standalone.html** in a browser. Serve the file over HTTP (e.g. `npx serve .` from this package). No build step; the file uses **`SearchClient.fromCDN`** with a worker URL (see the file).
3. **HF proxy + CSP-friendly demo**: run **`npm run demo:proxy`** from this package (builds, then starts **demo/proxy-server.mjs**). Open **http://localhost:5173/** — it serves **demo/hf-proxy-demo.html**, which sets **`transformersRemoteHost`** to the same-origin **`/api/hf-proxy/`** prefix. The proxy forwards to `huggingface.co` and **follows redirects server-side** so the browser only talks to localhost.

All demos need a **manifest URL** pointing at a hosted **`manifest.json`** (from the Python pipeline above or an existing collection).

## Development

From the repo root (with workspaces):

```bash
npm install
npm run build --workspace=@ai4data/search
```

Or from this directory:

```bash
cd packages/ai4data/search
npm install
npm run build
```

## Publishing (maintainers)

The package is published under the [@ai4data](https://www.npmjs.com/org/ai4data) npm organization. Only maintainers with publish access to the org can release.

**Prerequisites**

- npm account that is a member of the **ai4data** org with permission to publish.
- Logged in locally: `npm login` (use your npm credentials or a machine account token).

**Steps**

1. Bump the version in `package.json` (or use `npm version patch|minor|major` from this directory).
2. From the package directory, publish with public access (required for scoped packages):

   ```bash
   cd packages/ai4data/search
   npm publish --access public
   ```

   The `prepublishOnly` script runs `npm run build` automatically before packing, so the published tarball always includes an up-to-date `dist/`.

3. Optionally tag the release in git and push:

   ```bash
   git tag @ai4data/search@1.0.0
   git push origin @ai4data/search@1.0.0
   ```

**What gets published**

- Only the `dist/` directory (built ESM and types) and `README.md` are included (see `files` in `package.json`).
- Consumers install with: `npm install @ai4data/search`.

## Documentation

- [Main project docs](https://worldbank.github.io/ai4data)
- [Repo structure](../../../docs/repo-structure.md)

## License

MIT License with World Bank IGO Rider. See [LICENSE](../../../LICENSE) in the repo root.
