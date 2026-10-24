// Keeps published addresses of the site working.
//
// published-routes.txt lists every page route the site has published. redirects.json maps
// an old route to its new one. After each build, this plugin writes a redirect page at
// every old route and fails the build when a published route has neither a page nor a
// redirect. A built route that is missing from published-routes.txt fails the build in CI
// and is a warning locally; `npm run routes:update` adds the routes of the last build.
// See website/README.md, "Changing or removing a page".
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ROUTES_FILE = path.join(siteDir, 'published-routes.txt');
const REDIRECTS_FILE = path.join(siteDir, 'redirects.json');

// Routes of the HTML pages in a build folder, without the base URL: /x for x/index.html,
// /x.html for any other HTML file. The 404 page and the redirect pages written here are
// not routes.
export function builtRoutes(outDir) {
  const routes = new Set();
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.name.endsWith('.html')) {
        const rel = path.relative(outDir, full).split(path.sep).join('/');
        if (rel === '404.html' || fs.readFileSync(full, 'utf8').includes('content="published-routes redirect"')) continue;
        routes.add(rel.endsWith('index.html') ? '/' + rel.slice(0, -'index.html'.length).replace(/\/$/, '') : '/' + rel);
      }
    }
  };
  walk(outDir);
  return routes;
}

function readRoutes() {
  if (!fs.existsSync(ROUTES_FILE)) return new Set();
  return new Set(
    fs.readFileSync(ROUTES_FILE, 'utf8').split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('#')),
  );
}

function writeRoutes(routes) {
  const header =
    '# Page routes the site has published, without the base URL. Each one must stay a page\n' +
    '# or have an entry in redirects.json. `npm run routes:update` adds the routes of a build;\n' +
    '# a route is removed from this file only when the page is retired without a successor.\n';
  fs.writeFileSync(ROUTES_FILE, header + [...routes].sort().join('\n') + '\n');
}

function redirectPage(target) {
  const t = target.replace(/"/g, '&quot;');
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Moved</title>
<meta name="robots" content="noindex">
<meta name="generator" content="published-routes redirect">
<link rel="canonical" href="${t}">
<meta http-equiv="refresh" content="0; url=${t}">
<script>location.replace(${JSON.stringify(target)} + location.search + location.hash);</script>
</head><body><p>This page has moved to <a href="${t}">${t}</a>.</p></body></html>
`;
}

export default function publishedRoutes(context) {
  return {
    name: 'published-routes',
    async postBuild({outDir}) {
      const base = context.baseUrl.replace(/\/$/, '');
      const built = builtRoutes(outDir);
      const redirects = fs.existsSync(REDIRECTS_FILE) ? JSON.parse(fs.readFileSync(REDIRECTS_FILE, 'utf8')) : [];
      const errors = [];
      const redirected = new Set();
      for (const {from, to} of redirects) {
        const [toPath] = to.split('#');
        if (built.has(from)) errors.push(`redirects.json: ${from} is still a page, so it cannot redirect`);
        else if (!built.has(toPath.replace(/\/$/, '') || '/')) errors.push(`redirects.json: ${from} redirects to ${to}, which is not a page`);
        else {
          const file = from.endsWith('.html') ? path.join(outDir, from) : path.join(outDir, from, 'index.html');
          fs.mkdirSync(path.dirname(file), {recursive: true});
          fs.writeFileSync(file, redirectPage(base + to));
          redirected.add(from);
        }
      }
      const published = readRoutes();
      const missing = [...published].filter((r) => !built.has(r) && !redirected.has(r)).sort();
      for (const r of missing) errors.push(`${r} was published and is no longer a page; add a redirect in redirects.json`);
      const unlisted = [...built].filter((r) => !published.has(r)).sort();
      if (unlisted.length) {
        const message = `${unlisted.length} new route(s) are not in published-routes.txt; run \`npm run routes:update\`:\n  ` + unlisted.join('\n  ');
        if (process.env.CI) errors.push(message);
        else console.warn(`[published-routes] ${message}`);
      }
      if (errors.length) throw new Error('[published-routes]\n  ' + errors.join('\n  '));
      console.log(`[published-routes] ${built.size} pages, ${redirected.size} redirects, all published routes resolve`);
    },
  };
}

// node plugins/published-routes.mjs --update [build folder]
if (process.argv[1] === fileURLToPath(import.meta.url) && process.argv[2] === '--update') {
  const outDir = path.resolve(process.argv[3] || path.join(siteDir, 'build'));
  const routes = new Set([...readRoutes(), ...builtRoutes(outDir)]);
  writeRoutes(routes);
  console.log(`published-routes.txt: ${routes.size} routes`);
}
