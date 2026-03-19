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

Two demos are included:

1. **Local build** (uses the built package from `dist/`): run `npm run demo`, then open **http://localhost:5173/demo/**.
2. **Standalone HTML** (loads the package from npm via [esm.sh](https://esm.sh)): open **demo/standalone.html** in a browser. Serve the file over HTTP (e.g. from the package directory run `npx serve .` and open http://localhost:3000/demo/standalone.html). No build step; you must use **workerFactory** so the worker is loaded from the CDN (see the file for the pattern).

Both demos need a **manifest URL** that points to a `manifest.json` for a search collection (produce one with the pipeline above, or use an existing hosted collection).

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
