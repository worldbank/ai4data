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

### Custom worker path

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

## Demo

A minimal demo is included in this package. You need a **manifest URL** that points to a `manifest.json` for a search collection (from the Python index pipeline or compatible format).

From the package directory:

```bash
npm run demo
```

Then open **http://localhost:5173/demo/** in your browser, paste your manifest URL, click Connect, and run a search once the index is ready.

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
