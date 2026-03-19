/**
 * @ai4data/search
 *
 * Framework-agnostic semantic search client.
 * Combines HNSW approximate nearest-neighbour search, BM25 lexical search,
 * and hybrid ranking — all running in a Web Worker.
 *
 * @example Vanilla JS / any framework
 * ```ts
 * import { SearchClient } from '@ai4data/search'
 *
 * const client = new SearchClient('https://example.com/data/prwp/manifest.json')
 * client.on('index_ready', () => client.search('climate finance'))
 * client.on('results', ({ data }) => console.log(data))
 * ```
 *
 * Vue and React adapters are planned (future: @ai4data/search/vue, @ai4data/search/react).
 */

// ── Core client ────────────────────────────────────────────────────────────────
export { SearchClient } from './client/SearchClient'
export type { SearchClientOptions, SearchMode } from './client/SearchClient'

// ── Types ──────────────────────────────────────────────────────────────────────
export type {
  SearchResult,
  SearchStats,
  SearchOptions,
  SearchEngine,
  GeographicCoverage,
  FlatItem,
  ShardNode,
  Shard,
  BM25Engine,
} from './types/search'

export type {
  CollectionManifest,
  FlatIndexConfig,
  HNSWIndexConfig,
  HNSWConfig,
  BM25CorpusEntry,
} from './types/manifest'

export type {
  WorkerInboundMessage,
  WorkerOutboundMessage,
  WorkerInitMessage,
  WorkerSearchMessage,
  WorkerResultsMessage,
  WorkerReadyMessage,
  WorkerIndexReadyMessage,
  WorkerErrorMessage,
  WorkerProgressMessage,
} from './types/worker'

// ── Engine (advanced / direct use) ────────────────────────────────────────────
export { FlatEngine } from './engine/flat-engine'
export { HNSWEngine } from './engine/hnsw-engine'
export { HybridSearch } from './engine/hybrid-search'
export { fetchJson } from './engine/fetch-json'
export { dotProductMixed, l2NormalizeInPlace, toInt8Array, dequantize } from './engine/int8-codec'
