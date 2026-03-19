/**
 * Core search types: result shapes, options, and the shared engine interface.
 */

export interface SearchResult {
  /** Integer node ID (HNSW insertion order) or string document ID (flat mode) */
  id: number | string
  /** Document identifier from the source data (e.g. "WPS9999") */
  idno?: string
  /** Cosine similarity (semantic) or combined hybrid score in [0, 1] */
  score: number
  title: string
  /** Abstract / body text (present in flat mode; absent in HNSW-only results) */
  text?: string
  abstract?: string
  type?: string
  type_extra?: string
  sub_title?: string
  doi?: string
  url?: string
  geographic_coverage?: GeographicCoverage[]
  time_coverage?: string
  source?: string[]
  /** Normalized semantic contribution (0–1) in hybrid mode */
  semanticScore?: number
  /** Normalized BM25 contribution (0–1) in hybrid mode */
  lexicalScore?: number
  /** Score from cross-encoder reranker (higher = more relevant) */
  rerank_score?: number
  /** Allow arbitrary additional preview fields from the pipeline */
  [key: string]: unknown
}

export type GeographicCoverage =
  | string
  | { title?: string; name?: string; type?: string; [key: string]: unknown }

export interface SearchStats {
  /** Wall-clock milliseconds for the entire search() call */
  latencyMs: number
  /** New shard files fetched during this query (0 = fully cached) */
  shardsLoaded: number
  /** Total shards currently held in the worker's in-memory Map */
  totalCachedShards: number
}

export type SearchMode = 'semantic' | 'lexical' | 'hybrid'

export interface SearchOptions {
  topK?: number
  /** HNSW beam width at layer 0 (higher = better recall, more shard fetches) */
  ef?: number
  /** HNSW beam width for upper-layer descent */
  ef_upper?: number
  /** Minimum cosine similarity threshold (flat mode) */
  threshold?: number
}

/**
 * Common interface implemented by both FlatEngine and HNSWEngine.
 * Allows search.worker.ts to operate on either engine without type narrowing.
 */
export interface SearchEngine {
  readonly ready: boolean
  search(queryVec: Float32Array, opts?: SearchOptions): Promise<SearchResult[]> | SearchResult[]
  lastStats: SearchStats | null
}

/**
 * A single item in the flat index (flat/embeddings.int8.json).
 * Contains the int8 quantized vector plus all preview fields.
 */
export interface FlatItem {
  id: string | number
  idno?: string
  title: string
  text: string
  scale: number
  /** Stored as plain number[] in the JSON; converted to Int8Array on load */
  qv: number[] | Int8Array
  type?: string
  [key: string]: unknown
}

/**
 * A node in a layer-0 shard file.
 */
export interface ShardNode {
  id: number
  scale: number
  qv: number[]
  neighbors: number[]
}

/**
 * Contents of index/layer0/shard_NNN.json.
 */
export interface Shard {
  shard_id: number
  nodes: ShardNode[]
}

/**
 * Cached node entry in HNSWEngine.nodeCache.
 * Extends ShardNode with typed Int8Array and upper-layer metadata.
 */
export interface NodeCacheEntry {
  id: number
  scale: number
  qv: Int8Array
  neighbors: number[]
  layers?: Record<string, number[]>
  max_layer?: number
  /** true once layer-0 neighbors have been populated from a shard load */
  _l0loaded?: boolean
}

/** [score, nodeId] tuple used internally by the HNSW beam search */
export type ScoredNode = [score: number, nodeId: number]

/**
 * A minimal interface for wink-bm25-text-search, enough to type HybridSearch
 * without a full declaration file for the library.
 */
export interface BM25Engine {
  /** Returns [[docIdx, score], ...] sorted by score descending */
  search(query: string, topK: number): [number, number][]
}
