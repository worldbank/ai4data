/**
 * Discriminated union types for the search worker message protocol.
 *
 * Using discriminated unions lets TypeScript narrow the message type in each
 * `case` branch of the message handler, eliminating unsafe `as` casts and
 * ensuring inbound/outbound messages stay in sync.
 *
 * Usage in main thread:
 *   worker.postMessage({ type: 'init', manifestUrl: '...' } satisfies WorkerInboundMessage)
 *
 * Usage in worker:
 *   self.postMessage({ type: 'ready', mode: 'hnsw', config: manifest } satisfies WorkerOutboundMessage)
 */

import type { CollectionManifest } from './manifest'
import type { SearchResult, SearchStats } from './search'

// ── Inbound messages (main thread → worker) ──────────────────────────────────

export type WorkerInboundMessage =
  | WorkerInitMessage
  | WorkerSearchMessage
  | WorkerEmbedMessage
  | WorkerPingMessage
  | WorkerGetRecentMessage
  | WorkerGetHighlightsMessage
  | WorkerSearchCompareMessage

export interface WorkerInitMessage {
  type: 'init'
  /** Must be an absolute URL — resolve with new URL(url, location.href).href before posting */
  manifestUrl: string
  /** HuggingFace model ID, defaults to avsolatorio/GIST-small-Embedding-v0 */
  modelId?: string
  /** If true, skip loading the embedding model (for testing BM25 fallback). Index + BM25 still load. */
  skipModelLoad?: boolean
  /** Delay (seconds) before starting to load the embedding model; index + BM25 load first (for testing). */
  modelLoadDelaySeconds?: number
  /** If true, do not load or build BM25 even when the manifest has bm25_corpus (semantic-only). */
  skipBm25?: boolean
}

export interface WorkerSearchMessage {
  type: 'search'
  text: string
  topK?: number
  ef?: number
  ef_upper?: number
  threshold?: number
  mode?: 'semantic' | 'lexical' | 'hybrid'
}

export interface WorkerEmbedMessage {
  type: 'embed'
  text: string
}

export interface WorkerPingMessage {
  type: 'ping'
}

export interface WorkerGetRecentMessage {
  type: 'getRecent'
  limit?: number
}

export interface WorkerGetHighlightsMessage {
  type: 'getHighlights'
  limit?: number
}

export interface WorkerSearchCompareMessage {
  type: 'searchCompare'
  text: string
  topK?: number
  ef?: number
  ef_upper?: number
}

// ── Outbound messages (worker → main thread) ─────────────────────────────────

export type WorkerOutboundMessage =
  | WorkerProgressMessage
  | WorkerIndexReadyMessage
  | WorkerReadyMessage
  | WorkerResultsMessage
  | WorkerEmbeddingMessage
  | WorkerPongMessage
  | WorkerLoadingMessage
  | WorkerRecentMessage
  | WorkerHighlightsMessage
  | WorkerCompareMessage
  | WorkerErrorMessage

export interface WorkerProgressMessage {
  type: 'progress'
  phase: 'model' | 'index'
  message: string
}

/**
 * Sent when the index files + BM25 corpus are loaded.
 * Lexical search is available from this point on, even if the embedding model
 * is still downloading (BM25 fallback).
 */
export interface WorkerIndexReadyMessage {
  type: 'index_ready'
  bm25Ready: boolean
}

/**
 * Sent when the index is ready; if the embedding model was loaded, semantic/hybrid are available.
 * When skipModelLoad was used, modelLoaded is false and only lexical (BM25) runs for semantic/hybrid.
 */
export interface WorkerReadyMessage {
  type: 'ready'
  mode: 'flat' | 'hnsw'
  config: CollectionManifest
  /** false when init was called with skipModelLoad (embedding model not loaded). */
  modelLoaded?: boolean
}

export interface WorkerResultsMessage {
  type: 'results'
  data: SearchResult[]
  stats?: SearchStats | null
  /**
   * true when the result was produced via BM25 fallback because the embedding
   * model was not yet ready (requested mode was semantic or hybrid).
   */
  fallback?: boolean
}

export interface WorkerEmbeddingMessage {
  type: 'embedding'
  /** Transferred as ArrayBuffer for zero-copy */
  data: Float32Array
}

export interface WorkerPongMessage {
  type: 'pong'
}

export interface WorkerLoadingMessage {
  type: 'loading'
}

export interface WorkerRecentMessage {
  type: 'recent'
  data: SearchResult[]
}

export interface WorkerHighlightsMessage {
  type: 'highlights'
  data: SearchResult[]
}

export interface WorkerCompareMessage {
  type: 'compare'
  hnsw: SearchResult[]
  flat: SearchResult[]
  recall: number
  overlap: number
  k: number
}

export interface WorkerErrorMessage {
  type: 'error'
  message: string
  originalType?: string
}
