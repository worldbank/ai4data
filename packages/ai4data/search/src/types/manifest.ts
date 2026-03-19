/**
 * Types for the collection manifest format produced by the Python pipeline.
 * The manifest.json file is the browser's entry point to a search collection.
 */

export interface FlatIndexConfig {
  /** Relative path to the flat brute-force index, e.g. "flat/embeddings.int8.json" */
  path: string
}

export interface HNSWIndexConfig {
  /** Directory prefix for all HNSW index files, e.g. "index/" */
  path: string
  /** Relative path to index/config.json */
  config: string
  /** Relative path to index/upper_layers.json */
  upper_layers?: string
  /** Relative path to index/node_to_shard.json */
  node_to_shard?: string
  /** Relative path to index/titles.json (display metadata, no vectors) */
  titles?: string
  /** Relative path to index/cluster_centroids.json */
  cluster_centroids?: string
  /** Relative path to index/bm25_corpus.json (lightweight text-only corpus for BM25) */
  bm25_corpus?: string
}

export type SearchMode = 'flat' | 'hnsw'

export interface ManifestThresholds {
  /** Maximum n_items for flat (brute-force) mode; above this HNSW is used */
  flat_max: number
}

/**
 * Top-level manifest written by 03_build_index.py.
 * The browser worker fetches this URL first to determine the search mode.
 */
export interface CollectionManifest {
  version?: string
  collection_id: string
  n_items: number
  embedding_dim: number
  matryoshka_dim?: number | null
  quant?: string
  model_id: string
  search_mode: SearchMode
  /** Whether index files are gzip-compressed (.json.gz). Static hosts like GitHub Pages require false. */
  compressed: boolean
  flat?: FlatIndexConfig
  index?: HNSWIndexConfig
  thresholds?: ManifestThresholds
  /** Fields included in result metadata (e.g. ["idno","title","abstract","type","doi"]) */
  preview_fields?: string[]
  /** Fields used for BM25 lexical search */
  bm25_fields?: string[]
}

/**
 * Parsed HNSW config.json contents, loaded by HNSWEngine at init.
 */
export interface HNSWConfig {
  n_items: number
  dim: number
  matryoshka_dim: number | null
  quant: string
  hnsw_M: number
  hnsw_ef_construction: number
  n_layers: number
  n_clusters: number
  entry_node_id: number
  entry_layer: number
  recall_at_10: number
}

/**
 * A single node's upper-layer data from upper_layers.json.
 * Layer 0 neighbors are NOT here — they live in the shard files.
 */
export interface UpperLayerNode {
  max_layer: number
  scale: number
  qv: number[]
  /** Map of layer index (string) → array of neighbor node IDs */
  layers: Record<string, number[]>
}

/**
 * Contents of index/upper_layers.json.
 */
export interface UpperLayersData {
  max_layer: number
  entry_node_id: number
  nodes: Record<string, UpperLayerNode>
}

/**
 * A single entry in index/bm25_corpus.json.
 * Lightweight text corpus written by the pipeline for BM25 indexing.
 */
export interface BM25CorpusEntry {
  id: string | number
  title: string
  text: string
}
