/**
 * hnsw-engine.ts
 *
 * Approximate nearest-neighbour search using a pre-built HNSW index.
 * Upper layers (layers ≥ 1) are held entirely in memory; layer-0 is
 * loaded on demand from shard files via `ShardLoader`.
 */

import type { HNSWConfig, HNSWIndexConfig, UpperLayersData, CollectionManifest } from '../types/manifest'
import type {
  NodeCacheEntry,
  ScoredNode,
  SearchEngine,
  SearchResult,
  SearchStats,
  SearchOptions,
} from '../types/search'
import { dotProductMixed, toInt8Array } from './int8-codec'
import { ShardLoader } from './shard-loader'
import { fetchJson } from './fetch-json'

/** Options accepted by `HNSWEngine.init()` */
interface HNSWInitOptions {
  /** Cache Storage bucket name forwarded to `ShardLoader` and `fetchJson` */
  cacheName?: string
  /** Parsed manifest; used to resolve index file paths and compressed flag */
  manifest?: CollectionManifest | null
}

/**
 * Insert `item` into `arr` in ascending score order (binary search).
 * `arr[0]` is always the lowest-scoring element after insertion.
 *
 * @param arr  - Sorted array to insert into (mutated in place)
 * @param item - `[score, nodeId]` tuple to insert
 */
function _sortedInsert(arr: ScoredNode[], item: ScoredNode): void {
  const score = item[0]
  let lo = 0
  let hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (arr[mid][0] < score) lo = mid + 1
    else hi = mid
  }
  arr.splice(lo, 0, item)
}

/**
 * HNSW approximate nearest-neighbour search engine.
 *
 * Typical usage:
 * ```ts
 * const engine = new HNSWEngine()
 * await engine.init('/data/prwp/')
 * const results = await engine.search(queryVec, { topK: 10, ef: 50 })
 * ```
 */
export class HNSWEngine implements SearchEngine {
  /** Parsed `index/config.json` */
  private config: HNSWConfig | null
  /** Parsed `index/upper_layers.json` */
  private upperLayers: UpperLayersData | null
  /** Maps string node ID → shard ID */
  private nodeToShard: Record<string, number> | null
  /** Shard loader for layer-0 data */
  private loader: ShardLoader | null
  /** In-memory node cache (Int8 vectors, neighbours) */
  private nodeCache: Map<number, NodeCacheEntry>

  /** True once `init()` has completed successfully */
  readonly ready: boolean = false

  /** Statistics from the most recent `search()` call, or `null` before first search */
  lastStats: SearchStats | null

  constructor() {
    this.config = null
    this.upperLayers = null
    this.nodeToShard = null
    this.loader = null
    this.nodeCache = new Map()
    this.lastStats = null
  }

  /**
   * Load all index metadata and populate the upper-layer node cache.
   * This must be called (and awaited) before any call to `search()`.
   *
   * @param baseUrl - Base URL of the collection directory (e.g. `/data/prwp/`)
   * @param opts    - Optional cache name and manifest
   */
  async init(baseUrl: string, opts?: HNSWInitOptions): Promise<void> {
    const cacheName = opts?.cacheName ?? 'hnsw-shards-v1'
    const manifest = opts?.manifest ?? null

    const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/'
    const idx: Partial<HNSWIndexConfig> = manifest?.index ?? {}
    const configPath = base + (idx.config ?? 'index/config.json')
    const upperPath = base + (idx.upper_layers ?? 'index/upper_layers.json')
    const nodePath = base + (idx.node_to_shard ?? 'index/node_to_shard.json')
    const shardSuffix = manifest?.compressed ? '.json.gz' : '.json'

    const [config, upperLayers, nodeToShard] = await Promise.all([
      fetchJson<HNSWConfig>(configPath, { cacheName }),
      fetchJson<UpperLayersData>(upperPath, { cacheName }),
      fetchJson<Record<string, number>>(nodePath, { cacheName }),
    ])

    this.config = config
    this.upperLayers = upperLayers
    this.nodeToShard = nodeToShard
    this.loader = new ShardLoader(base + 'index/layer0/', cacheName, shardSuffix)

    // Pre-populate the node cache with upper-layer nodes
    for (const [idStr, node] of Object.entries(upperLayers.nodes)) {
      this.nodeCache.set(parseInt(idStr, 10), {
        id: parseInt(idStr, 10),
        scale: node.scale,
        qv: toInt8Array(node.qv),
        neighbors: [],
        layers: node.layers,
        max_layer: node.max_layer,
      })
    }

    ;(this as { ready: boolean }).ready = true  // bypass readonly for post-init assignment
  }

  /**
   * Search the HNSW index for the nearest neighbours of `queryVec`.
   *
   * @param queryVec - L2-normalised query embedding (Float32Array)
   * @param opts     - Optional search parameters (`topK`, `ef`, `ef_upper`)
   * @returns Top-K results sorted by descending score
   * @throws {Error} If called before `init()` has completed
   */
  async search(queryVec: Float32Array, opts?: SearchOptions): Promise<SearchResult[]> {
    if (!this.ready || !this.config || !this.upperLayers || !this.loader) {
      throw new Error('HNSWEngine: not initialized. Call init() first.')
    }

    const ef = opts?.ef ?? 50
    const ef_upper = opts?.ef_upper ?? 2
    const topK = opts?.topK ?? 10

    const t0 = Date.now()
    const prevCacheSize = this.loader.memoryCache.size

    // Greedy descent through upper layers to find good entry points for layer 0
    let entryPoints: ScoredNode[] = [
      [this._scoreUpperNode(queryVec, this.upperLayers.entry_node_id), this.upperLayers.entry_node_id],
    ]

    for (let layer = this.config.n_layers - 1; layer >= 1; layer--) {
      entryPoints = this._beamDescentLayer(queryVec, entryPoints, layer, ef_upper)
    }

    const results = await this._beamSearchLayer0(queryVec, entryPoints, ef)

    const shardsLoaded =
      this.loader.memoryCache.size - prevCacheSize + (this.loader.inflight.size > 0 ? 1 : 0)

    this.lastStats = {
      latencyMs: Date.now() - t0,
      shardsLoaded: Math.max(0, shardsLoaded),
      totalCachedShards: this.loader.memoryCache.size,
    }

    this.loader.evict(300)
    return results.slice(0, topK)
  }

  /**
   * Single-layer greedy beam descent for layers ≥ 1 (upper layers).
   * All nodes at these layers are already in `nodeCache`.
   *
   * @param queryVec    - L2-normalised query vector
   * @param entryPoints - Current best candidates as `[score, nodeId]` tuples
   * @param layer       - Layer index to traverse
   * @param ef_upper    - Beam width (number of candidates to keep)
   * @returns Updated candidate list for the next layer
   */
  private _beamDescentLayer(
    queryVec: Float32Array,
    entryPoints: ScoredNode[],
    layer: number,
    ef_upper: number,
  ): ScoredNode[] {
    const layerStr = String(layer)
    const seen = new Set<number>()
    const W: ScoredNode[] = []

    for (const [, nodeId] of entryPoints) {
      if (seen.has(nodeId)) continue
      seen.add(nodeId)
      const score = this._scoreUpperNode(queryVec, nodeId)
      _sortedInsert(W, [score, nodeId])
      if (W.length > ef_upper) W.shift()

      const node = this.nodeCache.get(nodeId)
      if (!node) continue

      const neighbors = node.layers?.[layerStr] ?? []
      for (const nid of neighbors) {
        if (seen.has(nid)) continue
        seen.add(nid)
        const s = this._scoreUpperNode(queryVec, nid)
        _sortedInsert(W, [s, nid])
        if (W.length > ef_upper) W.shift()
      }
    }

    return W
  }

  /**
   * Score a node that is present in `nodeCache` (upper-layer or already loaded layer-0).
   *
   * @param queryVec - L2-normalised query vector
   * @param nodeId   - Node to score
   * @returns Approximate dot-product similarity, or `-Infinity` if node is absent
   */
  private _scoreUpperNode(queryVec: Float32Array, nodeId: number): number {
    const node = this.nodeCache.get(nodeId)
    if (!node) return -Infinity
    return dotProductMixed(queryVec, node.qv, node.scale)
  }

  /**
   * Layer-0 beam search.  Loads shard files on demand as the search frontier expands.
   *
   * @param queryVec    - L2-normalised query vector
   * @param entryPoints - Entry candidates from upper-layer descent
   * @param ef          - Beam width (number of candidates to maintain in `W`)
   * @returns All candidates in `W` sorted by descending score as `SearchResult` objects
   */
  private async _beamSearchLayer0(
    queryVec: Float32Array,
    entryPoints: ScoredNode[],
    ef: number,
  ): Promise<SearchResult[]> {
    const visited = new Set<number>()
    let W: ScoredNode[] = []
    let C: ScoredNode[] = []

    for (const [, nodeId] of entryPoints) {
      if (visited.has(nodeId)) continue
      visited.add(nodeId)
      const node = await this._getLayer0Node(nodeId)
      if (!node) continue
      const s = dotProductMixed(queryVec, node.qv, node.scale)
      _sortedInsert(W, [s, nodeId])
      _sortedInsert(C, [s, nodeId])
    }

    if (W.length > ef) W = W.slice(-ef)
    if (C.length > ef) C = C.slice(-ef)

    while (C.length > 0) {
      const [cScore, cId] = C.pop()!
      const worstInW = W.length >= ef ? W[0][0] : -Infinity
      if (cScore < worstInW) break

      const cNode = await this._getLayer0Node(cId)
      if (!cNode) continue

      const unvisited = cNode.neighbors.filter(n => !visited.has(n))

      // Batch-prefetch all shards needed for unvisited neighbours
      const neededShards = new Set<number>()
      for (const nid of unvisited) {
        const sId = this.nodeToShard![String(nid)]
        if (sId != null && !this.loader!.memoryCache.has(sId)) {
          neededShards.add(sId)
        }
      }
      if (neededShards.size > 0) {
        await Promise.all([...neededShards].map(s => this.loader!.load(s)))
      }

      for (const nid of unvisited) {
        visited.add(nid)
        const nNode = await this._getLayer0Node(nid)
        if (!nNode) continue
        const score = dotProductMixed(queryVec, nNode.qv, nNode.scale)
        const currentWorst = W.length >= ef ? W[0][0] : -Infinity
        if (score > currentWorst || W.length < ef) {
          _sortedInsert(C, [score, nid])
          _sortedInsert(W, [score, nid])
          if (W.length > ef) W.shift()
        }
      }
    }

    return W
      .sort((a, b) => b[0] - a[0])
      .map(([score, id]) => ({ id, score, title: '' }))
  }

  /**
   * Retrieve a layer-0 node from cache, loading its shard file if necessary.
   * Once loaded, the node entry in `nodeCache` is augmented with `neighbors`
   * and `_l0loaded = true`.
   *
   * @param nodeId - Node to retrieve
   * @returns Fully populated cache entry, or `null` if the node cannot be found
   */
  private async _getLayer0Node(nodeId: number): Promise<NodeCacheEntry | null> {
    const cached = this.nodeCache.get(nodeId)
    if (cached?._l0loaded) return cached

    const shardId = this.nodeToShard![String(nodeId)]
    if (shardId == null) return this.nodeCache.get(nodeId) ?? null

    const shard = await this.loader!.load(shardId)

    for (const n of shard.nodes) {
      const existing = this.nodeCache.get(n.id)
      const entry: NodeCacheEntry = {
        ...(existing ?? {}),
        id: n.id,
        scale: n.scale,
        qv: existing?.qv ?? toInt8Array(n.qv),
        neighbors: n.neighbors,
        _l0loaded: true,
      }
      this.nodeCache.set(n.id, entry)
    }

    return this.nodeCache.get(nodeId) ?? null
  }
}
