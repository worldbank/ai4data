/**
 * shard-loader.ts
 *
 * Loads layer-0 shard files on demand, deduplicates in-flight requests,
 * and maintains a bounded in-memory cache to limit worker heap growth.
 */

import type { Shard } from '../types/search'
import { fetchJson } from './fetch-json'

/**
 * Loads, deduplicates, and caches HNSW layer-0 shard files.
 *
 * Shards are stored in files named `shard_NNN<suffix>` (e.g. `shard_007.json`)
 * under a common base URL.  The loader combines three levels of caching:
 *
 *  1. In-memory `Map` — fastest; survives across queries within the same worker.
 *  2. In-flight deduplication — a second caller for the same shard awaits the
 *     already-running fetch rather than issuing a duplicate request.
 *  3. Cache Storage (via `fetchJson`) — survives page reloads.
 */
export class ShardLoader {
  /** Base URL for shard files (always ends with `/`) */
  readonly baseUrl: string
  /** Cache Storage bucket name passed through to `fetchJson` */
  readonly cacheName: string
  /** File extension appended to each shard filename (e.g. `.json` or `.json.gz`) */
  readonly shardSuffix: string

  /** In-memory shard cache keyed by numeric shard ID */
  memoryCache: Map<number, Shard>
  /** Promises for shards currently being fetched, keyed by numeric shard ID */
  inflight: Map<number, Promise<Shard>>
  /** Insertion-order log used by `evict()` to expire the oldest entries first */
  _insertOrder: number[]

  constructor(
    baseUrl: string,
    cacheName = 'hnsw-shards-v1',
    shardSuffix = '.json',
  ) {
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/'
    this.cacheName = cacheName
    this.shardSuffix = shardSuffix
    this.memoryCache = new Map()
    this.inflight = new Map()
    this._insertOrder = []
  }

  /**
   * Load a shard by ID, returning the in-memory copy if already cached,
   * joining an in-flight fetch if one exists, or issuing a new network request.
   *
   * @param shardId - Numeric shard identifier
   * @returns Parsed shard data
   */
  async load(shardId: number): Promise<Shard> {
    if (this.memoryCache.has(shardId)) {
      return this.memoryCache.get(shardId)!
    }
    if (this.inflight.has(shardId)) {
      return this.inflight.get(shardId)!
    }

    const promise = this._fetchShard(shardId)
    this.inflight.set(shardId, promise)

    try {
      const data = await promise
      this.memoryCache.set(shardId, data)
      this._insertOrder.push(shardId)
      return data
    } finally {
      this.inflight.delete(shardId)
    }
  }

  /**
   * Kick off background loads for a set of shard IDs without awaiting them.
   * Useful for prefetching neighbours that will likely be needed soon.
   *
   * @param shardIds - Shard IDs to prefetch
   */
  prefetch(shardIds: number[]): void {
    for (const sid of shardIds) {
      if (!this.memoryCache.has(sid) && !this.inflight.has(sid)) {
        this.load(sid)
      }
    }
  }

  /**
   * Evict the oldest in-memory shard entries to keep heap usage bounded.
   *
   * @param maxEntries - Maximum number of shards to keep (default: 200)
   */
  evict(maxEntries = 200): void {
    while (this._insertOrder.length > maxEntries) {
      const oldest = this._insertOrder.shift()!
      this.memoryCache.delete(oldest)
    }
  }

  /**
   * Build the URL for a given shard ID.
   *
   * @param shardId - Numeric shard identifier
   * @returns Full URL string
   */
  private _shardUrl(shardId: number): string {
    return this.baseUrl + `shard_${String(shardId).padStart(3, '0')}${this.shardSuffix}`
  }

  /**
   * Fetch and parse a shard file from the network (or Cache Storage).
   *
   * @param shardId - Numeric shard identifier
   * @returns Parsed shard data
   */
  private _fetchShard(shardId: number): Promise<Shard> {
    return fetchJson<Shard>(this._shardUrl(shardId), { cacheName: this.cacheName })
  }
}
