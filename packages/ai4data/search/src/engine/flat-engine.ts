/**
 * flat-engine.ts
 *
 * Brute-force (flat) search engine backed by an Int8-quantized index.
 * Suitable for collections up to ~50 k documents where exact nearest-neighbour
 * search is fast enough without an ANN index structure.
 */

import type { FlatItem, SearchEngine, SearchOptions, SearchResult, SearchStats } from '../types/search'
import { dotProductMixed, toInt8Array } from './int8-codec'
import { fetchJson } from './fetch-json'

/** Shape of the JSON file loaded by `FlatEngine.load()` */
interface FlatIndexFile {
  dim: number
  items: FlatItem[]
}

/** Internal representation — `qv` is always an `Int8Array` after loading. Defined
 * explicitly (not via `Omit<FlatItem, 'qv'>`) to avoid `[key: string]: unknown`
 * index-signature narrowing that makes named properties return `unknown`. */
interface LoadedFlatItem {
  id: string | number
  idno?: string
  title: string
  text: string
  scale: number
  qv: Int8Array
  type?: string
  [key: string]: unknown
}

/**
 * Brute-force semantic search engine.
 *
 * Usage:
 * ```ts
 * const engine = new FlatEngine()
 * await engine.load('/data/flat/embeddings.int8.json')
 * const results = engine.search(queryVec, { topK: 10 })
 * ```
 */
export class FlatEngine implements SearchEngine {
  /** Internal item list with Int8-converted vectors */
  private items: LoadedFlatItem[]

  /** True once `load()` has completed successfully */
  readonly ready: boolean = false

  /** Statistics from the most recent `search()` call, or `null` before first search */
  lastStats: SearchStats | null

  constructor() {
    this.items = []
    this.lastStats = null
  }

  /**
   * Fetch and parse the flat index file, converting all `qv` arrays to `Int8Array`.
   *
   * @param url - URL of the `embeddings.int8.json` index file
   * @returns The raw item list from the JSON (before Int8 conversion)
   */
  async load(url: string): Promise<FlatItem[]> {
    const data = await fetchJson<FlatIndexFile>(url)
    this.items = data.items.map(item => ({
      ...item,
      qv: toInt8Array(item.qv as number[]),
    }))
    ;(this as { ready: boolean }).ready = true
    return data.items
  }

  /**
   * Run a brute-force cosine-similarity search over all loaded items.
   *
   * @param queryVec - L2-normalised query embedding (Float32Array)
   * @param opts     - Optional search parameters
   * @returns Top-K results sorted by descending score
   * @throws {Error} If called before `load()` has completed
   */
  search(queryVec: Float32Array, opts?: SearchOptions): SearchResult[] {
    if (!this.ready) throw new Error('FlatEngine: not loaded yet')

    const topK = opts?.topK ?? 20
    const threshold = opts?.threshold ?? 0.0

    const t0 = Date.now()
    const scores = new Float32Array(this.items.length)

    for (let i = 0; i < this.items.length; i++) {
      const item = this.items[i]
      scores[i] = dotProductMixed(queryVec, item.qv, item.scale)
    }

    // Collect candidate indices above threshold
    const candidates: number[] = []
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] >= threshold) candidates.push(i)
    }

    candidates.sort((a, b) => scores[b] - scores[a])

    const results = candidates.slice(0, topK).map(i => {
      const item = this.items[i]
      // Include all preview fields except internal-only ones
      const extra: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(item)) {
        if (!['id', 'scale', 'qv', 'title', 'text'].includes(k)) {
          extra[k] = v
        }
      }
      return {
        id: item.id,
        score: scores[i],
        title: item.title,
        text: item.text,
        ...extra,
      } as SearchResult
    })

    this.lastStats = {
      latencyMs: Date.now() - t0,
      shardsLoaded: 0,
      totalCachedShards: 0,
    }

    return results
  }
}
