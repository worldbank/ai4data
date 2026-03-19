/**
 * hybrid-search.ts
 *
 * Combines semantic (HNSW / flat) and lexical (BM25) search results using
 * min-max normalisation and a configurable linear blend.
 */

import type { SearchEngine, BM25Engine, SearchResult, SearchOptions } from '../types/search'

/** Options accepted by `HybridSearch.search()` */
export interface HybridSearchOptions {
  /** Number of results to return (default: 20) */
  topK?: number
  /** Weight applied to normalised semantic scores (default: 0.7) */
  semanticWeight?: number
  /** Weight applied to normalised BM25 scores (default: 0.3) */
  lexicalWeight?: number
  /** HNSW beam width forwarded to the semantic engine (default: 50) */
  ef?: number
  /** Search mode: `'semantic'`, `'lexical'`, or `'hybrid'` (default: `'hybrid'`) */
  mode?: 'semantic' | 'lexical' | 'hybrid'
}

/** Extended result type used internally during score merging */
interface MergedResult extends SearchResult {
  semanticScore: number
  lexicalScore: number
  rawSemanticScore: number
}

/**
 * Hybrid search combining a semantic vector engine and an optional BM25 engine.
 *
 * In `'hybrid'` mode both engines are queried in parallel; scores are
 * min-max normalised independently and then linearly blended.
 *
 * Example:
 * ```ts
 * const hybrid = new HybridSearch(hnswEngine, bm25Engine, id => titlesMap[id])
 * const results = await hybrid.search(queryVec, 'development finance', { topK: 10 })
 * ```
 */
export class HybridSearch {
  private readonly semantic: SearchEngine
  private readonly bm25: BM25Engine | null
  private readonly idToMeta: ((id: number | string) => Partial<SearchResult>) | null

  /**
   * @param semanticEngine - Initialised `SearchEngine` (FlatEngine or HNSWEngine)
   * @param bm25Engine     - Optional BM25 engine; pass `null` to disable lexical search
   * @param idToMeta       - Optional callback to look up display metadata by document ID
   */
  constructor(
    semanticEngine: SearchEngine,
    bm25Engine: BM25Engine | null = null,
    idToMeta: ((id: number | string) => Partial<SearchResult>) | null = null,
  ) {
    this.semantic = semanticEngine
    this.bm25 = bm25Engine
    this.idToMeta = idToMeta
  }

  /**
   * Run a hybrid (or single-mode) search query.
   *
   * @param queryVec  - L2-normalised query embedding, or `null` for lexical-only mode
   * @param queryText - Raw query string for BM25, or empty string for semantic-only mode
   * @param opts      - Search options
   * @returns Top-K results sorted by descending combined score
   */
  async search(
    queryVec: Float32Array | null,
    queryText: string,
    opts?: HybridSearchOptions,
  ): Promise<SearchResult[]> {
    const topK = opts?.topK ?? 20
    const semanticWeight = opts?.semanticWeight ?? 0.7
    const lexicalWeight = opts?.lexicalWeight ?? 0.3
    const ef = opts?.ef ?? 50
    const mode = opts?.mode ?? 'hybrid'

    const candidateK = topK * 3

    const searchOpts: SearchOptions = { topK: candidateK, ef }

    const [semanticResults, lexicalResults] = await Promise.all([
      mode !== 'lexical' && queryVec && this.semantic
        ? Promise.resolve(this.semantic.search(queryVec, searchOpts))
        : Promise.resolve([]),
      mode !== 'semantic' && this.bm25 && queryText
        ? Promise.resolve(this._runBM25(queryText, candidateK))
        : Promise.resolve([]),
    ])

    // Unwrap potential Promise from synchronous search implementations
    const semResults = await semanticResults
    const lexResults = await lexicalResults

    if (mode === 'semantic') return this._formatResults(semResults, topK, 'semantic')
    if (mode === 'lexical') return this._formatResults(lexResults, topK, 'lexical')

    // --- Merge & blend ---
    const scoreMap = new Map<string, MergedResult>()

    if (semResults.length > 0) {
      const maxSem = semResults[0].score || 1
      const minSem = semResults[semResults.length - 1].score || 0
      const rangeSem = maxSem - minSem || 1
      for (const r of semResults) {
        const normScore = (r.score - minSem) / rangeSem
        scoreMap.set(String(r.id), {
          ...r,
          title: r.title,
          semanticScore: normScore,
          lexicalScore: 0,
          rawSemanticScore: r.score,
        })
      }
    }

    if (lexResults.length > 0) {
      const maxBm25 = lexResults[0].score || 1
      const minBm25 = lexResults[lexResults.length - 1].score || 0
      const rangeBm25 = maxBm25 - minBm25 || 1
      for (const r of lexResults) {
        const normScore = (r.score - minBm25) / rangeBm25
        const idStr = String(r.id)
        if (scoreMap.has(idStr)) {
          scoreMap.get(idStr)!.lexicalScore = normScore
        } else {
          const meta = this.idToMeta ? this.idToMeta(r.id) : {}
          scoreMap.set(idStr, {
            ...r,
            ...meta,
            id: r.id,
            title: meta.title ?? r.title ?? '',
            text: meta.text ?? r.text ?? '',
            semanticScore: 0,
            lexicalScore: normScore,
            rawSemanticScore: 0,
          })
        }
      }
    }

    const merged = [...scoreMap.values()].map(r => ({
      ...r,
      score: semanticWeight * r.semanticScore + lexicalWeight * r.lexicalScore,
    }))

    merged.sort((a, b) => b.score - a.score)
    return merged.slice(0, topK)
  }

  /**
   * Run the BM25 engine and map raw `[docIdx, score]` tuples to `SearchResult` objects.
   *
   * @param queryText - Raw query string
   * @param topK      - Maximum number of results to return
   * @returns BM25 results as `SearchResult` objects (score order: descending)
   */
  private _runBM25(queryText: string, topK: number): SearchResult[] {
    if (!this.bm25) return []
    try {
      const raw = this.bm25.search(queryText, topK)
      return raw.map(([docIdx, score]) => {
        const meta = this.idToMeta ? this.idToMeta(docIdx) : {}
        return {
          ...meta,
          id: docIdx,
          score,
          title: meta.title ?? '',
          text: meta.text ?? '',
        } as SearchResult
      })
    } catch (e) {
      console.warn('BM25 search error:', e)
      return []
    }
  }

  /**
   * Format single-mode results, adding the appropriate `semanticScore` /
   * `lexicalScore` fields expected by callers.
   *
   * @param results - Raw results from one engine
   * @param topK    - Slice limit
   * @param source  - Which engine produced the results
   * @returns Results annotated with zeroed-out score fields for the unused engine
   */
  private _formatResults(
    results: SearchResult[],
    topK: number,
    source: 'semantic' | 'lexical',
  ): SearchResult[] {
    return results.slice(0, topK).map(r => ({
      ...r,
      semanticScore: source === 'semantic' ? (r.score ?? 0) : 0,
      lexicalScore: source === 'lexical' ? (r.score ?? 0) : 0,
    }))
  }
}
