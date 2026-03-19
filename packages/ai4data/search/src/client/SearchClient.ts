/**
 * SearchClient — framework-agnostic wrapper around the search Web Worker.
 *
 * Works in any JavaScript environment that supports Web Workers.
 * Extend or wrap with framework adapters (see adapters/vue.ts, adapters/react.ts).
 *
 * @example
 * ```ts
 * const client = new SearchClient('https://example.com/data/prwp/manifest.json')
 *
 * client.on('index_ready', () => {
 *   client.search('climate finance', { topK: 10, mode: 'hybrid' })
 * })
 *
 * client.on('results', ({ data, stats }) => {
 *   console.log(data)   // SearchResult[]
 *   console.log(stats)  // SearchStats | null
 * })
 *
 * // Clean up when done
 * client.destroy()
 * ```
 */

import type { WorkerOutboundMessage, WorkerInboundMessage } from '../types/worker'
import type { CollectionManifest } from '../types/manifest'
import type { SearchOptions } from '../types/search'

// ── Types ─────────────────────────────────────────────────────────────────────

export type SearchMode = 'semantic' | 'lexical' | 'hybrid'

export interface SearchClientOptions {
  /** HuggingFace model ID to use for embeddings (default: avsolatorio/GIST-small-Embedding-v0) */
  modelId?: string
  /** If true, skip loading the embedding model (for testing BM25 fallback). */
  skipModelLoad?: boolean
  /** Delay (seconds) before loading the embedding model; index + BM25 load first (for testing). */
  modelLoadDelaySeconds?: number
  /**
   * Factory function that creates the Web Worker.
   * Defaults to the bundled search worker created via `new URL()`.
   * Override when you need a custom worker path (e.g. CDN, service worker proxy).
   *
   * @example
   * ```ts
   * // Vite / webpack 5 (recommended — bundler resolves the path)
   * new SearchClient(url, {
   *   workerFactory: () => new Worker(new URL('@ai4data/search/worker', import.meta.url), { type: 'module' })
   * })
   * ```
   */
  workerFactory?: () => Worker
}

type MessageHandler<T extends WorkerOutboundMessage['type']> = (
  msg: Extract<WorkerOutboundMessage, { type: T }>,
) => void

// ── SearchClient ──────────────────────────────────────────────────────────────

export class SearchClient {
  // ── Public state (plain properties — no reactivity) ──

  /** True once the index + BM25 corpus are loaded. Lexical search available. */
  isIndexReady = false

  /** True once the ONNX embedding model is ready. Semantic + hybrid search available. */
  isModelReady = false

  /** Latest progress/status message from the worker. */
  loadingMessage = 'Initializing…'

  /** True when the last search fell back to BM25 because the model wasn't ready. */
  activeFallback = false

  /** Parsed collection manifest, available after `index_ready`. */
  manifest: CollectionManifest | null = null

  // ── Private ──

  private readonly worker: Worker
  private readonly handlers = new Map<string, Set<(msg: WorkerOutboundMessage) => void>>()
  private destroyed = false

  // ── Constructor ──────────────────────────────────────────────────────────────

  /**
   * @param manifestUrl - Absolute or relative URL to `manifest.json`.
   *   Relative URLs are resolved against `location.href`.
   * @param opts        - Optional configuration.
   */
  constructor(manifestUrl: string, opts: SearchClientOptions = {}) {
    this.worker = opts.workerFactory
      ? opts.workerFactory()
      : new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' })

    this.worker.onmessage = (e: MessageEvent<WorkerOutboundMessage>) => {
      this._handleMessage(e.data)
    }

    this.worker.onerror = (err) => {
      console.error('[SearchClient] Worker error:', err)
      this.loadingMessage = 'Search worker error'
    }

    // Resolve relative manifest URLs against current page origin
    const resolvedUrl = new URL(
      manifestUrl,
      globalThis.location?.href ?? 'http://localhost/',
    ).href

    const initMsg: WorkerInboundMessage = {
      type: 'init',
      manifestUrl: resolvedUrl,
      modelId: opts.modelId,
      skipModelLoad: opts.skipModelLoad,
      modelLoadDelaySeconds: opts.modelLoadDelaySeconds,
    }
    this.worker.postMessage(initMsg)
  }

  // ── Event bus ─────────────────────────────────────────────────────────────────

  /**
   * Subscribe to a specific worker message type.
   * Returns an unsubscribe function — call it to remove the handler.
   *
   * @example
   * ```ts
   * const off = client.on('results', ({ data }) => setResults(data))
   * // later…
   * off()
   * ```
   */
  on<T extends WorkerOutboundMessage['type']>(
    type: T,
    handler: MessageHandler<T>,
  ): () => void {
    if (!this.handlers.has(type)) this.handlers.set(type, new Set())
    const bucket = this.handlers.get(type)!
    const h = handler as (msg: WorkerOutboundMessage) => void
    bucket.add(h)
    return () => bucket.delete(h)
  }

  // ── Actions ───────────────────────────────────────────────────────────────────

  /**
   * Submit a search query. No-op if the index is not yet ready.
   *
   * @param text - Natural-language query
   * @param opts - Optional topK, ef, mode ('semantic' | 'lexical' | 'hybrid')
   */
  search(text: string, opts: SearchOptions & { mode?: SearchMode } = {}): void {
    if (!this.isIndexReady) return
    this.worker.postMessage({
      type: 'search',
      text,
      topK: opts.topK ?? 20,
      ef: opts.ef ?? 50,
      ef_upper: opts.ef_upper ?? 2,
      threshold: opts.threshold ?? 0.0,
      mode: opts.mode ?? 'hybrid',
    } satisfies WorkerInboundMessage)
  }

  /**
   * Fetch the most-recent items from the index (useful for pre-search state).
   */
  getRecent(limit = 10): void {
    this.worker.postMessage({ type: 'getRecent', limit } satisfies WorkerInboundMessage)
  }

  /**
   * Ping the worker. Resolves when the worker responds with 'pong'.
   */
  ping(): Promise<void> {
    return new Promise((resolve) => {
      const off = this.on('pong', () => { off(); resolve() })
      this.worker.postMessage({ type: 'ping' } satisfies WorkerInboundMessage)
    })
  }

  /**
   * Terminate the worker and clean up all event listeners.
   * The client is unusable after this call.
   */
  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    this.worker.terminate()
    this.handlers.clear()
  }

  // ── Internal ──────────────────────────────────────────────────────────────────

  private _handleMessage(msg: WorkerOutboundMessage): void {
    // Update plain-property state
    switch (msg.type) {
      case 'progress':
        this.loadingMessage = msg.message
        break
      case 'index_ready':
        this.isIndexReady = true
        break
      case 'ready':
        this.isModelReady = msg.modelLoaded !== false
        this.manifest = msg.config
        break
      case 'results':
        this.activeFallback = msg.fallback ?? false
        break
      case 'error':
        this.isIndexReady = true  // exit loading state on error
        this.loadingMessage = `Error: ${msg.message}`
        console.error('[SearchClient] Worker error message:', msg.message)
        break
    }

    // Dispatch to registered handlers
    const bucket = this.handlers.get(msg.type)
    if (bucket) {
      for (const h of bucket) h(msg)
    }
  }
}
