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

/** Injected at build time from package.json version; fallback for source builds */
declare const __PACKAGE_VERSION__: string | undefined
const DEFAULT_CDN_WORKER_URL = `https://unpkg.com/@ai4data/search@${typeof __PACKAGE_VERSION__ !== 'undefined' ? __PACKAGE_VERSION__ : '0.0.0'}/dist/worker.mjs`

// ── Types ─────────────────────────────────────────────────────────────────────

export type SearchMode = 'semantic' | 'lexical' | 'hybrid'

export interface SearchClientOptions {
  /** HuggingFace model ID to use for embeddings (default: avsolatorio/GIST-small-Embedding-v0) */
  modelId?: string
  /** If true, skip loading the embedding model (for testing BM25 fallback). */
  skipModelLoad?: boolean
  /** Delay (seconds) before loading the embedding model; index + BM25 load first (for testing). */
  modelLoadDelaySeconds?: number
  /** If true, do not load or build BM25 even when the manifest has bm25_corpus (semantic-only). */
  skipBm25?: boolean
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
  /**
   * URL to the worker script (e.g. from a CDN). Use this when loading the package from a CDN
   * so the worker is created via fetch + blob and works cross-origin. Ignored if workerFactory is set.
   *
   * @example
   * ```ts
   * const client = await SearchClient.fromCDN(manifestUrl, {
   *   workerUrl: 'https://esm.sh/@ai4data/search@0.1.0/worker'
   * })
   * ```
   */
  workerUrl?: string
}

type MessageHandler<T extends WorkerOutboundMessage['type']> = (
  msg: Extract<WorkerOutboundMessage, { type: T }>,
) => void

/**
 * Create a Worker from a cross-origin URL by fetching the script and instantiating
 * from a blob URL. Use this when loading the package from a CDN so the worker is
 * same-origin and browsers allow it.
 *
 * @param url - Full URL to the worker script (e.g. https://esm.sh/@ai4data/search@0.1.0/worker)
 * @returns Promise that resolves with the Worker instance
 */
/**
 * Resolve a CDN worker URL to one that returns a single script (no import statements).
 * ESM.sh returns a wrapper with imports that fail when run from a blob: URL; unpkg/jsDelivr
 * serve the raw dist/worker.mjs which is self-contained.
 */
function getBundledWorkerUrl(url: string): string {
  const esmMatch = url.match(/esm\.sh\/@ai4data\/search@([^/]+)\/worker/)
  if (esmMatch) {
    const version = esmMatch[1].split('?')[0]
    return `https://unpkg.com/@ai4data/search@${version}/dist/worker.mjs`
  }
  const jdelivrMatch = url.match(/cdn\.jsdelivr\.net\/npm\/@ai4data\/search@([^/]+)\//)
  if (jdelivrMatch) return url
  return url
}

export function createWorkerFromUrl(url: string): Promise<Worker> {
  const fetchUrl = getBundledWorkerUrl(url)
  return fetch(fetchUrl, { mode: 'cors' })
    .then((r) => {
      if (r.status === 404)
        throw new Error(
          `Worker not found (404). Version may not be published to npm yet. Publish with: npm publish --access public (from packages/ai4data/search). Or use a published version in workerUrl.`
        )
      if (!r.ok) throw new Error(`Failed to fetch worker: ${r.status} ${r.statusText}`)
      return r.text()
    })
    .then((code) => {
      const trimmed = code.trim()
      if (trimmed.startsWith('<!') || trimmed.startsWith('<html'))
        throw new Error('Worker URL returned HTML (likely 404 or error page). Check the worker URL and that the package is published.')
      if (trimmed.length < 1000)
        throw new Error(`Worker script too short (${trimmed.length} chars). Expected a bundled script. Check the worker URL.`)
      // Reject wrapper scripts that would fail from blob (imports like /node/... don't resolve from blob:)
      if (/^\s*import\s+/.test(trimmed))
        throw new Error(
          'Worker URL returned a wrapper with import statements; it cannot run from a blob. Use a CDN that serves the raw bundle (e.g. unpkg.com/@ai4data/search@VERSION/dist/worker.mjs).'
        )
      const blob = new Blob([code], { type: 'application/javascript' })
      const blobUrl = URL.createObjectURL(blob)
      try {
        return new Worker(blobUrl, { type: 'module' })
      } catch (e) {
        URL.revokeObjectURL(blobUrl)
        throw e instanceof Error ? e : new Error(String(e))
      }
    })
}

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
    if (opts.workerFactory) {
      this.worker = opts.workerFactory()
    } else if (opts.workerUrl) {
      throw new Error(
        'SearchClient: workerUrl is only supported with SearchClient.fromCDN(). Use fromCDN(manifestUrl, { workerUrl }) or pass workerFactory.'
      )
    } else {
      this.worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' })
    }

    this.worker.onmessage = (e: MessageEvent<WorkerOutboundMessage>) => {
      this._handleMessage(e.data)
    }

    this.worker.onerror = (err: ErrorEvent) => {
      const e = err as ErrorEvent & { error?: Error }
      const msg =
        e.error?.message ??
        e.message ??
        (e as unknown as { message?: string }).message ??
        'Worker error (no details; check DevTools Console for the worker context or Network tab for failed requests)'
      const filename = e.filename || ''
      const lineno = e.lineno ?? ''
      console.error('[SearchClient] Worker error:', msg, filename ? `at ${filename}` : '', lineno ? `:${lineno}` : '', e.error ? e.error.stack : '')
      this.loadingMessage = `Search worker error: ${msg}`
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
      skipBm25: opts.skipBm25,
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
   * Fetch curated highlight items from `index/highlights.json` when the manifest
   * sets `index.highlights`. Listen with `client.on('highlights', ({ data }) => …)`.
   * If the file is missing or empty, `data` is `[]`.
   */
  getHighlights(limit = 10): void {
    this.worker.postMessage({ type: 'getHighlights', limit } satisfies WorkerInboundMessage)
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

  /**
   * Create a SearchClient when loading the package from a CDN. Fetches the worker
   * script and creates the worker from a blob URL so it works cross-origin.
   *
   * @param manifestUrl - URL to the collection manifest.json
   * @param opts - Options; workerUrl defaults to unpkg for this package version
   * @returns Promise that resolves with the SearchClient
   *
   * @example
   * ```ts
   * const client = await SearchClient.fromCDN('https://example.com/data/manifest.json')
   * client.on('results', ({ data }) => console.log(data))
   * ```
   */
  static fromCDN(
    manifestUrl: string,
    opts: SearchClientOptions & { workerUrl?: string } = {},
  ): Promise<SearchClient> {
    const { workerUrl = DEFAULT_CDN_WORKER_URL, ...rest } = opts
    return createWorkerFromUrl(workerUrl).then((worker) => {
      return new SearchClient(manifestUrl, { ...rest, workerFactory: () => worker })
    })
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
