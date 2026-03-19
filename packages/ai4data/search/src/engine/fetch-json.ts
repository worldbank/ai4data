/**
 * fetch-json.ts
 *
 * Utility for fetching JSON (plain or gzip-compressed) with optional
 * Cache Storage read/write so repeat cold-starts skip the network.
 */

export interface FetchJsonOptions {
  /**
   * When provided, the response is read from (and written to) a named
   * Cache Storage bucket.  Pass `null` to disable caching entirely.
   */
  cacheName?: string | null
}

/**
 * Fetch a JSON resource, transparently handling gzip-compressed responses.
 *
 * Caching behaviour:
 *  1. If `cacheName` is set and the Cache API is available, attempt a cache hit.
 *  2. On a miss, fetch from the network.
 *  3. Decompress if the URL ends with `.gz` and the server did not already
 *     decompress it (i.e. `Content-Encoding` is absent or non-gzip).
 *  4. Write the parsed object back to Cache Storage for future requests.
 *
 * @param url  - Absolute or relative URL to fetch
 * @param opts - Optional caching configuration
 * @returns Parsed JSON payload cast to `T`
 * @throws {Error} On non-2xx HTTP responses
 */
export async function fetchJson<T = unknown>(
  url: string,
  opts?: FetchJsonOptions,
): Promise<T> {
  const cacheName = opts?.cacheName ?? null;
  const isGz = url.endsWith('.gz');

  // --- cache read ---
  if (cacheName && typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(cacheName);
      const cached = await cache.match(url);
      if (cached) {
        return cached.json() as Promise<T>;
      }
    } catch (_) {
      // Cache API unavailable or denied — fall through to network fetch
    }
  }

  // --- network fetch ---
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`fetchJson: HTTP ${resp.status}: ${url}`);
  }

  let data: T;

  if (isGz) {
    const encoding = resp.headers.get('Content-Encoding');
    if (encoding === 'gzip' || encoding === 'x-gzip') {
      // The server already decompressed it for us
      data = (await resp.json()) as T;
    } else {
      // Decompress manually in the browser via DecompressionStream
      const stream = resp.body!.pipeThrough(new DecompressionStream('gzip'));
      data = (await new Response(stream).json()) as T;
    }
  } else {
    data = (await resp.json()) as T;
  }

  // --- cache write ---
  if (cacheName && typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(cacheName);
      cache.put(
        url,
        new Response(JSON.stringify(data), {
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    } catch (_) {
      // Best-effort — ignore write failures
    }
  }

  return data;
}
