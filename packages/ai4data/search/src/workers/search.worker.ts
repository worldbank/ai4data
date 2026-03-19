/**
 * search.worker.ts
 *
 * Unified Web Worker: embedding inference + search routing.
 *
 * Two-phase loading:
 *   Phase 1 (fast):  manifest + index files + bm25_corpus → build BM25 → post index_ready
 *   Phase 2 (slow):  ONNX model download → post ready
 *
 * Both phases start in parallel via Promise.all.
 */

// @ts-ignore — @xenova/transformers may have incomplete types
import { pipeline, env } from "@xenova/transformers";
// @ts-ignore — wink-bm25-text-search is a CommonJS default export
import winkBM25 from "wink-bm25-text-search";
import winkNLP from "wink-nlp";
// @ts-ignore — wink-eng-lite-web-model has no bundled type declaration
import model from "wink-eng-lite-web-model";
import { fetchJson } from "../engine/fetch-json";
import { l2NormalizeInPlace } from "../engine/int8-codec";
import { FlatEngine } from "../engine/flat-engine";
import { HNSWEngine } from "../engine/hnsw-engine";
import { HybridSearch } from "../engine/hybrid-search";
import type { CollectionManifest, BM25CorpusEntry } from "../types/manifest";
import type { SearchResult, SearchEngine } from "../types/search";
import type {
  WorkerOutboundMessage,
  WorkerInboundMessage,
} from "../types/worker";

// ── Transformers.js config (Xenova v2) ────────────────────────────────────────

env.allowRemoteModels = true;
// Skip local /models/ check so models load from Hugging Face Hub when serving from localhost.
env.allowLocalModels = false;

// ── Constants ─────────────────────────────────────────────────────────────────

const DEFAULT_MODEL = "avsolatorio/GIST-small-Embedding-v0";
const CACHE_NAME = "hnsw-shards-v1";

// ── State ─────────────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let extractor: any = null;
let manifest: CollectionManifest | null = null;
let baseUrl = "";
let searchEngine: SearchEngine | null = null;
let hybridEngine: HybridSearch | null = null;
let bm25Engine: ReturnType<typeof winkBM25> | null = null;
let bm25Corpus: BM25CorpusEntry[] | null = null;
let titlesMap: Record<string, Record<string, unknown>> | null = null;
let flatCompareEngine: FlatEngine | null = null;
let isModelReady = false;
let isIndexReady = false;

// ── Helpers ───────────────────────────────────────────────────────────────────

function postMsg(msg: WorkerOutboundMessage): void {
  self.postMessage(msg);
}

/**
 * Enrich HNSW results with metadata from titlesMap (keyed by HNSW node integer ID).
 * HNSW engine returns bare `{id, score, title: ''}` — titlesMap adds idno, title, type, etc.
 */
function enrichFromTitlesMap(results: SearchResult[]): SearchResult[] {
  if (!titlesMap) return results;
  return results.map((r) => {
    const meta = titlesMap![String(r.id)];
    if (!meta) return r;
    // titlesMap entry wins for metadata fields; keep score/semanticScore/lexicalScore from result
    return {
      ...meta,
      ...r,
      title: (meta.title as string) || r.title,
    } as SearchResult;
  });
}

// ── BM25 setup ────────────────────────────────────────────────────────────────

const tokenizeForBM25 = (nlp: any, its: any) => (text: string) =>
  nlp
    .readDoc(text)
    .tokens()
    .filter((t: any) => t.out(its.type) === "word")
    .out(its.normal);

function buildBM25Engine(
  corpus: Array<{ id: string | number; title: string; text: string }>,
): ReturnType<typeof winkBM25> {
  const nlp = winkNLP(model);
  const its = nlp.its;
  const tokenize = tokenizeForBM25(nlp, its);
  const prepTasks = [tokenize, (tokens: string[]) => tokens];

  const engine = winkBM25();
  engine.defineConfig({ fldWeights: { title: 3, text: 1 } });
  // Default prep tasks are used for search query tokenization; without these,
  // prepareInput(text, 'search') returns the raw string and .filter() throws.
  engine.definePrepTasks(prepTasks);
  engine.definePrepTasks(prepTasks, "title");
  engine.definePrepTasks(prepTasks, "text");
  corpus.forEach((item, idx) => {
    engine.addDoc({ title: item.title, text: item.text }, idx);
  });
  engine.consolidate();
  return engine;
}

// ── Model loading ─────────────────────────────────────────────────────────────

async function loadModel(modelId?: string): Promise<void> {
  postMsg({
    type: "progress",
    phase: "model",
    message: "Loading embedding model…",
  });
  try {
    extractor = await pipeline("feature-extraction", modelId ?? DEFAULT_MODEL, {
      dtype: "q8",
      device: "webgpu",
    } as any);
  } catch {
    extractor = await pipeline("feature-extraction", modelId ?? DEFAULT_MODEL, {
      dtype: "q8",
      device: "wasm",
    } as any);
  }
  postMsg({
    type: "progress",
    phase: "model",
    message: "Embedding model ready",
  });
}

// ── Embedding ─────────────────────────────────────────────────────────────────

async function getEmbedding(text: string): Promise<Float32Array> {
  if (!extractor) throw new Error("Embedding model not loaded");
  // Xenova: model(tokenizer(text)) → sentence_embedding.data (matches Python pipeline / index).
  const result = await extractor.model(extractor.tokenizer(text));
  const raw = new Float32Array(result.sentence_embedding.data);
  l2NormalizeInPlace(raw);
  return raw;
}

// ── Index loading ─────────────────────────────────────────────────────────────

async function initIndex(manifestUrl: string): Promise<void> {
  postMsg({
    type: "progress",
    phase: "index",
    message: "Fetching index manifest…",
  });

  const resp = await fetch(manifestUrl);
  if (!resp.ok) {
    throw new Error(
      `Failed to fetch manifest: ${manifestUrl} (HTTP ${resp.status})`,
    );
  }
  manifest = (await resp.json()) as CollectionManifest;

  baseUrl = manifestUrl.replace(/manifest\.json$/, "");
  if (!baseUrl.endsWith("/")) baseUrl += "/";

  if (manifest.search_mode === "flat") {
    postMsg({
      type: "progress",
      phase: "index",
      message: "Loading flat index…",
    });
    const engine = new FlatEngine();
    const flatItems = await engine.load(baseUrl + manifest.flat!.path);
    searchEngine = engine;
    // Build BM25 corpus directly from flat items (they already carry title + text)
    bm25Corpus = flatItems.map((item) => ({
      id: item.id as string,
      title: String(item.title ?? ""),
      text: String(item.text ?? ""),
    }));
  } else {
    // HNSW mode: load graph, titles, and BM25 corpus in parallel
    postMsg({
      type: "progress",
      phase: "index",
      message: "Loading HNSW index and BM25 corpus…",
    });
    const engine = new HNSWEngine();
    const titlesUrl = manifest.index?.titles
      ? baseUrl + manifest.index.titles
      : null;
    const bm25Url = manifest.index?.bm25_corpus
      ? baseUrl + manifest.index.bm25_corpus
      : null;

    const [, titlesData, bm25Data] = await Promise.all([
      engine.init(baseUrl, { cacheName: CACHE_NAME, manifest }),
      titlesUrl
        ? fetchJson<Record<string, Record<string, unknown>>>(titlesUrl).catch(
            () => null,
          )
        : Promise.resolve(null),
      bm25Url
        ? fetchJson<BM25CorpusEntry[]>(bm25Url, {
            cacheName: CACHE_NAME,
          }).catch(() => null)
        : Promise.resolve(null),
    ]);
    searchEngine = engine;
    titlesMap = titlesData;
    bm25Corpus = bm25Data;
  }

  // Build BM25 engine from corpus
  let bm25Ready = false;
  if (bm25Corpus && bm25Corpus.length > 0) {
    postMsg({
      type: "progress",
      phase: "index",
      message: "Building BM25 index…",
    });
    try {
      bm25Engine = buildBM25Engine(bm25Corpus);

      // Resolve display metadata by BM25 insertion index (wink returns [docIdx, score] tuples)
      const idToMeta = (id: number | string): Partial<SearchResult> => {
        const docIdx = typeof id === "string" ? parseInt(id, 10) : id;
        const item = bm25Corpus![docIdx];
        if (!item) return {};
        if (titlesMap) {
          const meta = titlesMap[String(item.id)];
          return meta
            ? ({
                ...meta,
                title: item.title,
                text: item.text,
              } as Partial<SearchResult>)
            : { id: item.id, title: item.title, text: item.text };
        }
        return { id: item.id, title: item.title, text: item.text };
      };

      hybridEngine = new HybridSearch(searchEngine!, bm25Engine, idToMeta);
      bm25Ready = true;
    } catch (e) {
      console.warn("[search.worker] BM25 init failed:", e);
    }
  }

  isIndexReady = true;
  postMsg({ type: "index_ready", bm25Ready });
  postMsg({ type: "progress", phase: "index", message: "Index ready" });
}

// ── Orchestration ─────────────────────────────────────────────────────────────

async function init(
  manifestUrl: string,
  modelId?: string,
  skipModelLoad?: boolean,
  modelLoadDelaySeconds?: number,
): Promise<void> {
  try {
    if (skipModelLoad) {
      // Load index + BM25 only; leave embedding model unloaded to test BM25 fallback.
      await initIndex(manifestUrl);
      postMsg({
        type: "ready",
        mode: manifest?.search_mode ?? "flat",
        config: manifest!,
        modelLoaded: false,
      });
    } else {
      const delaySec = Math.max(0, modelLoadDelaySeconds ?? 0);
      const loadModelAfterDelay = async () => {
        if (delaySec > 0) {
          postMsg({
            type: "progress",
            phase: "model",
            message: `Waiting ${delaySec} s before loading embedding model…`,
          });
          await new Promise((r) => setTimeout(r, delaySec * 1000));
        }
        await loadModel(modelId);
      };
      await Promise.all([loadModelAfterDelay(), initIndex(manifestUrl)]);
      isModelReady = true;
      postMsg({
        type: "ready",
        mode: manifest?.search_mode ?? "flat",
        config: manifest!,
        modelLoaded: true,
      });
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    postMsg({ type: "error", message });
  }
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (
  e: MessageEvent<WorkerInboundMessage | { ping?: boolean; text?: string }>,
) => {
  const msg = e.data;

  // Legacy bare-message compatibility: { ping: true } or { text: '...' }
  if (!("type" in msg)) {
    if ((msg as any).ping) {
      self.postMessage("pong");
      return;
    }
    if ((msg as any).text && isIndexReady) {
      // Treat as a hybrid search with default options
      const text = (msg as any).text as string;
      try {
        const vec = isModelReady ? await getEmbedding(text) : null;
        const engine = hybridEngine ?? (searchEngine as any);
        if (hybridEngine && vec) {
          const results = await hybridEngine.search(vec, text, {
            topK: 20,
            mode: "hybrid",
          });
          postMsg({
            type: "results",
            data: enrichFromTitlesMap(results),
            fallback: false,
          });
        } else if (hybridEngine && !vec) {
          const results = await hybridEngine.search(null, text, {
            topK: 20,
            mode: "lexical",
          });
          postMsg({
            type: "results",
            data: enrichFromTitlesMap(results),
            fallback: true,
          });
        } else if (engine && vec) {
          const results = await (engine as SearchEngine).search(vec, {
            topK: 20,
          });
          postMsg({
            type: "results",
            data: enrichFromTitlesMap(results as SearchResult[]),
            fallback: false,
          });
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        postMsg({ type: "error", message });
      }
    }
    return;
  }

  switch ((msg as WorkerInboundMessage).type) {
    case "init": {
      const m = msg as Extract<WorkerInboundMessage, { type: "init" }>;
      await init(
        m.manifestUrl,
        m.modelId,
        m.skipModelLoad,
        m.modelLoadDelaySeconds,
      );
      break;
    }

    case "ping": {
      postMsg({ type: "pong" });
      break;
    }

    case "embed": {
      const m = msg as Extract<WorkerInboundMessage, { type: "embed" }>;
      try {
        const vec = await getEmbedding(m.text);
        postMsg({ type: "embedding", data: vec });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        postMsg({ type: "error", message, originalType: "embed" });
      }
      break;
    }

    case "search": {
      const m = msg as Extract<WorkerInboundMessage, { type: "search" }>;
      if (!isIndexReady) return;

      const {
        text,
        topK = 20,
        ef = 50,
        ef_upper = 2,
        threshold = 0.0,
        mode = "hybrid",
      } = m;
      const useFallback =
        !isModelReady && (mode === "semantic" || mode === "hybrid");

      try {
        let results: SearchResult[];

        if (mode === "lexical" || useFallback) {
          // BM25 fallback or explicitly lexical
          if (hybridEngine) {
            results = await hybridEngine.search(null, text, {
              topK,
              ef,
              mode: "lexical",
            });
          } else {
            results = [];
          }
          postMsg({
            type: "results",
            data: enrichFromTitlesMap(results),
            stats: searchEngine?.lastStats ?? null,
            fallback: useFallback,
          });
          return;
        }

        // Semantic or hybrid — model must be ready at this point
        const vec = await getEmbedding(text);

        if (mode === "semantic") {
          if (hybridEngine) {
            results = await hybridEngine.search(vec, text, {
              topK,
              ef,
              mode: "semantic",
            });
          } else if (searchEngine) {
            const raw = await searchEngine.search(vec, {
              topK,
              ef,
              ef_upper,
              threshold,
            });
            results = raw as SearchResult[];
          } else {
            results = [];
          }
        } else {
          // hybrid
          if (hybridEngine) {
            results = await hybridEngine.search(vec, text, {
              topK,
              ef,
              mode: "hybrid",
            });
          } else if (searchEngine) {
            const raw = await searchEngine.search(vec, {
              topK,
              ef,
              ef_upper,
              threshold,
            });
            results = raw as SearchResult[];
          } else {
            results = [];
          }
        }

        // Enrich HNSW results with idno/title/type from titlesMap
        results = enrichFromTitlesMap(results);

        postMsg({
          type: "results",
          data: results,
          stats: searchEngine?.lastStats ?? null,
          fallback: false,
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        postMsg({ type: "error", message, originalType: "search" });
      }
      break;
    }

    case "getRecent": {
      const m = msg as Extract<WorkerInboundMessage, { type: "getRecent" }>;
      const limit = m.limit ?? 10;

      let recent: SearchResult[] = [];

      if (titlesMap) {
        // HNSW mode: take the first `limit` entries from titlesMap
        recent = Object.values(titlesMap)
          .slice(0, limit)
          .map(
            (meta) =>
              ({
                id: (meta.id as string | number) ?? "",
                score: 0,
                title: (meta.title as string) ?? "",
                ...meta,
              }) as SearchResult,
          );
      } else if (bm25Corpus) {
        // Flat mode: take from BM25 corpus
        recent = bm25Corpus.slice(0, limit).map(
          (item) =>
            ({
              id: item.id,
              score: 0,
              title: item.title,
              text: item.text,
            }) as SearchResult,
        );
      }

      postMsg({ type: "recent", data: recent });
      break;
    }

    case "searchCompare": {
      const m = msg as Extract<WorkerInboundMessage, { type: "searchCompare" }>;
      if (!isIndexReady || !isModelReady) return;

      try {
        const vec = await getEmbedding(m.text);
        const topK = m.topK ?? 10;
        const ef = m.ef ?? 50;
        const ef_upper = m.ef_upper ?? 2;

        // Run HNSW search
        const hnswResults = searchEngine
          ? ((await searchEngine.search(vec, {
              topK,
              ef,
              ef_upper,
            })) as SearchResult[])
          : [];

        // Lazy-load the flat compare engine
        if (!flatCompareEngine && manifest?.flat?.path) {
          flatCompareEngine = new FlatEngine();
          await flatCompareEngine.load(baseUrl + manifest.flat.path);
        }

        let flatResults: SearchResult[] = [];
        if (flatCompareEngine) {
          flatResults = flatCompareEngine.search(vec, { topK });
        }

        // Compute recall and overlap
        const hnswIds = new Set(hnswResults.map((r) => String(r.id)));
        const flatIds = new Set(flatResults.map((r) => String(r.id)));
        let overlap = 0;
        for (const id of hnswIds) {
          if (flatIds.has(id)) overlap++;
        }
        const k = Math.max(hnswResults.length, flatResults.length, 1);
        const recall = overlap / Math.min(topK, flatResults.length || topK);

        postMsg({
          type: "compare",
          hnsw: hnswResults,
          flat: flatResults,
          recall,
          overlap,
          k,
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        postMsg({ type: "error", message, originalType: "searchCompare" });
      }
      break;
    }

    default:
      break;
  }
};
