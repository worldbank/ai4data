# Search Pipeline

Python scripts that build a browser-compatible semantic search index from any
document collection. The pipeline produces a sharded HNSW vector index and a
BM25 text corpus that the browser can load on demand.

---

## Prerequisites

### Python environment

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run any pipeline script via uv (resolves deps automatically)
uv run python pipeline/01_fetch_and_prepare.py --help
```

Key Python dependencies (see `pyproject.toml` / `uv.lock`):

| Package | Purpose |
|---|---|
| `faiss-cpu` | K-Means clustering and HNSW index building |
| `sentence-transformers` | Document encoding (step 02 only) |
| `numpy` | Embedding math and quantization |
| `fire` | CLI argument parsing |
| `requests` | World Bank API fetching |
| `pandas` | Excel/CSV loading (optional) |
| `openpyxl` | `.xlsx` support for pandas (optional) |

---

## Scripts

### `01_fetch_and_prepare.py` — Ingest documents

Fetches or loads raw documents and writes `metadata.json` — a JSON array of
records, each with at least `id`, `content` (text used for embedding), and
display fields.

**Supported sources:**

| `--source` | Description |
|---|---|
| `worldbank_api` | World Bank Open Knowledge Repository search API |
| `excel` | `.xlsx` or `.csv` file via pandas |
| `json` | JSON array or dict-of-objects file |

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--source` | `worldbank_api` | Data source type |
| `--output_dir` | `data/collection` | Directory for `metadata.json` |
| `--doctype` | `"Policy Research Working Paper"` | WB API document type filter |
| `--max_docs` | `0` (all) | Limit number of documents fetched |
| `--input_file` | — | Path to `.xlsx`, `.csv`, or `.json` file |
| `--sheet_name` | — | Sheet name for Excel inputs |
| `--id_field` | `idno` | Column/key used as document ID |
| `--content_fields` | `title,abstract` | Comma-separated fields merged into embedding text |
| `--preview_fields` | `idno,title,abstract,type,doi,url,date_published` | Fields to keep for display |

**Output:** `{output_dir}/metadata.json`

---

### `02_generate_embeddings.py` — Encode and quantize

Encodes documents with a sentence-transformers model and exports:

- `raw_embeddings.npy` — L2-normalized float32 `[N, D]` array (input for step 03)
- `flat/embeddings.int8.json` — SQ8-quantized flat index for small collections

Supports **Matryoshka truncation** (`--matryoshka_dim`) for models trained with
Matryoshka Representation Learning (e.g. `nomic-embed-text-v1.5`). Do **not**
set this flag for standard models like `GIST-small-Embedding-v0`.

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--metadata_path` | — | Path to `metadata.json` from step 01 |
| `--embeddings_path` | — | Pre-computed embeddings JSON (skips encoding) |
| `--output_dir` | `data/collection` | Output directory |
| `--model` | `avsolatorio/GIST-small-Embedding-v0` | HuggingFace model ID |
| `--batch_size` | `64` | Encoding batch size |
| `--id_field` | `idno` | Document ID field |
| `--content_field` | `abstract` | Field used as main content text |
| `--title_field` | `title` | Field used as title |
| `--preview_fields` | `idno,title,abstract,type,doi` | Fields in flat index |
| `--matryoshka_dim` | `None` | Truncate to this many dims (MRL models only) |
| `--bm25_text_field` | `abstract` | Field stored as `text` in the flat index |

**Output:** `raw_embeddings.npy`, `metadata.json` (updated), `flat/embeddings.int8.json`

---

### `03_build_index.py` — Build the sharded HNSW index

Builds a sharded HNSW index from `raw_embeddings.npy`. For small collections
(`n_items <= flat_threshold`), only writes the manifest.

**Output files:**

```
{output_dir}/
  manifest.json                       ← top-level manifest for search worker
  flat/
    embeddings.int8.json[.gz]         ← flat index (small collections)
  index/
    config.json[.gz]                  ← HNSW build parameters
    upper_layers.json[.gz]            ← HNSW layers 1+ (fetched at init)
    node_to_shard.json[.gz]           ← node ID → shard ID lookup
    cluster_centroids.json[.gz]       ← quantized cluster centroids
    titles.json[.gz]                  ← lightweight display metadata (no text)
    bm25_corpus.json[.gz]             ← NEW: text corpus for browser BM25
    layer0/
      shard_000.json[.gz]             ← base-layer shard (one per cluster)
      shard_001.json[.gz]
      ...
```

#### `bm25_corpus.json` (new output)

```json
[
  {"id": "WPS9999", "title": "Growth and Poverty", "text": "This paper examines..."},
  ...
]
```

This file contains **only** `id`, `title`, and `text` for every document. It
enables browser-side BM25 keyword search without downloading the full flat
index (~44 MB). Typical sizes: ~6 MB uncompressed, ~2 MB gzip.

The `text` field comes from `metadata.json`'s `text` field (the abstract/body
text). Note that `titles.json` intentionally excludes `text` to stay small;
`bm25_corpus.json` is the dedicated source for BM25.

The manifest references it as:

```json
{
  "index": {
    "bm25_corpus": "index/bm25_corpus.json"
  }
}
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--output_dir` | `data/collection` | Directory with `raw_embeddings.npy` |
| `--collection_id` | `collection` | Short name stored in manifest |
| `--model_id` | `avsolatorio/GIST-small-Embedding-v0` | Model ID for manifest |
| `--hnsw_M` | `16` | HNSW M parameter (bidirectional links per node) |
| `--ef_construction` | `200` | HNSW efConstruction |
| `--flat_threshold` | `2000` | Use flat mode for collections at or below this size |
| `--n_clusters` | auto | Number of layer0 shards (default: `max(10, sqrt(N))`) |
| `--kmeans_niter` | `30` | K-Means iterations |
| `--compress` | `gzip` | `"gzip"` or `"none"` (see GitHub Pages note below) |

---

### `decompress_for_github_pages.py` — Decompress for static hosting

GitHub Pages cannot serve `.json.gz` files with the correct
`Content-Encoding: gzip` headers, causing 500 errors. Run this script after
building with `--compress=gzip` to decompress all index files in-place and
update `manifest.json` to reference the `.json` paths.

```bash
uv run python pipeline/decompress_for_github_pages.py \
  --output_dir=data/prwp
```

Alternatively, build with `--compress=none` from the start (recommended if
you only deploy to GitHub Pages).

---

### `pipeline.py` — End-to-end orchestrator

Runs all three steps in sequence. Accepts every flag from the individual
scripts as top-level arguments.

```bash
uv run python pipeline/pipeline.py --help
```

**Key control flags:**

| Flag | Description |
|---|---|
| `--skip_fetch` | Skip step 01; use existing `metadata.json` |
| `--skip_embed` | Skip step 02; use existing `raw_embeddings.npy` |
| `--output_dir` | Override default `../../data/{collection_id}` |
| `--compress` | `"gzip"` (default) or `"none"` |

---

## Step-by-step example: PRWP collection

```bash
cd /path/to/ai-for-data-blog
export PATH="$HOME/.local/bin:$PATH"

# Full pipeline (fetch → embed → index)
uv run python search/pipeline/pipeline.py prwp \
  --source=worldbank_api \
  --doctype="Policy Research Working Paper" \
  --model=avsolatorio/GIST-small-Embedding-v0 \
  --output_dir=search/data/prwp \
  --hnsw_M=16 \
  --ef_construction=200 \
  --n_clusters=110 \
  --compress=gzip

# After building, serve locally (Node server handles .json.gz automatically)
cd search
npm run serve
# Open http://localhost:3000/?manifest=data/prwp/manifest.json
```

To rebuild only the index (skipping fetch and embed):

```bash
uv run python search/pipeline/pipeline.py prwp \
  --skip_fetch --skip_embed \
  --output_dir=search/data/prwp \
  --n_clusters=110 \
  --compress=gzip
```

---

## GitHub Pages deployment

GitHub Pages serves files as-is but does **not** handle pre-compressed `.gz`
files correctly. Use one of these two approaches:

### Option A — Build uncompressed (simplest)

```bash
uv run python search/pipeline/pipeline.py prwp \
  --output_dir=search/data/prwp \
  --compress=none
```

Commit the resulting `data/prwp/` directory and push. Index files are plain
`.json` and GitHub Pages serves them correctly.

### Option B — Build compressed, then decompress

```bash
# Build with compression (faster transfers on local/Node server)
uv run python search/pipeline/pipeline.py prwp \
  --output_dir=search/data/prwp \
  --compress=gzip

# Decompress for GitHub Pages
uv run python search/pipeline/decompress_for_github_pages.py \
  --output_dir=search/data/prwp

# Commit and push
git add search/data/prwp
git commit -m "Update PRWP search index"
git push
```

---

## Notes on index size

| File | Uncompressed | Gzip |
|---|---|---|
| `flat/embeddings.int8.json` | ~44 MB | ~14 MB |
| `index/upper_layers.json` | ~835 KB | ~250 KB |
| `index/titles.json` | ~1.4 MB | ~450 KB |
| `index/bm25_corpus.json` | ~6 MB | ~2 MB |
| `index/layer0/` (110 shards) | ~14 MB | ~4 MB |
| Per-query cold bandwidth | — | ~383–511 KB (3–4 shards) |

The `bm25_corpus.json` is only fetched when the user activates BM25 mode,
so it does not affect initial load time.
