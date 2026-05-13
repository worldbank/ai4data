# `ai4data.discovery`

NADA (IHSN) catalog HTTP helpers, cached metadata/PDF workflows, Jinja templates for embedding text, and PDF/chunk helpers.

Install: `uv pip install ai4data[discovery]`

## Environment variables

Configuration uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/). Variable names are **case-insensitive** on all platforms. Prefixes are fixed per settings group.

On import, `discovery.config` calls `load_dotenv()`, so a **`.env`** file in the current working directory is loaded automatically when present.

### Catalog (`AI4DATA_METADATA_CATALOG_*`)

Defined in [`config.py`](./config.py) (`MetadataCatalogConfig`). Used by [`catalog/http.py`](./catalog/http.py), [`catalog/data_api.py`](./catalog/data_api.py), and thumbnail URLs in [`metadata/utils.py`](./metadata/utils.py).

| Variable | Required | Description |
|----------|-----------|-------------|
| `AI4DATA_METADATA_CATALOG_URL` | No | Base URL for the NADA catalog UI/API (default: `https://data-compass.ihsn.org/index.php`). Search and JSON endpoints are built from this value. |
| `AI4DATA_METADATA_CATALOG_THUMBNAIL_URL` | No | Format string for document thumbnails; must include `{db_id}` (default matches NADA public thumbnails). |
| `AI4DATA_METADATA_CATALOG_X_API_KEY` | Only for authenticated APIs | API key sent as header `x-api-key`. Used by [`catalog/data_api.py`](./catalog/data_api.py), and — when set — also attached by [`catalog/http.py`](./catalog/http.py) (search, JSON metadata) and [`metadata/document_fetch.py`](./metadata/document_fetch.py) (PDF downloads). For PDF downloads the header is only sent when the URL's host matches the configured catalog host (or is allow-listed via `AI4DATA_METADATA_CATALOG_X_API_KEY_HOSTS`), so the credential is never sent to third-party hosts embedded as external resources. |
| `AI4DATA_METADATA_CATALOG_X_API_KEY_HOSTS` | No | Comma-separated list of additional hostnames allowed to receive `x-api-key` for catalog-resolved downloads (e.g. `training.ihsn.org`). The catalog host itself is always allowed; use this when downloads are served from a separate subdomain. |

### Embedding templates (`AI4DATA_EMBEDDING_*`)

Defined in [`config.py`](./config.py) (`EmbeddingTemplatesConfig`). Controls where Jinja2 templates live for [`metadata/templates/render.py`](./metadata/templates/render.py).

| Variable | Required | Description |
|----------|-----------|-------------|
| `AI4DATA_EMBEDDING_CONTENT_TEMPLATES_PATH` | No | Directory containing per-type template folders (`indicator`, `document`, …). Defaults to `metadata/templates` next to `config.py` inside the installed package. |

The `AI4DATA_EMBEDDING_` prefix is shared with **embedding inference** (below). Unrecognized keys are ignored per settings class.

### Embedding inference (`AI4DATA_EMBEDDING_*`)

Defined in [`config.py`](./config.py) (`EmbeddingInferenceConfig`). Used by [`embeddings.py`](./embeddings.py) for `HuggingFaceEmbeddings` and the token text splitter when you call [`wiring.register_discovery_processors()`](./wiring.py) (see below). The `discovery` extra installs `torch`, `sentence-transformers` (>=5.4.1), and `langchain-text-splitters` needed for those code paths.

| Variable | Required | Description |
|----------|-----------|-------------|
| `AI4DATA_EMBEDDING_MODEL` | No | HuggingFace model id (default: `avsolatorio/GIST-Embedding-v0`). |
| `AI4DATA_EMBEDDING_BATCH_SIZE` | No | Encode batch size (default: `64`). |
| `AI4DATA_EMBEDDING_DEVICE` | No | Device string, e.g. `cuda`, `mps`, `cpu`. If unset, auto-pick cuda → mps → cpu. |
| `AI4DATA_EMBEDDING_SHOW_PROGRESS` | No | Progress bars during encoding (default: `true`). |

**Wiring (required for `embed_documents` / `get_doc_reps`):** call once at process startup (not on import):

```python
from ai4data.discovery.wiring import register_discovery_processors

register_discovery_processors()
```

To clear cached models in long-running jobs or tests, use `ai4data.discovery.embeddings.clear_embedding_caches()`.

### Local data / cache (`AI4DATA_DISCOVERY_*`)

Defined in [`config.py`](./config.py) (`DiscoveryDataConfig`). Used by [`paths.py`](./paths.py) for `metadata_ids`, `metadata_cache`, `document_cache`, and related subpaths.

| Variable | Required | Description |
|----------|-----------|-------------|
| `AI4DATA_DISCOVERY_DATA_PATH` | No | Root directory for discovery caches. If unset, defaults to `ai4data/data` under the installed package (in this repo from source: `src/ai4data/data`). |

**Precedence:** `init_discovery_paths(Path("..."))` with a non-`None` path always wins over the env var. Calling `init_discovery_paths(None)` (or `init_discovery_paths()` with default) applies `_default_data_root()`, which respects `AI4DATA_DISCOVERY_DATA_PATH` when set.

### Optional: Apache Tika

[`processors/document.py`](./processors/document.py) uses **`tika`** for `tika_parse_pdf`. The Python package may require a running Tika server or local JAR; see [tika-python](https://github.com/chrismattmann/tika-python) documentation for server/JVM setup. That is runtime configuration outside `ai4data.discovery.config`.
