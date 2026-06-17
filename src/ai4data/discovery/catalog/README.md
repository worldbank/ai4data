# `ai4data.discovery.catalog`

Public surface for **NADA** metadata: HTTP client, batch jobs, and ids used by downstream packages (e.g. `nada-opensearch`).

## Modules

| Module | Role |
|--------|------|
| `http.py` | Catalog HTTP API: `get_metadata_json`, `search_metadata`, `get_ids_type`, `get_metadata_ids`. Routes to extract API when `AI4DATA_METADATA_CATALOG_EXTRACT_PATH` is set. |
| `extract.py` | IHSN search-metadata-extract client: paginated `/studies`, `study_to_catalog_metadata`, param mapping. Study-level `resources` with `_links.type == "download"` are normalized to catalog `external_resources`. |
| `batch.py` | Save id lists, `scrape_all_ids`, `scrape_all_metadata`, Fire `main`. Run: `python -m ai4data.discovery.catalog.batch`. |
| `data_api.py` | Authenticated JSON GET (`x-api-key`) for separate IHSN/API resources. |
| `langdoc_id.py` | `get_langdoc_uuid` and UUID helper for chunk ids. |
| `__init__.py` | Re-exports HTTP + `get_langdoc_uuid`. Import `MetadataLoader` / `get_metadata_langdocs` from `ai4data.discovery.metadata.handler` (avoiding a circular import). |

## Related (not in `discovery/catalog/`)

- **PDF download/cache** for document metadata: `ai4data.discovery.metadata.document_fetch` (`cache_download_pdf`, `download_pdf`).

### Extract resources and PDF cache naming

When extract mode is enabled, each study may include a top-level `resources` array. Only entries with `_links.type == "download"` are kept (external `"link"` mirrors are ignored). They are mapped to `external_resources` with:

- `url` ← `_links.download` (NADA admin download URL)
- `resource_id` preserved from the extract payload

Document PDFs are cached under `document_cache/document/` as:

- **Extract mode:** `document_{idno}--{resource_id}.pdf` (double hyphen separates idno from resource id because idno contains underscores)
- **Classic catalog:** `document_{idno}.pdf` (unchanged)

Re-index with `--force` or delete old PDFs after switching naming conventions.

## Legacy shims

`ai4data.catalog` re-exports this package for backward compatibility. `ai4data.scraper.metadata`, `ai4data.scraper.document`, and `ai4data.scraper.data_api` also shim older paths. Prefer imports from `ai4data.discovery.catalog`.

## Consumers

- `nada-opensearch` uses `ai4data.discovery.catalog` for HTTP + loader-related exports.

## Maintenance

Update `discovery/catalog/http.py` when catalog REST behavior changes. Batch CLI flags live in `discovery/catalog/batch.py`.

## Configuration

Environment variables for the catalog URL, API key, and related settings are documented in [`../README.md`](../README.md).
