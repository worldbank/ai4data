# `ai4data.discovery.metadata`

Metadata loading, LangChain document building, Jinja2 embedding templates, parsers, and search filter facets for NADA-style records.

## Modules

| Module | Role |
|--------|------|
| `handler.py` | `MetadataLoader`, per-type `*Metadata` classes, `get_metadata_langdocs`. |
| `document_fetch.py` | PDF download/cache for document metadata. |
| `utils.py` | Id helpers, thumbnail URL, cached id lists. |
| `parsers.py` | Structured parsing helpers per metadata type. |
| `filters.py` | Pydantic filter facets for search. |
| `templates/` | Jinja2 templates + `render.py` for embedding text. |

## Legacy shims

`ai4data.metadata` re-exports this package for backward compatibility.

## Related

- Catalog HTTP / batch jobs: `ai4data.discovery.catalog`.
- PDF loading / chunk helpers used by document metadata: `ai4data.discovery.processors.document` (`load_pdf`, etc.).
- On-disk paths for metadata id lists, JSON/PDF cache, contextual dimensions: `ai4data.discovery.paths` (Qdrant indexed-id files stay in `ai4data.paths`).
