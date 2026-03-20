# Metadata Project

Tools for World Bank document metadata transformation and schema mapping.

## WDS to Schema Mapping

![WDS to Document Metadata Schema Mapping](/docs/images/metadata-wds-dms-app.png)

The [wds_to_schema.py](wds_to_schema.py) module converts World Bank Search API (WDS) v3 document metadata to the document schema (Chapter 5 of the [Metadata Standards Guide](https://worldbank.github.io/schema-guide/chapter05.html)). See [docs/WDS_TO_SCHEMA_MAPPING.md](docs/WDS_TO_SCHEMA_MAPPING.md) for the full mapping documentation and rationale.

```bash
# Transform from JSON file
uv run python scripts/metadata/wds_to_schema.py --input=doc.json --output=document.json

# Fetch by guid and save
uv run python scripts/metadata/wds_to_schema.py --guid=969971600710472848 --output_dir=metadata

# Preserve all values (no deduplication) for full provenance
uv run python scripts/metadata/wds_to_schema.py --guid=969971600710472848 --output_dir=metadata --no-deduplicate
```

## WDS Schema Mapper App

A mini web application to visualize WDS metadata and its mapped schema side-by-side.

```bash
uv run python scripts/metadata/wds_mapper_app.py
```

Open [http://localhost:5051](http://localhost:5051), paste a WDS URL or guid, and click Map. Enable "Validate" to verify all WDS fields are preserved in the mapped schema.

**Tip:** The mapped schema's `additional` block (at the bottom) preserves raw WDS values like `additional.wds_countrycode` (e.g. `"1W"`) for traceability alongside transformed values in `document_description`.

## PDF-to-Metadata Extraction Pipeline

The [pdf_to_metadata.py](pdf_to_metadata.py) pipeline extracts content from PDFs using [pymupdf4llm](https://github.com/pymupdf/pymupdf4llm) and generates draft metadata via an LLM (litellm). Output conforms to the Document Metadata Schema.

**Prerequisites:** Install with `uv sync --extra metadata` (adds `pymupdf4llm` and `litellm`). Set provider-specific env vars for the LLM (see below).

```bash
# From local PDF (default: first 5 pages)
uv run python scripts/metadata/pdf_to_metadata.py --input=report.pdf --output=document.json

# From URL
uv run python scripts/metadata/pdf_to_metadata.py --url=https://example.com/doc.pdf --output=document.json

# Extract all pages (slower, more tokens)
uv run python scripts/metadata/pdf_to_metadata.py --input=report.pdf --output=document.json --max_pages=0

# Use a different model
uv run python scripts/metadata/pdf_to_metadata.py --input=report.pdf --model=claude-3-5-sonnet-20241022

# Azure OpenAI (set AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION in .env)
uv run python scripts/metadata/pdf_to_metadata.py --input=report.pdf --model=azure/gpt-4o-mini
```

| Flag | Default | Description |
|------|---------|--------------|
| `--input` | — | Path to local PDF |
| `--url` | — | URL to PDF |
| `--output` | `document_<stem>.json` | Output JSON path |
| `--max_pages` | `5` | Max pages to extract; `0` = all |
| `--model` | `gpt-4o-mini` | litellm model string |
| `--idno` | derived from filename | Document identifier |
| `--api_base` | — | Custom API base URL |
| `--api_version` | — | API version (for Azure, e.g. `2024-02-15-preview`) |
| `--no-pydantic` | — | Use `json_object` instead of Pydantic schema (fallback if model rejects schema) |
| `--max_content_tokens` | `100000` | Max tokens of document content to send (uses tiktoken) |

By default, content is truncated to 100k tokens (not characters) using tiktoken. Increase for long documents if your model's context window allows.

By default, the pipeline uses a Pydantic schema for structured output ([litellm JSON mode](https://docs.litellm.ai/docs/completion/json_mode)), which enforces the document metadata structure. Use `--no-pydantic` if your model does not support it.

**Environment variables (provider-specific):**

| Provider | Env vars |
|----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| **Azure OpenAI** (API key) | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |
| **Azure OpenAI** (client credentials) | `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`, `AZURE_API_BASE`, `AZURE_SCOPE` (optional) |

For Azure, use `--model=azure/<deployment_name>` (e.g. `azure/gpt-4o-mini`). With **client credentials** (client ID + secret), set `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`; `AZURE_SCOPE` defaults to `https://cognitiveservices.azure.com/.default` or use your API scope exactly as in your working setup. Use `--azure_scope` to override without changing `.env`. Optionally override via `--api_base` and `--api_version`.

By default, only the first 5 pages are sent to the LLM. Most metadata (title, authors, abstract, dates) appears in the first few pages; use `--max_pages=0` for full-document extraction when keywords/themes from the body are needed.

## WDS-to-Schema Validator

The [wds_schema_validator.py](wds_schema_validator.py) module ensures all information from the original WDS JSON is available in the mapped document schema.

```bash
# Validate from JSON files
uv run python scripts/metadata/wds_schema_validator.py --wds=wds_doc.json --mapped=mapped_schema.json

# In Python
from wds_schema_validator import validate_wds_to_schema, validate_and_raise

result = validate_wds_to_schema(wds_doc, mapped_schema)
if not result["valid"]:
    for m in result["missing"]:
        print(f"Missing: {m['field']} = {m['value']}")

# Or raise on failure
validate_and_raise(wds_doc, mapped_schema)
```
