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
