# UNHCR/ReliefWeb to Schema Guide Mapping Documentation

This document maps the metadata schema from UNHCR documents scraped from [ReliefWeb](https://reliefweb.int) to the Document Metadata Schema defined in the [Metadata Standards for Improved Data Discoverability and Usability - Chapter 5 Documents](https://worldbank.github.io/schema-guide/chapter05.html).

**Source structure:** Each metadata file is a flat JSON object with keys scraped from ReliefWeb HTML `<dt>` elements. Keys may appear in singular or plural forms (e.g., `Theme` / `Themes`) depending on how ReliefWeb renders the page.

**Target schema structure:** The schema guide defines a hierarchical structure with `metadata_information`, `document_description`, `provenance`, `tags`, and `additional`.

**Implementation:** [`src/scripts/unhcr_to_schema.py`](../../../../../../src/scripts/unhcr_to_schema.py)

---

## 1. Source Schema Inventory

All 338 metadata files were analyzed. The following keys are present:

| Key | Count | Type | Notes |
|-----|-------|------|-------|
| `report_url` | 338/338 | `str` | ReliefWeb report page URL |
| `title` | 338/338 | `str` | Document title from `<h1>` |
| `Format` | 338/338 | `str` | Document format |
| `Source` | 338/338 | `list` | Source organization(s) |
| `Posted` | 338/338 | `str` | Date posted (e.g., `"20 Aug 2019"`) |
| `Originally published` | 338/338 | `str` | Original publication date |
| `Primary country` | 338/338 | `list` | Primary country (usually 1 element) |
| `pdf_url` | 338/338 | `str` | Direct PDF download URL |
| `Language` | 316/338 | `list` | Language(s) — singular variant |
| `Sources` | 253/338 | `list` | Additional sources — plural variant |
| `Origin` | 239/338 | `str` | Always `"View original"` — **skipped** |
| `Themes` | 204/338 | `list` | Themes — plural variant |
| `Theme` | 126/338 | `list` | Themes — singular variant |
| `Other countries` | 119/338 | `list` | Other countries — plural |
| `Other country` | 62/338 | `list` | Other countries — singular |
| `Disaster type` | 48/338 | `list` | Disaster types — singular |
| `Disaster` | 25/338 | `list` | Disasters — singular |
| `Languages` | 22/338 | `list` | Languages — plural variant |
| `Disaster types` | 18/338 | `list` | Disaster types — plural |
| `Disasters` | 6/338 | `list` | Disasters — plural |

### Unique `Format` Values

Analysis, Appeal, Assessment, Evaluation and Lessons Learned, Infographic, Manual and Guideline, Map, News and Press Release, Other, Situation Report, UN Document

---

## 2. Key Normalization

Singular/plural key variants are merged and deduplicated before mapping:

| Normalized Field | Source Keys |
|---|---|
| sources | `Source`, `Sources` |
| themes | `Theme`, `Themes` |
| languages | `Language`, `Languages` |
| other_countries | `Other country`, `Other countries` |
| disasters | `Disaster`, `Disasters` |
| disaster_types | `Disaster type`, `Disaster types` |

---

## 3. Direct Mappings (document_description)

### 3.1 Title and Identifiers

| UNHCR Field | Target Path | Rationale |
|---|---|---|
| filename stem | `document_description.title_statement.idno` | No explicit ID in source. Filename stem (e.g., `57696` from `57696_metadata.json`) serves as unique identifier within the corpus. The `_metadata` suffix is stripped. |
| `title` | `document_description.title_statement.title` | Direct mapping. Scraped from the `<h1>` element of the ReliefWeb page. |
| — | `document_description.title_statement.sub_title` | **Not available** → `""` |
| — | `document_description.title_statement.alternate_title` | **Not available** → `""` |
| — | `document_description.title_statement.translated_title` | **Not available** → `""` |
| — | `document_description.identifiers` | **Not available** → `[]` |

### 3.2 Authors and Contributors

| UNHCR Field | Target Path | Rationale |
|---|---|---|
| `Source` + `Sources` | `document_description.authors` | Merged, deduplicated. Each source becomes `{first_name: "", initial: "", last_name: "", affiliation: "", full_name: source_name}`. Sources are organizational names (e.g., `"UNHCR"`, `"UNICEF"`), not individual authors. |

### 3.3 Dates

| UNHCR Field | Target Path | Rationale |
|---|---|---|
| `Originally published` | `document_description.date_published` | Original publication date. Parsed from `"20 Aug 2019"` format to `YYYY-MM-DD`. |
| `Posted` | `document_description.date_available` | Date posted on ReliefWeb. Schema: "date when the document was made available." |
| — | `document_description.date_created` | **Not available** → `""` |
| — | `document_description.date_modified` | **Not available** → `""` |

**Date parsing:** ReliefWeb uses `"%d %b %Y"` format (e.g., `"20 Aug 2019"`). Converted to ISO 8601 `YYYY-MM-DD`.

### 3.4 Document Type

| UNHCR Field | Target Path | Rationale |
|---|---|---|
| `Format` | `document_description.type` | Document format (e.g., `"Situation Report"`, `"Infographic"`). Schema: "nature of the resource." |

### 3.5 Geographic Coverage

| UNHCR Field | Target Path | Rationale |
|---|---|---|
| `Primary country` + `Other country/ies` | `document_description.ref_country` | Merged (primary first, then others). Each becomes `{name, code}` where `code` is resolved to ISO3 via the `countrycode` library. |

**Country code resolution:**
- Standard names resolved via `countrycode` library (`country.name.en.regex` → `iso3c`)
- Special cases handled explicitly: `"World"` → `"WLD"`, `"occupied Palestinian territory"` → `"PSE"`, `"the Republic of North Macedonia"` → `"MKD"`
- Unresolvable names get `code: ""`

### 3.6 Languages

| UNHCR Field | Target Path | Rationale |
|---|---|---|
| `Language` + `Languages` | `document_description.languages` | Merged, deduplicated. Each becomes `{name, code}` with ISO 639 code lookup from a built-in mapping table. Default: `[{name: "English", code: "en"}]` when no language keys are present. |

### 3.7 Themes

| UNHCR Field | Target Path | Vocabulary | Rationale |
|---|---|---|---|
| `Theme` + `Themes` | `document_description.themes` | `UNHCR - theme` | Merged, deduplicated. Each becomes `{id: "", name, parent_id: "", vocabulary: "UNHCR - theme", uri: ""}`. |

### 3.8 URLs and Access

| UNHCR Field | Target Path | Rationale |
|---|---|---|
| `report_url` | `document_description.url` | ReliefWeb report page. Schema: "URL of the document, preferably a permanent URL." |
| `pdf_url` | `document_description.relations` | `{name: pdf_url, type: "hasFormat"}`. PDF is an alternate format of the report page. |

---

## 4. Metadata Information Block

| UNHCR Field | Target Path | Rationale |
|---|---|---|
| `title` | `metadata_information.title` | Same as document title. |
| filename stem | `metadata_information.idno` | Same as document idno. |
| (static) | `metadata_information.producers` | `[{name: "UNHCR", abbr: "UNHCR", affiliation: "UNHCR", role: "Source"}]` |
| (derived) | `metadata_information.production_date` | Current date at conversion time (`YYYY-MM-DD`). |
| (derived) | `metadata_information.version` | `"Converted from UNHCR/ReliefWeb metadata on YYYY-MM-DD"` |

---

## 5. Provenance Block

| UNHCR Field | Target Path | Rationale |
|---|---|---|
| (derived) | `provenance.origin_description.harvest_date` | Current ISO 8601 timestamp at conversion time. |
| (static) | `provenance.origin_description.base_url` | `https://reliefweb.int` |
| `report_url` | `provenance.origin_description.identifier` | Source page URL for traceability. Falls back to `idno` if `report_url` is empty. |
| `date_published` or `date_available` | `provenance.origin_description.date_stamp` | Earliest known date from source. |
| (static) | `provenance.origin_description.metadata_namespace` | `UNHCR_RELIEFWEB` |
| (static) | `provenance.origin_description.altered` | `false` (direct mapping, no enrichment). |

---

## 6. Additional Fields

UNHCR-specific fields not mapped to standard schema elements are preserved in the `additional` block:

| UNHCR Field | Target Path | Rationale |
|---|---|---|
| `report_url` | `additional.unhcr_report_url` | Preserve source URL for traceability. |
| `Format` | `additional.unhcr_format` | Preserve raw format value. |
| `Disaster` + `Disasters` | `additional.unhcr_disasters` | List of disaster names. No schema equivalent. |
| `Disaster type` + `Disaster types` | `additional.unhcr_disaster_types` | List of disaster type names. No schema equivalent. |

---

## 7. Skipped Fields

| UNHCR Field | Reason |
|---|---|
| `Origin` | Always contains `"View original"` — a link label with no semantic value. |

---

## 8. Schema Elements Not Populated from UNHCR

The following schema elements have no source in the UNHCR metadata and are set to empty defaults:

| Schema Element | Default | Rationale |
|---|---|---|
| `document_description.abstract` | `""` | Not scraped from ReliefWeb. |
| `document_description.date_created` | `""` | Not available. |
| `document_description.date_modified` | `""` | Not available. |
| `document_description.identifiers` | `[]` | No explicit IDs in source. |
| `document_description.geographic_units` | `[]` | Not available. |
| `document_description.spatial_coverage` | `""` | Not available. |
| `document_description.volume` | `""` | Not applicable. |
| `document_description.number` | `""` | Not applicable. |
| `document_description.series` | `""` | Not applicable. |
| `document_description.publisher_address` | `""` | Not available. |
| `document_description.organization` | `""` | Not available. |
| `document_description.keywords` | `[]` | Not available. |
| `document_description.topics` | `[]` | Not available. |
| `document_description.security_classification` | `""` | Not available. |
| `document_description.access_restrictions` | `""` | Not available. |
| `document_description.edition` | `""` | Not available. |
| `document_description.contacts` | `[]` | Not available. |
| `document_description.usage_terms` | `""` | Not available. |
| `document_description.notes` | `[]` | Not available. |
| `tags` | `[]` | Disasters mapped to `additional` instead. |

**Empty defaults rationale:** Empty strings (`""`) and empty lists (`[]`) are used instead of `null` to ensure HuggingFace datasets compatibility — all output files must share an identical schema shape for `Features` inference.

---

## References

- [ReliefWeb](https://reliefweb.int)
- [Schema Guide Chapter 5 - Documents](https://worldbank.github.io/schema-guide/chapter05.html)
- [WDS mapping reference](./WDS_TO_SCHEMA_MAPPING.md)
- [Implementation](../../../../../../src/scripts/unhcr_to_schema.py)
