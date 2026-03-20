# WDS API to Schema Guide Mapping Documentation

This document maps the metadata schema from the [World Bank Search API (WDS) v3](https://search.worldbank.org/api/v3/wds) to the document schema defined in the [Metadata Standards for Improved Data Discoverability and Usability - Chapter 5 Documents](https://worldbank.github.io/schema-guide/chapter05.html).

**Source API structure:** Documents are returned under `documents.<key>` where each document is a flat object with abbreviated field names (e.g., `docdt`, `admreg`, `docty`).

**Target schema structure:** The schema guide defines a hierarchical structure with `metadata_information`, `document_description`, `provenance`, `tags`, `lda_topics`, `embeddings`, and `additional`.

**Sample documents analyzed:**
- Project document (guid=099021126100514593): Disbursement Letter, Bolivia
- PRWP (guid=969971600710472848): "The Role of Inequality for Poverty Reduction"
- Working Paper (guid=473751495468210827): Nicaragua PES study
- Working Paper with project (guid=824641572985831195): "Benefit Sharing at Scale"

---

## 1. Direct Mappings (document_description)

### 1.1 Title and Identifiers

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `id` | `document_description.title_statement.idno` | The WDS `id` is the primary catalog identifier. Schema requires a unique `idno`; WDS numeric ID serves this purpose. Use `guid` as alternate if human-readable ID preferred. |
| `guid` | `document_description.identifiers` (type: "World Bank GUID") | The `guid` (e.g., "099021126100514593") is a stable, human-readable identifier. Store in `identifiers` with type "World Bank GUID" for traceability. |
| `display_title` | `document_description.title_statement.title` | Dublin Core "title" element. `display_title` is the canonical display name. |
| `docna` | `document_description.title_statement.sub_title` or `alternate_title` | `docna` contains the document filename/title variant. Map to `sub_title` when it describes a variant, or `alternate_title` when it is an abbreviated form. |
| `repnme` | `document_description.title_statement.alternate_title` (if docna absent) | Report name; alternate title when docna is not present. |
| `alt_title` | `document_description.title_statement.alternate_title` / `translated_title` | Object `{"0":{"alt_title":"..."}, "1":{"alt_title":"..."}}`. Join with " / " for `alternate_title`; if language-indicated, use for `translated_title`. |
| `chronical_docm_id` | `document_description.identifiers` (type: "World Bank Chronical ID") | Internal WB document ID; preserve for provenance and linking. |
| `entityids.entityid` | `document_description.identifiers` (type: "World Bank Entity ID") | Entity identifier when present. |

**Rationale for idno choice:** The schema recommends globally unique identifiers (DOI, ISBN). WDS has no DOI/ISBN for many project documents. Using `id` as `idno` ensures uniqueness within the catalog; `guid` can be used if the target system prefers the curated GUID format.

### 1.2 Authors and Contributors

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `authors` | `document_description.authors` | WDS `authors` is `{"0":{"author":"LEGLA"},...}`. Each entry maps to `{full_name: author_value}` since WDS does not provide first/last name split. Schema allows `full_name` when first/last cannot be distinguished. |
| `owner` | `document_description.contributors` or `additional.wds_owner` | WB organizational unit (e.g., "DECRG: Poverty & Inequality"). Map to contributor with `contribution: "Owner"` or preserve in `additional`. |
| `origu` | `document_description.contributors` or `additional.wds_origin_unit` | Origin unit; preserve for traceability. |

**Transformation:** For each `authors.<k>.author`, create `{first_name: "", last_name: "", full_name: author_value}`. If `author` is a list (v2 API), take first element.

### 1.3 Dates

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `docdt` | `document_description.date_created` | Document date in WDS typically indicates when the document was produced. Schema: "date when the document was produced." |
| `disclosure_date` | `document_description.date_available` | When the document was made available (disclosed). Schema: "date when the document was made available." |
| `datestored` | `document_description.date_modified` | Date stored in system; closest to "last modified" in schema. |
| `last_modified_date` | `document_description.date_modified` (prefer over datestored) | More explicit "last modified" semantics. Use if both present; prefer `last_modified_date`. |
| `lupdate` | `provenance.origin_description.date_stamp` | System update timestamp; use for harvest provenance, not document lifecycle. |

**Note:** WDS v3 does not expose `publishtoextweb_dt` (publish date). Use `docdt` or `disclosure_date` as fallback for `date_published` if needed.

### 1.4 Document Type and Classification

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `docty` | `document_description.type` | Document type (e.g., "Disbursement Letter", "Policy Research Working Paper"). Schema: "nature of the resource" with controlled vocabulary. |
| `majdocty` | `tags` (tag_group: "major_document_type") | Major document type (e.g., "Project Documents", "Publications & Research"). Use for faceting. |
| `seccl` | `document_description.security_classification` | Security classification (e.g., "Public"). Direct semantic match. |
| `disclstat` | `document_description.access_restrictions` or `notes` | Disclosure status ("Disclosed"). Indicates access; can go in `access_restrictions` as free text or `notes`. |
| `versiontyp` | `document_description.edition` or `notes` | Version type (e.g., "Final", "Revised", "Buff Cover"). Edition or note. |

### 1.5 Geographic Coverage

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `count` | `document_description.ref_country` | Country names (comma-separated, e.g., "Bolivia", "World"). Split and map to `{name, code}` using countrycode lookup. Schema recommends ISO 3166. |
| `countrycode` | `document_description.ref_country[].code` | ISO2 code (e.g., "BO", "NI") or "1W" for World. When `count` has one country, use `countrycode` for code; "1W" -> WLD. |
| `admreg` | `document_description.spatial_coverage` | Administrative region (e.g., "Latin America and Caribbean"). Schema: "qualify the geographic coverage" as free text. |
| `geo_regions` | `document_description.geographic_units` or `spatial_coverage` | Object `{"0":{"geo_region":"Central America"},...}`. Extract to `geographic_units` with `type: "region"` or append to `spatial_coverage`. |

**Transformation:** Parse `count` by comma; for each, get ISO3 via countrycode. If `countrycode` present: "1W" -> WLD. Otherwise convert ISO2 to ISO3.

### 1.6 Topics, Themes, and Keywords

| WDS Field | Target Path | Vocabulary | Rationale |
|-----------|-------------|------------|-----------|
| `theme` | `document_description.themes` | `World Bank - theme` | Comma-separated themes. Source tracked in vocabulary. |
| `majtheme` | `document_description.themes` | `World Bank - majtheme` | Major theme; add as single theme entry. |
| `sectr` | `document_description.themes` | `World Bank - sector` | Sector object `{"0":{"sector":"Social Protection"},...}`. Extract to themes. |
| `topicv3_name` | `document_description.topics` | `World Bank - topicv3_name` | Topic names (comma-separated). Processed first. |
| `ent_topic` | `document_description.topics` | `World Bank - ent_topic` | Entity topics; added if not duplicate of topicv3_name. |
| `subtopic` | `document_description.topics` | `World Bank - subtopic` | Comma-separated subtopics. |
| `teratopic` | `document_description.topics` | `World Bank - teratopic` | Tertiary topics; add to topics. |
| `historic_topic` | `additional.wds_historic_topic` | — | Legacy topic classification; preserved in additional. |
| `keywd` | `document_description.keywords` | `World Bank - keywd` | Object `{"0":{"keywd":"..."},...}`. Extract each keywd. |
| `subsc` | `document_description.keywords` | `World Bank - subsc` | Subject codes (e.g., "Public Administration - Social Protection"). |

**Source-tracking:** When multiple WDS fields are combined into a single repeatable schema element (themes, topics, keywords), the `vocabulary` field encodes the source WDS field as `World Bank - {wds_field}`. See [Section 6](#6-source-tracking-in-combined-fields) for details.

### 1.7 Language

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `lang` | `document_description.languages` | Primary language (e.g., "English"). Map to `[{name: "English", code: "en"}]` using ISO 639-2. |
| `available_in` | `document_description.languages` | Comma-separated list of languages. Merge with `lang`; each becomes `{name, code}`. |
| `fullavailablein` | `document_description.languages` | Additional languages; merge into `languages` array. |

### 1.8 URLs and Access

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `pdfurl` | `document_description.url` | Primary document URL. Schema: "URL of the document, preferably a permanent URL." |
| `url` | `document_description.url` (fallback) | Generic URL when `pdfurl` absent. |
| `wrdurl` | `document_description.relations` (type: "hasFormat") or `additional.wds_wrdurl` | Word document URL. Alternate format of same resource. |
| `txturl` | `document_description.relations` (type: "hasFormat") | Text extraction URL; alternate format of same resource. |

### 1.9 Project Context (World Bank-specific)

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `projn` | `document_description.relations` or `additional.project_name` | Project name. Schema has no "project" element; use `relations` with type "isPartOf" (link to project) or `additional.project_name` per schema 5.2.7. |
| `projectid` | `document_description.identifiers` (type: "World Bank Project ID") or `additional.project_id` | Project identifier. Preserve for linking. |
| `prdln` | `document_description.organization` or `tags` (tag_group: "product_line") | Product line (e.g., "IBRD/IDA", "Advisory Services & Analytics"). Organization or tag for faceting. |
| `lndinstr` | `document_description.notes` or `tags` (tag_group: "lending_instrument") | Lending instrument. WB-specific; use `notes` or `tags`. |
| `project_status` | `document_description.notes` or `tags` (tag_group: "project_status") | "Active", "Closed" etc. Preserve in `notes` or `tags`. |

### 1.10 Bibliographic & Series Information

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `repnb` | `document_description.number` | Report number (e.g., "WPS9409", "115197"). Schema: journal/report number. |
| `volnb` | `document_description.volume` | Volume number. |
| `totvolnb` | `document_description.notes` or `additional.wds_totvolnb` | Total volume number; preserve if present. |
| `colti` | `document_description.series` or `journal` | Collection title (e.g., "Policy Research working paper|no. WPS 9409"). Map to `series` or `journal`. |
| `placeprod` | `document_description.publisher_address` | Place of production (e.g., "Washington, D.C. : World Bank Group"). |

### 1.11 Abstract and Description

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `abstracts.cdata!` | `document_description.abstract` | v3 exposes abstract in `abstracts.cdata!`. Primary source. |
| `abstracts.#text` | `document_description.abstract` (fallback) | Alternative key for abstract content. |

---

## 2. Metadata Information Block

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| (derived) | `metadata_information.title` | Use `display_title`; metadata doc title should match document title. |
| `id` | `metadata_information.idno` | Same as document idno for consistency. |
| (static) | `metadata_information.producers` | No WDS equivalent. Use `[{name: "World Bank", abbr: "WB", affiliation: "World Bank", role: "Source"}]` for harvested metadata. |
| (derived) | `metadata_information.production_date` | Use harvest date (e.g., `lupdate` or current date). |
| (derived) | `metadata_information.version` | e.g., "Harvested from WDS API v3 on YYYY-MM-DD". |

---

## 3. Provenance Block

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| (derived) | `provenance.origin_description.harvest_date` | ISO 8601 harvest timestamp. |
| (static) | `provenance.origin_description.base_url` | `https://search.worldbank.org/api/v3/wds` |
| `guid` | `provenance.origin_description.identifier` | Source catalog identifier for traceability. |
| `lupdate` | `provenance.origin_description.date_stamp` | Last update in source catalog. |
| (static) | `provenance.origin_description.metadata_namespace` | "WB_WDS" |
| (static) | `provenance.origin_description.altered` | `false` if direct mapping; `true` if enriched. |

---

## 4. Tags (for Faceting)

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `majdocty` | `tags` (tag_group: "major_document_type") | Enable "Project Documents" filter. |
| `docty` | `tags` (tag_group: "document_type") | More granular type if different from majdocty. |
| `prdln` | `tags` (tag_group: "product_line") | IBRD/IDA etc. |
| `sectr` | `tags` (tag_group: "sector") | Sector facets. |
| `project_status` | `tags` (tag_group: "project_status") | Active/Closed. |

---

## 5. Additional Fields (schema 5.2.7)

Per schema guide, custom elements must be added within the `additional` block with names prefixed by `additional.`. No WDS field is left out: all fields are either mapped to schema elements or preserved in `additional`.

| WDS Field | Target Path | Rationale |
|-----------|-------------|-----------|
| `action` | `additional.wds_action` | Internal sync action (INSERT/UPDATE); preserve for debugging. |
| `projectid_sort` | `additional.wds_projectid_sort` | Sort key; preserve if needed for ordering. |
| `docm_id` | `additional.wds_docm_id` | If present in v3. |
| `ml_repnme` | `additional.wds_ml_repnme` | ML-generated report name. |
| `origu` | `additional.wds_origin_unit` | Origin unit (when not in contributors). |
| `owner` | `additional.wds_owner` | Owner unit (when not in contributors). |
| `projectid` | `additional.project_id` | Project identifier (also in identifiers); preserve for linking. |
| `projn` | `additional.project_name` | Project name (also in relations); preserve for linking. |
| `entityids` | `additional.wds_entityids` | Entity IDs object. |
| `historic_topic` | `additional.wds_historic_topic` | Legacy topic classification; preserve. |
| `totvolnb` | `additional.wds_totvolnb` | Total volume number; preserve if present. |
| `lndinstr_exact` | `additional.wds_lndinstr_exact` | Exact lending instrument string (e.g. "Investment Project Financing"). |
| `prdln_exact` | `additional.wds_prdln_exact` | Exact product line string (e.g. "IBRD/IDA"). |
| `countrycode` | `additional.wds_countrycode` | Raw ISO2/special code (e.g. "1W") for traceability; transformed value in `ref_country`. |

**Catch-all:** Any WDS field not explicitly mapped above is preserved in `additional.wds_{fieldname}` so that no field is ever dropped. This ensures forward compatibility when the WDS API adds new fields.

---

## 6. Source-Tracking in Combined Fields

When multiple WDS fields are merged into a single repeatable schema element (themes, topics, keywords), the `vocabulary` field records the **source WDS field** so each value remains traceable to its origin.

### 6.1 Rationale

- **Provenance:** Enables auditing and debugging of harvested metadata.
- **Quality:** Supports filtering or weighting by source (e.g., prefer `topicv3_name` over `teratopic`).
- **Interoperability:** Downstream systems can interpret or map values by source.

### 6.2 Vocabulary Format

Format: `World Bank - {wds_field}`

| Schema Element | WDS Sources | Vocabulary Values |
|----------------|-------------|-------------------|
| `document_description.themes` | `theme`, `majtheme`, `sectr` | `World Bank - theme`, `World Bank - majtheme`, `World Bank - sector` |
| `document_description.topics` | `topicv3_name`, `ent_topic`, `subtopic`, `teratopic` | `World Bank - topicv3_name`, `World Bank - ent_topic`, `World Bank - subtopic`, `World Bank - teratopic` |
| `document_description.keywords` | `keywd`, `subsc` | `World Bank - keywd`, `World Bank - subsc` |

### 6.3 Processing Order and Deduplication

**Deduplication is optional** (default: on). Use `deduplicate_combined_fields=False` in `wds_to_schema()` or `--no-deduplicate` in the CLI to preserve all values for full provenance.

**Themes:** Processed in order `theme` → `majtheme` → `sectr`. With deduplication, sectors are added only if the name is not already present from theme or majtheme.

**Topics:** Processed in order `topicv3_name` → `ent_topic` → `subtopic` → `teratopic`. With deduplication, duplicates (by `name`) are skipped so the first occurrence keeps its vocabulary.

**Keywords:** Processed in order `keywd` → `subsc`. With deduplication, subsc terms already present from keywd are skipped.

### 6.4 Example

WDS input:
```json
{
  "theme": "Social Protection,Climate Change",
  "majtheme": "Poverty Reduction",
  "sectr": {"0":{"sector":"Social Protection"}},
  "topicv3_name": "Economic Growth",
  "subtopic": "Inequality,Poverty Reduction",
  "teratopic": "Social Development"
}
```

Mapped output (excerpt):
```json
{
  "themes": [
    {"name": "Social Protection", "vocabulary": "World Bank - theme", "uri": ""},
    {"name": "Climate Change", "vocabulary": "World Bank - theme", "uri": ""},
    {"name": "Poverty Reduction", "vocabulary": "World Bank - majtheme", "uri": ""}
  ],
  "topics": [
    {"name": "Economic Growth", "vocabulary": "World Bank - topicv3_name", "uri": ""},
    {"name": "Inequality", "vocabulary": "World Bank - subtopic", "uri": ""},
    {"name": "Poverty Reduction", "vocabulary": "World Bank - subtopic", "uri": ""},
    {"name": "Social Development", "vocabulary": "World Bank - teratopic", "uri": ""}
  ]
}
```

Note: `sectr` "Social Protection" is not added to themes because it duplicates `theme` "Social Protection".

### 6.5 Fields Not Using Source-Tracking

- **identifiers:** Source is already in `type` (e.g., "World Bank GUID", "World Bank Project ID").
- **relations:** Schema has `{name, type}` only; no vocabulary field.
- **languages:** Schema has `{name, code}` only; no vocabulary field.
- **ref_country, geographic_units:** No vocabulary field in schema.

---

## 7. Schema Elements Not Populated from WDS

The following schema elements have no direct WDS source; they require enrichment or manual input:

- `description` (brief description; abstract is distinct)
- `toc`, `toc_structured`
- `license`, `rights`, `copyright`, `usage_terms`, `disclaimer` (can add static WB text)
- `bibliographic_citation` (unless derived from other fields)
- `ref_country` codes (unless countrycode lookup applied)
- `lda_topics`, `embeddings` (ML augmentation)
- `dois`, `isbn`, `issn` (add to identifiers when v3 exposes them)

---

## 8. Implementation Notes

- **API version:** The v3 API uses `guid` for lookup; v2 uses `id`. The docrep_to_schema uses v2; ensure transformation handles both.
- **Date format:** WDS returns ISO 8601 (e.g., `2026-02-14T14:04:52Z`). Schema prefers ISO 8601; use as-is or convert to `YYYY-MM-DD`.
- **Country codes:** `countrycode` (Python) for `count` → ISO3. `countrycode` "1W" → WLD. ISO2 → ISO3 conversion for standard codes.
- **Authors structure:** WDS v3 `authors` is `{"0":{"author":"LEGLA"}}`; v2 may use `author` as array. Handle both.
- **keywd:** Filter empty strings; some entries have `keywd: ""`.

---

## References

- [WDS API v3](https://search.worldbank.org/api/v3/wds)
- [Schema Guide Chapter 5 - Documents](https://worldbank.github.io/schema-guide/chapter05.html)
- [Existing implementation: docrep_to_schema.py](../../search/pipeline/docrep_to_schema.py) (uses v2 API)
- [v3 implementation: wds_to_schema.py](../wds_to_schema.py)
