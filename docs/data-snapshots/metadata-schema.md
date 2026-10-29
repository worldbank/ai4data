---
title: Data Snapshot Metadata Schema v1.4
sidebar_label: Metadata Schema
slug: /data-snapshots/metadata-schema/
description: The current production Data Snapshot Metadata Schema v1.4 defines semantic metadata fields, nested structures, representation conventions, and validation rules.
---

# Data Snapshot Metadata Schema v1.4

## Purpose and scope

This page documents the current production schema used by the
[Data Snapshots tool](index.md). A record describes the analytical content visible
in one snapshot image, including its title, variables, units, context, and sources.
The schema describes the represented data; it does not transcribe numerical
observations, reconstruct table cells, or digitize chart values.

The metadata extractor uses image-visible evidence. Filenames, directory names,
and parent-document metadata are excluded from its evidence. The `provenance`
field describes derivation sources and credited agents visible in the snapshot,
not source-file coordinates or the pipeline's execution history.

**Data Snapshot Metadata Schema v1.4** describes one snapshot using its visible
content. The canonical validator and field definitions are the
[Pydantic models](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/metadata_schema/models.py).
The exported JSON Schema identifies Draft 2020-12 and schema version `1.4`.
The version is schema metadata, not a required property of each record.

## Fields

All top-level fields are optional and nullable. Missing and null values represent
unavailable information. Non-null lists require at least one item, and unknown
fields are rejected. An empty top-level record is structurally valid; it does not
demonstrate a useful or complete extraction.

| Field | Non-null type | Meaning |
|---|---|---|
| `title` | String | Primary visible title, caption, or heading. |
| `document_label` | String | Visible label such as `Table 3` or `Figure 4.2`. |
| `subject_domains` | List of strings | Broad thematic or sectoral domains. |
| `subject_summary` | String | Concise description of the analytical subject. |
| `panel_titles` | List of strings | Explicit headings of panels within a composite. |
| `variables` | List of `Variable` | Measured concepts with units, currencies, roles, and statistical forms. |
| `dimensions` | List of `Dimension` | Groupings, categories, category groups, and table presentation roles. |
| `population_group` | String | Human population represented by the data. |
| `visualization_types` | List of `VisualizationTypeTerm` | Visual forms, including identifiable component types in composites. |
| `temporal_coverage` | `TemporalCoverage` | Represented-data period and reporting interval. |
| `geographic_coverage` | `GeographicCoverage` | Overall scope, named locations, and reporting level. |
| `comparisons` | List of strings | Explicit comparative relationships or named comparators. |
| `provenance` | `Provenance` | Derivation sources and credited agents visible in the image. |
| `languages` | List of `Language` | Languages of the snapshot's visible text. |
| `interpretive_notes` | List of strings | Complete substantive notes not fully represented elsewhere. |
| `project` | `Project` | Project name, identifiers, and components. |
| `intervention_types` | List of strings | Represented interventions or activities. |
| `financing` | `Financing` | Financing measures, funders, and instruments. |
| `analysis_methods` | List of strings | Explicit analytical methods. |
| `data_collection_methods` | List of `CodedTerm` | Explicit data-collection methods. |

## Nested structures and field boundaries

`Variable` keeps the measured concept separate from its qualifiers. `unit` retains
the displayed expression and can add a code and magnitude exponent; `currency`
retains currency wording and an optional code. A variable requires a name, unit,
currency, or statistical form. Analytical and axis roles additionally require a
name. `Dimension` describes what groups or organizes the measures, such as year,
sex, or region, including visible `categories` and `category_groups`.

| Structure | Fields or vocabulary | Rule |
|---|---|---|
| `Dimension.presentation_roles` | `row`, `column` | Describes how categories vary through a table. |
| `Variable.analytical_roles` | `outcome`, `predictor`, `instrumental`, `control` | Requires analytical evidence; table position alone is insufficient. |
| `Variable.axis_roles` | `x_axis`, `y_axis` | Describes ordinary Cartesian axes in a figure. |
| `Variable.multi_axis_assignments` | `dimension`, `position`, `position_index` | Distinguishes multiple axes; x uses top/bottom, y uses left/right, and the index starts at one nearest the plot. |
| Normalized terms | `source_text`, `normalized_value` | Preserves source wording separately from an allowed normalized value. |
| `CodedTerm` | `source_text`, `code`, `scheme`, `uri` | Requires text, a code, or a URI; code and scheme must occur together. |
| `TemporalExpression` | `source_text`, `start`, `end`, `relation`, `precision` | Normalized bounds require a relation and precision; point, interval, and open-ended interval have different boundary requirements. |
| `GeographicCoverage` | `scope`, `locations`, `level` | Separates the overall area from additional places and reporting level. |
| `Provenance` | `sources`, `attributions` | Separates derivation datasets from credited agents. |

The complete vocabularies for visualization types, statistical forms, geographic
levels, and reporting intervals are defined in the canonical models. Their field
metadata also records links to standards and the relationship of each mapping.
Those mappings do not imply that every application-owned vocabulary is a direct
copy of an external standard or that a record is certified by that standard.

`source_text` is reserved for wording or symbols actually visible in the image.
An inferred visualization form belongs in `normalized_value`, without invented
source wording. Temporal coverage describes when the represented data apply,
not the document's publication date. Roles, sources, notes, and category headings
retain their distinct meanings rather than being copied into several fields.

## Example and validation

This illustrative record describes an imagined table headed "Population by
year", labeled "Table 1", with years in columns, units of persons, and visible
coverage of 2020–2024. It contains no numerical observations and is not a model
output or a gold annotation.

```json
{
  "title": "Population by year",
  "document_label": "Table 1",
  "variables": [{"name": "Population", "unit": {"source_text": "Persons"}}],
  "dimensions": [{"name": "Year", "presentation_roles": ["column"]}],
  "visualization_types": [{"normalized_value": "table"}],
  "temporal_coverage": {
    "period": {
      "source_text": "2020–2024",
      "start": "2020",
      "end": "2024",
      "relation": "interval",
      "precision": "year"
    }
  },
  "languages": [{"tag": "en"}]
}
```

For an existing metadata file named `metadata.json`, the following code validates
it and prints a representation that omits unavailable values. It does not modify
the file or call a model:

```python
from pathlib import Path

from ai4data.data_snapshot import DataSnapshotMetadata

metadata = DataSnapshotMetadata.model_validate_json(
    Path("metadata.json").read_text(encoding="utf-8")
)
print(metadata.model_dump_json(indent=2, exclude_none=True))
```

`DataSnapshotMetadata.model_json_schema(mode="validation")` returns the schema
for the installed version. JSON Schema carries the exported structural rules;
format assertions require a format-aware validator. Python additionally applies
Unicode NFC normalization, outer whitespace trimming, stable exact deduplication,
and checks for URI syntax, language-tag casing and uniqueness, calendar dates,
and chronological ordering. Neither validator establishes external registry
membership, source-grounding, or semantic correctness.

## Relationship to schema-development research

*Artifact-Driven Metadata Schema Development: A Human-in-the-Loop Methodology
with Multimodal LLMs* describes an earlier, flat schema developed through
artifact-driven discovery and human adjudication. Publication on arXiv is pending.

Production Schema v1.4 uses the nested structures and validation rules documented
here. The paper's historical validation findings do not independently validate
all subsequent representation and implementation changes. The production
Pydantic models remain the source of truth for this contract.

The [experimental schema-development workflow](https://github.com/ajdajd/data-snapshot/blob/main/notebooks/schema_discovery/README.md)
preserves the research process and versioned artifacts. The research package's
Unified Evaluation Schema v1.3 is a separate contract for layout annotations and
predictions; it is not this semantic metadata schema.

## Implementation references

- [Canonical Pydantic models](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/metadata_schema/models.py) define every field, vocabulary, example, and validation rule.
- [Metadata extraction](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/metadata_extraction/extraction.py) derives the model-facing response format from the canonical schema and validates completed responses.
- [Metadata file writing](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/metadata_extraction/batch.py) saves validated records as JSON.
- [Schema tests](https://github.com/worldbank/ai4data/blob/main/tests/data_snapshot/test_metadata_schema.py) check public field examples, rejection of unknown fields, and schema version markers.
- [Production README](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/README.md) documents installation and extraction commands.
