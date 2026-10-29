---
title: Data Snapshot Metadata Schema v1.4
sidebar_label: Metadata Schema
slug: /data-snapshots/metadata-schema/
description: The current production Data Snapshot Metadata Schema v1.4 defines semantic metadata fields, nested structures, representation conventions, and validation rules.
---

# Data Snapshot Metadata Schema v1.4

## Purpose and scope

The **Data Snapshot Metadata Schema v1.4** defines the structure of semantic metadata used by the [Data Snapshots tool](index.md). Each record describes the analytical content of a single snapshot image, including its title, variables, units, geographic and temporal coverage, and sources. The schema describes what the snapshot represents; it does not transcribe numerical observations, reconstruct table cells, or digitize chart values.

Metadata generation relies on evidence visible within the snapshot image. Filenames, directory names, and parent-document metadata are not supplied to the model as evidence. The `provenance` field describes derivation sources and credited agents visible in the snapshot, rather than source-file coordinates or pipeline execution history.

The [canonical Pydantic models](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/metadata_schema/models.py) define the schema's fields and validation rules. The exported JSON Schema follows Draft 2020-12 and identifies schema version `1.4`. This version is schema metadata and is not a required property of individual records.

## Fields

The schema defines the following top-level metadata fields. All fields are optional and nullable, allowing records to represent information that is unavailable or not applicable to a particular snapshot.

Non-null lists must contain at least one item, and unknown fields are rejected. Although an empty record is structurally valid, it does not necessarily represent a useful or complete metadata extraction.

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

Several metadata fields use nested structures to represent related information while preserving important semantic distinctions.

The `Variable` structure describes measured concepts and their associated properties. Its `unit` field preserves the displayed expression and may include a code and magnitude exponent, while `currency` retains the displayed currency and an optional code. A variable must contain at least a name, unit, currency, or statistical form. Variables assigned analytical or axis roles must also have a name.

The `Dimension` structure describes how measurements are grouped or organized, such as by year, sex, or region. It can represent individual `categories`, one-level category groupings (`category_groups`), and presentation roles within tables.

The following table summarizes selected nested structures and their representation rules.

| Structure | Fields or vocabulary | Rule |
|---|---|---|
| `Dimension.presentation_roles` | `row`, `column` | Describes how categories vary through a table. |
| `Variable.analytical_roles` | `outcome`, `predictor`, `instrumental`, `control` | Requires analytical evidence; table position alone is insufficient. |
| `Variable.axis_roles` | `x_axis`, `y_axis` | Describes ordinary Cartesian axes in a figure. |
| `Variable.multi_axis_assignments` | `dimension`, `position`, `position_index` | Distinguishes multiple axes; x uses top/bottom, y uses left/right, and the index starts at one nearest the plot. |
| Normalized terms | `source_text`, `normalized_value` | Preserves source wording separately from an allowed normalized value. |
| `CodedTerm` | `source_text`, `code`, `scheme`, `uri` | Requires text, a code, or a URI; code and scheme must occur together. |
| `TemporalExpression` | `source_text`, `start`, `end`, `relation`, `precision` | Normalized bounds require a relation and precision; point, interval, and open-ended intervals have different boundary requirements. |
| `GeographicCoverage` | `scope`, `locations`, `level` | Separates the overall area from additional places and reporting level. |
| `Provenance` | `sources`, `attributions` | Separates derivation datasets from credited agents. |

The schema distinguishes between source-visible information and normalized representations. The `source_text` field is reserved for wording or symbols actually visible in the snapshot. For example, an inferred visualization type can be recorded in `normalized_value` without inventing corresponding source text.

Similar distinctions apply to other fields. Temporal coverage describes the period represented by the data, not the document's publication date. Analytical roles, data sources, interpretive notes, and category headings should retain their intended meanings rather than being duplicated across fields.

The complete controlled vocabularies for visualization types, statistical forms, geographic levels, and reporting intervals are defined in the [canonical Pydantic models](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/metadata_schema/models.py). These models also document mappings to relevant standards. Such mappings do not imply that every vocabulary is directly adopted from an external standard or that schema-valid records are certified as compliant with those standards.

## Example and validation

### Example metadata record

The following example illustrates how Data Snapshot Metadata Schema v1.4 represents the analytical content of a real data snapshot.

The snapshot, sourced from a [policy brief on the Shona community in Kenya](https://reliefweb.int/report/kenya/policy-brief-stateless-citizens-journey-shona-community-kenya), compares labor-force status between Urban Nationals and Shona following the COVID-19 outbreak. It presents employed and unemployed populations as percentages across five periods. The extracted image is available in the [Data Snapshot dataset](https://huggingface.co/datasets/ai4data/data-snapshot).

[![Figure 1: Labor force status after the COVID-19 outbreak for people aged 18–64, comparing Urban Nationals and Shona across five periods.](/img/snapshot/policy_brief_on_shona_community_from_stateless_to_citizens_figure_000.png)](/img/snapshot/policy_brief_on_shona_community_from_stateless_to_citizens_figure_000.png)

The metadata record below comes from the human-reviewed [Data Snapshot Metadata Gold](https://huggingface.co/datasets/ai4data/data_snapshot_metadata_gold) dataset and validates against the production Schema v1.4. It describes the chart's variables, population groups, temporal coverage, and sources without transcribing the numerical observations.

[Open the complete metadata JSON](/img/snapshot/policy_brief_on_shona_community_from_stateless_to_citizens_figure_000.json).

```json file=../../website/static/img/snapshot/policy_brief_on_shona_community_from_stateless_to_citizens_figure_000.json
```

### Validating a metadata record

The production schema is implemented using Pydantic. The following example validates an existing `metadata.json` file and prints its contents, excluding fields with null values:

```python
from pathlib import Path

from ai4data.data_snapshot import DataSnapshotMetadata

metadata = DataSnapshotMetadata.model_validate_json(
    Path("metadata.json").read_text(encoding="utf-8")
)
print(metadata.model_dump_json(indent=2, exclude_none=True))
```

This operation does not modify the original file or invoke a language model.

The schema can also be exported as JSON Schema using `DataSnapshotMetadata.model_json_schema(mode="validation")`. The exported schema defines structural constraints, while format assertions require a format-aware validator.

Pydantic additionally applies Unicode NFC normalization, leading and trailing whitespace trimming, and stable exact deduplication. It also validates URI syntax, language-tag formatting and uniqueness, calendar dates, and chronological ordering.

**Schema validation ensures that metadata conforms to the required representation, but does not establish factual accuracy or completeness.** It also does not verify external registry membership or whether extracted values are supported by the source image. These properties require separate checks.

## References

- *Artifact-Driven Metadata Schema Development: A Human-in-the-Loop Methodology with Multimodal LLMs* (publication pending) documents the schema-development methodology. The paper presents an earlier, flat 37-field schema, while the current production Schema v1.4 uses a nested representation.
- [Experimental schema-development workflow](https://github.com/ajdajd/data-snapshot/blob/main/notebooks/schema_discovery/README.md) preserves the research code and versioned artifacts.
- [Canonical Pydantic models](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/metadata_schema/models.py) define every field, vocabulary, example, and validation rule.
- [Metadata extraction](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/metadata_extraction/extraction.py) derives the model-facing response format from the canonical schema and validates completed responses.
- [Metadata file writing](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/metadata_extraction/batch.py) saves validated records as JSON.
- [Schema tests](https://github.com/worldbank/ai4data/blob/main/tests/data_snapshot/test_metadata_schema.py) check public field examples, rejection of unknown fields, and schema version markers.
- [Production README](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/README.md) documents installation and extraction commands.
