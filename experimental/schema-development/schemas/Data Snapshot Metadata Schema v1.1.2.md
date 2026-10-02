# Data Snapshot Metadata Schema v1.1.2

Status: Standards-informed flat specification

## Overview

The Data Snapshot Metadata Schema v1.1.2 defines 37 metadata fields for
describing a data snapshot: a self-contained table, chart, map, dashboard, or
composite figure extracted from an institutional document.

The schema remains flat so that its fields, definitions, and populated records
can be inspected directly. Standards alignment refines its terminology,
semantic boundaries, and usage guidance without requiring one external
vocabulary to cover every field.

## Revision notes

Version 1.1.2 is a standards-informed revision of the validated v1.1.1 schema.
It:

- preserves the accepted information coverage;
- adopts clearer application-facing field names where the standards review
  identified ambiguity or multiplicity;
- splits the overloaded `data_source` field into `data_sources` and
  `attributions`;
- clarifies the resource, context, and evidence described by each field;
- records relevant standards mappings and value guidance; and
- preserves open source-grounded values when no exact external vocabulary is
  appropriate.

## Scope

The schema describes metadata about one data snapshot. It excludes extracted
numerical observations, OCR text, reconstructed chart data, table-cell
contents, and other content-extraction outputs.

Metadata must be supported by the snapshot or, where a field explicitly
permits it, by relevant source-document context. Snapshot evidence remains
primary. Parent-document URLs, identifiers, document types, publication dates,
authors, and publishers are not duplicated in the snapshot record.

## Applicability and cardinality

Every field is optional because different artifact types expose different
metadata. Missing values are represented as null rather than inferred.

The notation `0..1` indicates an optional single text value. The notation
`0..*` indicates an optional ordered list of text values. The schema does not
encode relationships among values stored in different flat fields.

## Standards-informed representation principles

1. Populate metadata only when supported by explicit evidence.
2. Preserve source-visible terminology when normalization would remove or
   distort information.
3. Use an established term, format, or code only when the mapping is exact and
   remains understandable in the flat field.
4. Do not force a value into a controlled vocabulary when the mapping is
   ambiguous or domain-dependent.
5. Do not infer temporal precision, geographic level, analytical role, or
   semantic relationships that the artifact does not establish.
6. Preserve source order for ordered values and remove only exact duplicates
   within the same field.
7. Do not translate source-visible text as part of metadata recording.

# Schema modules

## Identity and discovery

| Field | Cardinality | Definition | Standards alignment and usage guidance |
|---|---:|---|---|
| `title` | 0..1 | The primary title, caption, or heading that identifies the data snapshot. | Aligns with DCMI `title` and Schema.org `name`. Preserve the snapshot title rather than substituting the parent-document title. |
| `document_label` | 0..1 | The label assigned to the snapshot within its source document. | JATS `label` is the closest pattern; general identifier properties are broader. Use for labels such as “Figure 3” or “Table 4.2,” not for global identifiers. |
| `subject_domains` | 0..* | The broad thematic, policy, or sectoral domains represented by the snapshot. | Related to DCMI `subject`, Schema.org `about`, DCAT `theme`, and SKOS concepts. Use recognizable domain labels without requiring one universal vocabulary. |
| `subject_summary` | 0..1 | A concise summary of the snapshot's primary analytical subject or purpose. | Schema.org `abstract` is a close match; DCMI `description` is broader. Summarize the snapshot rather than the parent document. |
| `panel_titles` | 0..* | The titles or headings explicitly shown for individual panels in a multi-panel snapshot. | JATS figure-group and caption-title patterns provide the closest structural analogue. Preserve explicit titles in source order and do not infer titles for unlabeled panels. |

## Subject and semantics

| Field | Cardinality | Definition | Standards alignment and usage guidance |
|---|---:|---|---|
| `variable_names` | 0..* | The variables, indicators, metrics, or measured concepts represented by the snapshot. | Closely aligned with Schema.org `variableMeasured`, DDI Variable, and SDMX measure concepts. Record represented concepts, not inferred analytical roles. |
| `category_dimensions` | 0..* | The conceptual variables or dimensions used to organize, group, classify, or compare represented values. | Closely aligned with SDMX Dimension, DDI RepresentedVariable, and the RDF Data Cube dimension pattern. |
| `category_labels` | 0..* | The explicit category names or labels associated with the represented category dimensions. | SDMX Code/Codelist, DDI Category/CodeList, and SKOS labels provide relevant patterns. Preserve displayed labels and source order; do not add a code without a known scheme. |
| `population_group` | 0..1 | The human population, beneficiary group, or demographic group that is the primary subject of the represented data. | Closely aligned with DDI Universe and Schema.org `populationType`. Describe who the data concern rather than a category used only to organize them. |

## Temporal context

| Field | Cardinality | Definition | Standards alignment and usage guidance |
|---|---:|---|---|
| `time_period` | 0..1 | The date, interval, or source expression describing when the represented data apply. | Aligns with Schema.org `temporalCoverage`, DCMI `temporal`, and SDMX `TIME_PERIOD`. Preserve the complete source expression and do not infer unsupported bounds or precision. |
| `temporal_granularity` | 0..1 | The temporal resolution or reporting interval of the represented data. | SDMX frequency, DCAT temporal resolution, and OWL-Time provide related terminology. Use an SDMX frequency only when exact; preserve irregular or event-based descriptions when appropriate. |

## Spatial context

| Field | Cardinality | Definition | Standards alignment and usage guidance |
|---|---:|---|---|
| `geographic_scope` | 0..1 | The overall place or geographic area that the snapshot principally covers or concerns. | Aligns with Schema.org `spatialCoverage`, DCMI `spatial`, and DCAT spatial coverage. The scope may be broader than the named entities shown within it. |
| `geographic_entities` | 0..* | Named geographic entities explicitly represented within the snapshot. | Closely aligned with Schema.org Place and DDI GeographicLocation. Preserve displayed names; use a standardized name or code only when the mapping is unambiguous. |
| `geographic_level` | 0..1 | The administrative, geographic, or reporting level at which the data are represented. | Closely aligned with DDI GeographicLevel and, where applicable, ISO 3166-2. Do not infer an administrative level without sufficient jurisdictional context. |
| `geographic_roles` | 0..* | The explicit semantic roles played by geographic entities, such as country of origin, host country, or destination. | No reviewed vocabulary provides a universal exact match. Preserve source-grounded roles rather than imposing a closed vocabulary. |
| `location_types` | 0..* | The physical, administrative, or operational kinds of locations represented, such as camp, school, settlement, or district. | IATI LocationType and SKOS provide optional patterns. Do not force an approximate external category. |

## Measurement context

| Field | Cardinality | Definition | Standards alignment and usage guidance |
|---|---:|---|---|
| `units_of_measure` | 0..* | The units needed to interpret the reported quantitative values. | Aligns with SDMX `UNIT_MEASURE`, Schema.org `unitCode`, and UN/CEFACT Recommendation 20. Preserve displayed units and apply a code only when exact. |
| `currencies` | 0..* | The currency denominations used for monetary values. | Aligns with Schema.org `currency`, ISO 4217, and the SDMX currency list. Prefer an unambiguous ISO 4217 alphabetic code while preserving source wording when necessary. |
| `statistical_forms` | 0..* | The statistical or quantitative forms in which values are expressed, such as count, percentage, rate, arithmetic mean, or index. | Grounded in SDMX statistical operations and DDI Summary Statistic Type, with Schema.org `statType` as an interoperability relation. Use precise terms when the source supports them. |
| `comparisons` | 0..* | The explicit comparators, benchmarks, reference groups, cohorts, scenarios, entities, or complete comparative relationships presented by the snapshot. | No reviewed term covers the complete local meaning. Preserve expressions such as “Male vs Female” and named comparators such as “Europe & Central Asia benchmark” without inventing an implicit comparison side. |

## Structural organization

| Field | Cardinality | Definition | Standards alignment and usage guidance |
|---|---:|---|---|
| `row_dimensions` | 0..* | The conceptual variables represented by table rows. | General statistical-dimension terms are broader. Record only explicit row assignments. |
| `column_dimensions` | 0..* | The conceptual variables represented by table columns. | General statistical-dimension terms are broader. Record only explicit column assignments. |
| `visualization_types` | 0..* | The visualizations or visible combination of visualization forms used by the snapshot. | DCMI `type` and Schema.org `additionalType` are broader; Vega-Lite provides useful terminology for several forms. Use the most specific recognizable type while permitting unfamiliar explicit forms. |

## Provenance and attribution

| Field | Cardinality | Definition | Standards alignment and usage guidance |
|---|---:|---|---|
| `data_sources` | 0..* | The named datasets, surveys, publications, or other entities from which the represented data originate. | Reflects the derivation distinction expressed by PROV-O `wasDerivedFrom` and DCMI `source`. Do not copy parent-document authors or publishers solely because they are associated with the document. |
| `attributions` | 0..* | The agents explicitly credited with producing or contributing to the snapshot artifact. | Reflects PROV-O `wasAttributedTo` and related Schema.org attribution properties. Preserve an explicit role together with the credited agent as a complete source-grounded expression. |
| `source_document_title` | 0..1 | The title of the parent document containing the snapshot. | Aligns with DCMI `title` and Schema.org `name` at the parent-document resource level. This is a human-readable reference; other parent-document administrative metadata remain outside the snapshot record. |
| `languages` | 0..* | The languages explicitly used within the snapshot. | Aligns with DCMI `language`, Schema.org `inLanguage`, and IETF BCP 47. Use recognizable language names or exact tags without inferring an unexpressed region or script. |
| `interpretive_notes` | 0..* | Explanatory, methodological, uncertainty, sample-size, or provenance statements explicitly provided with the snapshot that aid interpretation or traceability. | Related to DCMI and Schema.org description properties and SDMX annotations. Preserve complete statements rather than decomposing them into unsupported fields. |

## Project and operational context

| Field | Cardinality | Definition | Standards alignment and usage guidance |
|---|---:|---|---|
| `project_name` | 0..1 | The project, program, operation, or initiative associated with the snapshot. | Schema.org Project is narrower; IATI activity patterns are applicable in specific domains. Preserve the broader local scope. |
| `project_identifiers` | 0..* | The formal identifiers assigned to the associated project or operation. | Aligns with Schema.org `identifier` and IATI activity identifiers. Preserve the assigned value and identify its scheme in the text when explicitly available. |
| `project_components` | 0..* | The project components, workstreams, results areas, or other subordinate parts represented by the snapshot. | Schema.org `hasPart` and IATI related-activity patterns are narrower. Preserve the displayed component terminology. |
| `intervention_types` | 0..* | The interventions, services, policies, or operational activities represented by the snapshot. | No universal exact match exists. IATI classifications and SKOS schemes may be used when their domain and meaning fit exactly. |
| `financing_measures` | 0..* | The financial quantities or funding-related measures represented, such as project cost, allocation, disbursement, or funding gap. | IATI transaction and budget classifications and SDMX measure patterns cover narrower subsets. Preserve each specific financial meaning. |
| `funders` | 0..* | The organizations or other funding sources explicitly identified as providing financial support. | Aligns with Schema.org `funder` and IATI funding-role organizations. Preserve displayed funder names. |
| `financing_instruments` | 0..* | The financing mechanisms associated with the represented activity. | Closely aligned with IATI FinanceType and Schema.org Grant and LoanOrCredit patterns. Apply an external code only when the domain and mapping are exact. |

## Analytical and methodological context

| Field | Cardinality | Definition | Standards alignment and usage guidance |
|---|---:|---|---|
| `analysis_methods` | 0..* | The explicitly stated analytical, statistical, or computational methods used to produce the reported results. | Schema.org `measurementTechnique`, PROV-O activity patterns, and discipline-specific vocabularies are related but not universally equivalent. Preserve explicitly stated methods. |
| `data_collection_methods` | 0..* | The explicitly stated methods or instruments used to collect the underlying data. | DDI Mode of Collection provides the closest established vocabulary; Schema.org `measurementMethod` is also relevant. Apply a controlled term only when exact. |

## Metadata recording guidelines

1. Populate only fields supported by the available evidence.
2. Multiple values may be assigned to fields with `0..*` cardinality.
3. Preserve the relationship implied by a complete source phrase when the flat
   schema cannot encode that relationship separately.
4. Do not create unlisted fields for numerical observations, table cells,
   chart reconstruction, parent-document administration, or specialized
   concepts outside the intended descriptive scope.
5. An explicit note containing a number, such as a sample-size statement, may
   be preserved in `interpretive_notes`; this does not make numerical
   observation extraction part of the schema.

Version: **1.1.2**

Status: Standards-informed flat specification
