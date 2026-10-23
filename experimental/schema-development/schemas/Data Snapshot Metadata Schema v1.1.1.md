# Data Snapshot Metadata Schema v1.1.1

## Overview

The Data Snapshot Metadata Schema v1.1.1 defines a standardized set of metadata fields for describing **data snapshots**—self-contained visual data artifacts extracted from institutional documents, including tables, charts, maps, dashboards, and composite figures.

The schema is intended to support:

- metadata extraction
- indexing and search
- retrieval
- downstream analytics

The schema is intentionally domain-agnostic and is designed to generalize across institutional documents from multiple organizations and subject areas.

The schema was developed through an iterative, human-in-the-loop process combining multimodal large language models, ontology induction, expert review, and quantitative validation.

---

## Revision Notes for v1.1.1

Version 1.1.1 is a documentation-only revision informed by Schema Validation Exercise 1. Human review of the held-out validation results found no meaningful missing metadata fields that required changing the v1.1 field inventory.

Accordingly, v1.1.1:

- preserves every v1.1 field;
- adds no new metadata fields;
- removes or renames no metadata fields;
- clarifies the scope and intended use of existing fields; and
- defers normalized role-bearing and component-based representations to v1.2.

---

# Scope

The schema describes metadata **about a data snapshot**.

It intentionally excludes:

- extracted numerical values
- OCR text
- table cell contents
- chart data reconstruction
- other extraction outputs

Metadata should only be populated when explicitly supported by the snapshot or its accompanying document context.

Metadata extractors **must not infer or hallucinate** values that are not evidenced by the source.

The schema does not duplicate the administrative metadata of the parent document. Parent-document URLs or locators, identifiers, document types, publication dates, and authorship or attribution belong to the linked source-document record in the downstream application. The existing `source_document_title` field provides a human-readable reference to the parent document but does not extend this schema to the parent document as a whole.

The schema also does not introduce dedicated fields for specialized or low-value concepts when they do not materially support the project's cataloging, discovery, or knowledge-graph use cases. Examples reviewed during validation include cartographic scale and coordinates, classification-system details, variable transformations, and panel count.

---

# Applicability

Each metadata field is optional.

Metadata should only be populated when explicitly supported by evidence contained within:

- the snapshot itself; or
- the accompanying source document, where explicitly permitted as supporting context for the snapshot.

Snapshot evidence is primary. Source-document context may clarify the meaning or provenance of the snapshot, but it must not be used to copy unrelated parent-document administrative metadata into the snapshot record.

The absence of a metadata field should be represented as null rather than inferred.

---

# Schema Modules

## Identity & Discovery

### title

**Definition**

The primary title, caption, or heading that identifies the data snapshot.

**Examples**

- Inflation Rate by Country
- Annual Government Expenditure
- Monthly labor income in Afghanistan and remittances from abroad
- Table 6: Determinants of illegal land reallocation at village level

---

### internal_identifier

**Definition**

A document-assigned identifier used to reference the snapshot within the source document.

**Examples**

- Figure 3
- Table 4.2
- Annex B
- Exhibit 7

---

### subject_domain

**Definition**

The broad thematic, policy, or sectoral domain represented by the snapshot.

**Examples**

- Education
- Health
- Macroeconomics
- Agriculture
- Forced Displacement

---

### subject_summary

**Definition**

A concise summary describing the primary analytical subject or purpose of the snapshot.

**Examples**

- Trends in primary school enrollment
- Distribution of humanitarian funding
- Comparison of poverty rates across regions

---

### panel_title

**Definition**

The title or heading of an individual panel within a multi-panel snapshot.

Populate only when panel titles are explicitly present.

**Examples**

- (A) Poverty Rate
- (B) Literacy Rate
- Monthly Returns

---

# Subject & Semantics

### variable_name

**Definition**

The primary variable, indicator, metric, or measured concept represented by the snapshot.

In v1.1.1, this field records the variable's name or measured concept, not a normalized analytical role. Use `category_dimension`, `category_labels`, `row_dimension`, and `column_dimension` where those structural roles apply. Dedicated outcome, predictor, control, or axis-assignment fields are not part of v1.1.1.

**Examples**

- GDP Growth
- Inflation
- Literacy Rate
- Refugee Population

---

### category_dimension

**Definition**

The conceptual variable or dimension used to organize, group, classify, or compare the represented values.

**Examples**

- Country
- Year
- Education Level
- Industry Sector
- Scenario

---

### category_labels

**Definition**

The explicit category names or labels associated with a category dimension.

**Examples**

- Male, Female
- Agriculture, Manufacturing, Services
- Kenya, Uganda, Tanzania
- Low, Medium, High

---

### population_group

**Definition**

The human population, beneficiary group, or demographic group that is the primary subject of the represented data. This field describes who the data are about, not how they are categorized or disaggregated.

**Examples**

- Refugees
- Children under five
- Female respondents
- Host communities
- Technical education graduates

---

# Temporal Context

### time_period

**Definition**

The period or date range represented by the data.

This field describes **when the represented data apply**. It does not describe when the snapshot artifact or parent document was created, prepared, issued, published, revised, or retrieved. When an explicit artifact date appears only as part of a footer or provenance statement, preserve the complete statement in `interpretive_note` rather than treating the date as `time_period`.

**Examples**

- 2015–2020
- FY2023
- January 2024

---

### temporal_granularity

**Definition**

The temporal resolution at which the represented data are reported.

**Examples**

- Annual
- Monthly
- Quarterly
- Daily

---

# Spatial Context

### geographic_scope

**Definition**

The primary geographic area represented by the snapshot.

**Examples**

- Global
- Kenya
- Sub-Saharan Africa
- Latin America

---

### geographic_entities

**Definition**

Named geographic entities explicitly represented within the snapshot.

**Examples**

- Uganda
- Nairobi
- West Africa
- Burkina Faso

---

### geographic_granularity

**Definition**

The administrative or spatial level at which data are reported.

**Examples**

- Country
- Province
- District
- Facility

---

### geographic_role

**Definition**

The semantic role played by geographic entities within the represented data.

**Examples**

- Country of origin
- Host country
- Destination
- Reporting location

---

### location_type

**Definition**

The type of physical location represented.

**Examples**

- Refugee camp
- Hospital
- School
- District

---

# Measurement Context

### unit_of_measure

**Definition**

The unit used to interpret reported quantitative values.

**Examples**

- Percent
- USD
- People
- Kilometers

---

### currency

**Definition**

The currency denomination used for monetary values.

**Examples**

- USD
- EUR
- JPY

---

### measure_type

**Definition**

The statistical form in which values are expressed.

**Examples**

- Count
- Percentage
- Rate
- Index
- Average

---

### comparison_group

**Definition**

The benchmark, comparator, reference group, cohort, scenario, or entity against which the represented data are compared.

Populate only when the snapshot explicitly presents a comparative relationship. This field captures the intended comparison or benchmark represented by the snapshot, not simply the categories used to organize the data.

**Examples**

- Male vs Female
- Rural vs Urban
- Baseline vs Endline
- Treatment vs Control
- Before vs After
- Low-income vs Middle-income vs High-income
- Europe & Central Asia benchmark
- Sub-Saharan Africa benchmark

---

# Structural Organization

### row_dimension

**Definition**

The conceptual variable represented by table rows.

**Examples**

- Country
- Indicator
- Sector

---

### column_dimension

**Definition**

The conceptual variable represented by table columns.

**Examples**

- Year
- Region
- Funding Source

---

### visualization_type

**Definition**

The primary visualization used to encode the represented data.

For a composite or multi-panel snapshot, record a concise description of the overall visualization type or visible combination when no single type adequately describes the artifact. Use `panel_title` for explicit panel headings. v1.1.1 does not separately encode component-to-type relationships or `panel_count`; those are normalization concerns for v1.2.

**Examples**

- Bar chart
- Line chart
- Table
- Map
- Heatmap
- Composite figure: line charts and map

---

# Provenance & Attribution

### data_source

**Definition**

The named dataset, survey, publication, organization, or credited agent from which the represented data originate or which is explicitly credited with producing the snapshot artifact.

In v1.1.1, `data_source` may retain multiple names but does not structurally distinguish roles such as data source, data producer, map maker, preparer, or contributor. A role-bearing representation is deferred to v1.2. Do not copy the parent document's authors or publisher into this field solely because they are associated with the document; the source or attribution must be explicitly relevant to the snapshot or its represented data.

**Examples**

- World Development Indicators
- DHS
- UNHCR Registration Data
- National Census
- Map Design Unit

---

### source_document_title

**Definition**

The title of the parent document containing the snapshot.

This field is a human-readable reference to the linked parent document. Other parent-document administrative metadata—including its URL, identifier, document type, publication date, authors, and publisher—remain on the source-document record and are not duplicated in v1.1.1.

**Examples**

- World Development Report 2024
- Global Trends Report

---

### language

**Definition**

The language used within the snapshot.

**Examples**

- English
- French
- Arabic

---

### interpretive_note

**Definition**

Explanatory, methodological, uncertainty, or provenance statements explicitly provided within the snapshot that aid interpretation or traceability.

This field may preserve complete notes containing sample-size statements, explanations of confidence intervals, standard errors or uncertainty bands, and footer statements that include an artifact date or production credit. It retains the statement as text; it does not create separate structured fields for sample size, uncertainty representation, or artifact publication date.

Populate only when such notes are explicitly present.

**Examples**

- Values are provisional.
- Estimates exclude informal employment.
- Data collected using 2022 census boundaries.
- Sample: 1,204 respondents.
- Shaded areas show 95% confidence intervals.
- Prepared by the Map Design Unit, March 2024.

---

# Project & Operational Context

### project_name

**Definition**

The project, program, operation, or initiative associated with the snapshot.

**Examples**

- Niger - COVID-19 Emergency Response Project
- Jordan Health Sector Reform Project
- Lebanon - Health Resilience Project

---

### project_identifier

**Definition**

The formal identifier assigned to the associated project or operation.

**Examples**

- P171254
- P178944

---

### project_component

**Definition**

The project component, workstream, or results area represented by the snapshot.

**Examples**

- Component 3: Project management
- Results Area 1

---

### intervention_type

**Definition**

The intervention, service, policy, or operational activity represented.

**Examples**

- Cash Transfer
- Vaccination
- School Construction

---

### financial_measure

**Definition**

The financial quantity or funding-related measure represented by the snapshot.

**Examples**

- Project Cost
- Disbursement
- Financing Gap
- Budget Allocation

---

### financing_source

**Definition**

The organization or funding source providing financial support.

**Examples**

- IDA
- IBRD
- Government
- European Union

---

### financing_instrument

**Definition**

The financing mechanism associated with the represented activity.

**Examples**

- Grant
- Loan
- Credit
- Trust Fund

---

# Analytical & Methodological Context

### analysis_method

**Definition**

The analytical, statistical, or computational method used to produce the reported results.

Populate only when explicitly stated.

**Examples**

- Difference-in-Differences
- Regression
- Tobit model
- Cost-Benefit Analysis

---

### data_collection_method

**Definition**

The method or instrument used to collect the underlying data.

Populate only when explicitly stated.

**Examples**

- Household Survey
- Administrative Records
- Key Informant Interviews
- Census

---

# Metadata Extraction Guidelines

Metadata extractors should follow these principles:

1. Populate metadata only when supported by explicit evidence.

2. Do not infer, hallucinate, or fabricate metadata.

3. Prefer semantic meaning over literal wording.

4. Multiple values may be assigned where appropriate (for example, multiple geographic entities).

5. Preserve original terminology whenever practical.

6. Record missing metadata as null rather than inferred values.

7. The schema describes the snapshot—not the parent document as a whole. Use source-document context only to support interpretation of the snapshot or populate an explicitly permitted field.

8. Numerical observations, OCR text, reconstructed table contents, and operational cell values belong to extraction outputs and are outside the scope of this metadata schema. An explicit methodological or provenance note that contains a number, such as a sample-size statement, may still be preserved in `interpretive_note`.

9. Use the existing variable and dimension fields for names and structural organization. Do not create ad hoc fields for analytical roles or axis assignments in v1.1.1.

10. Represent the core type of a composite snapshot in `visualization_type` and its explicit panel headings in `panel_title`. Do not add a separately authored `panel_count`; it may be derived after component normalization in v1.2.

11. Do not create unlisted fields for parent-document administration, specialized classification semantics, cartographic properties, or other concepts excluded by the intended application.

---

Version: **1.1.1**

Status: Stable — documentation revision following Schema Validation Exercise 1

Last Updated: September 2026
