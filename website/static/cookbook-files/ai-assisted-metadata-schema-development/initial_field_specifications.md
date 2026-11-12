# Selected initial field specifications — v1.1

Evidence status: selected transcription from the initial Data Snapshot Metadata
Schema v1.1. The field blocks below preserve its definitions, field-specific
instructions, and illustrative examples. The examples are specification examples;
they are not all recorded annotations of the cookbook's selected artifacts.
This excerpt is not the full 36-field specification or a later revised schema.

Research attribution: Dy & Solatorio (2026). Artifact-Driven Metadata Schema
Development: A Human-in-the-Loop Methodology with Multimodal LLMs. Working paper.
The accompanying initial research specification supplies the field wording.

## General rules from the initial specification

Each metadata field is optional. Metadata should only be populated when explicitly
supported by the snapshot or its accompanying document context. Accompanying
source-document evidence is used where explicitly permitted. Missing metadata is
represented as null. Unsupported values must not be inferred or fabricated.
These sentences summarize the initial applicability and extraction rules.

## Verbatim field excerpts

### title

**Definition**

The primary title, caption, or heading that identifies the data snapshot.

**Examples**

- Inflation Rate by Country
- Annual Government Expenditure
- Monthly labor income in Afghanistan and remittances from abroad
- Table 6: Determinants of illegal land reallocation at village level

### panel_title

**Definition**

The title or heading of an individual panel within a multi-panel snapshot.

Populate only when panel titles are explicitly present.

**Examples**

- (A) Poverty Rate
- (B) Literacy Rate
- Monthly Returns

### variable_name

**Definition**

The primary variable, indicator, metric, or measured concept represented by the snapshot.

**Examples**

- GDP Growth
- Inflation
- Literacy Rate
- Refugee Population

### time_period

**Definition**

The period or date range represented by the data.

**Examples**

- 2015–2020
- FY2023
- January 2024

### interpretive_note

**Definition**

Explanatory notes, caveats, assumptions, footnotes, or methodological remarks explicitly provided within the snapshot to aid interpretation.

Populate only when such notes are explicitly present.

**Examples**

- Values are provisional.
- Estimates exclude informal employment.
- Data collected using 2022 census boundaries.
