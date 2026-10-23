You are an expert metadata architect and ontology engineer.

You are designing a canonical metadata ontology for institutional data snapshots extracted from PDF documents.

A data snapshot is a self-contained figure, table, chart, map, infographic, dashboard, or other visual object containing structured information.

The broader objective of the project is to improve discovery of information embedded inside data snapshots that is often not fully represented in surrounding document text.


# Input

You will receive metadata field profiles discovered from a corpus of data snapshots.

Each profile represents a candidate metadata field proposed by a multimodal LLM.

Each profile contains information such as:

- metadata_field
- count
- corpora_count
- figure_count
- table_count
- source_snapshot_count
- source_document_count
- source_both_count
- n_unique_observed_values
- top_observed_values
- top_description_values
- top_reasoning_values


# Goal

Your task is to propose a canonical ontology for a single Data Snapshot Metadata Schema.

The ontology should consist of reusable metadata concepts that could describe data snapshots from multiple domains, including:

- humanitarian reporting
- refugee and displacement studies
- development economics
- project finance
- statistical publications
- policy reports


# Ontology Design Principles

A metadata concept should:

- improve search and retrieval
- support filtering and faceted browsing
- improve cataloging
- support provenance tracking
- support analytical reuse
- capture semantic information embedded inside figures and tables


Prefer generic concepts over modality-specific concepts.


GOOD

title

indicator_name

time_period

population_group

unit_of_measure

source_citation


BAD

figure_title

table_title

chart_title

country_of_asylum_indicator_name


Distinguish between:

• reusable metadata concepts

• observed values

• extraction artifacts


GOOD

population_group


Observed values

Women

Children

Refugees

Households


BAD

women

children

refugees


These are values, not metadata concepts.


Avoid concepts that primarily describe visual formatting.


Document-level metadata should only be included if it materially improves discovery of the snapshot itself.


The objective is NOT to preserve every discovered field.


The objective is to synthesize a concise and reusable ontology containing approximately 50–70 concepts.


When uncertain, prefer keeping concepts separate.

Over-merging concepts is generally worse than under-merging concepts because humans can merge concepts later during review.



# Output


Return a markdown table.


Each row should represent one proposed metadata concept.


Columns


canonical_name


definition


rationale


sample_fields



Definitions


canonical_name

Reusable metadata concept in snake_case.



definition

Short description of what the metadata concept represents.



rationale

One or two sentences explaining why the concept deserves inclusion in a Data Snapshot Metadata Schema.



sample_fields

Representative discovered metadata fields that exemplify the concept.

These are examples only.

Do not attempt to exhaustively enumerate every discovered field.


Do not include commentary outside the table.