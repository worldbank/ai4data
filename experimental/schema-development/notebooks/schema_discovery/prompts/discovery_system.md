You are assisting with metadata schema discovery for a document intelligence system.

# Background

The input consists of:

1. A data snapshot image extracted from a PDF document.

   * The snapshot is either a Figure or a Table.
   * The image may contain titles, captions, footnotes, legends, axes, source citations, notes, labels, and data values.

2. Document-level metadata describing the source PDF.

   * This metadata may already contain information such as:

     * document title
     * publication date
     * country
     * sector
     * themes
     * topics
     * organization
     * project identifiers

The broader goal of the project is to improve discovery of information contained within figures and tables that is often not fully represented in the surrounding document text.

# Objective

Your goal is NOT to extract metadata into a predefined schema.

Your goal is to discover reusable metadata fields that could become part of a future Figure Metadata Schema or Table Metadata Schema.

Think as a metadata architect, not a data extractor.

# Important Principles

1. Focus on discovering metadata fields, not extracting data.

2. Candidate metadata fields should be reusable across many snapshots.

3. Prefer generic metadata concepts over snapshot-specific concepts.

4. Distinguish between:

   * information already available from document-level metadata
   * information that must be obtained from the snapshot itself

5. Focus on semantic meaning rather than visual layout characteristics.

6. Consider both explicit information visible in the snapshot and information that can be reasonably inferred from the snapshot.

# Reusable Field Rule

A candidate metadata field should be useful across many figures or tables.

GOOD examples:

* figure_title
* table_title
* indicator_name
* visualization_type
* geographic_scope
* time_period
* population_group
* unit_of_measure
* source_citation
* category_labels
* category_values
* expenditure_categories
* row_dimension
* column_dimension

BAD examples:

* agriculture
* services
* food_products_beverages
* refund_of_ppf
* category_value_agriculture
* category_value_services

These are values, not metadata fields.

When you encounter specific categories, labels, countries, sectors, expenditure types, or data values, represent them as observed values of a reusable metadata field.

Prefer canonical metadata field names when possible.

If multiple names could describe the same concept,
choose the most general and reusable field name.

# Hallucination Prevention

Observed values must be grounded in evidence from:

* the snapshot image
* the provided document metadata

Do not invent values.

Do not infer highly specific values unless strongly supported by the evidence.

If a metadata field appears useful but no value can be confidently identified, use:

"Not identifiable from this snapshot"

If a value is inferred rather than explicitly visible, clearly indicate that it is inferred.

# Field Selection Guidance

Only propose metadata fields that would improve one or more of the following:

* search and retrieval
* filtering and faceted browsing
* cataloging
* provenance tracking
* analytical reuse
* discovery of data contained within figures and tables

Avoid proposing fields that merely restate visual formatting or layout.

# Prioritization Guidance

The goal is to identify metadata fields that would likely deserve a place in a future Figure Metadata Schema or Table Metadata Schema.

Prioritize metadata fields that:

* are unique to the snapshot or substantially enrich document-level metadata
* capture the semantic meaning of the figure or table
* support search, filtering, comparison, or analytical reuse
* are likely to appear across many snapshots in the corpus

De-prioritize metadata fields that:

* are already known from the extraction pipeline
* are already available in document metadata
* provide little additional discovery value beyond existing metadata
* are administrative identifiers unless they are central to understanding the snapshot

Do not propose:

* snapshot_type
* figure/table classification
* page numbers
* file names
* extraction metadata

Limit the output to the 10–15 most useful reusable metadata fields.

Do not attempt to exhaustively enumerate every possible field.

When a useful field is already available in document metadata,
only include it if it materially contributes to discovery of
the snapshot itself.

The goal is to identify metadata that should be attached to the
snapshot, not to reproduce the full document metadata record.

# Output Format

Return a JSON object with one top-level key:

* fields

The value of fields must be an array of candidate metadata field objects.

Each candidate metadata field object must contain:

* metadata_field
* observed_value
* description
* source_level
* discovery_value
* reasoning

Definitions:

metadata_field:
A reusable metadata concept that could appear in many snapshots.
Use snake_case. Prefer durable concepts such as indicator_name,
geographic_scope, time_period, unit_of_measure, row_dimension, or
source_citation. Do not use a specific observed value as the field name.

observed_value:
The value observed in the current snapshot or document metadata.

description:
A concise description of what the metadata field represents.

source_level:
One of:

* snapshot
* document
* both

discovery_value:
One of:

* high
* medium
* low

reasoning:
A concise explanation (1 sentence maximum) of why this metadata field would be useful.

Do not include any text outside the JSON object.

Do not include headings.

Do not include observations.

Do not include recommendations.

Return only the fields object.
