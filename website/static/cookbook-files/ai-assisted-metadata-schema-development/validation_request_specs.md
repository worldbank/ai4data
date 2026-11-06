# Validation request specifications: teaching adaptations

These independently written, unexecuted specifications explain two historical
assessment tasks. They are not complete historical prompts, provider payloads,
or live runners. Each request needs one image, its identity, corpus and artifact
type, available parent-document context, and the complete frozen schema.
Selected field excerpts cannot substitute for that complete input.

Research attribution: Dy & Solatorio (2026). Artifact-Driven Metadata Schema
Development: A Human-in-the-Loop Methodology with Multimodal LLMs. Working paper.
Appendix E supplies the methodological and output-contract reference.

## Open residual discovery

Treat the supplied operational schema as frozen. Independently identify reusable
metadata concepts explicitly evidenced by this snapshot or relevant document
context, then compare each with the exact schema definitions. Record concepts
whose coverage is substantively at issue; do not inventory routine extraction
outputs. Do not infer absent information or import an evolving vocabulary from
earlier assessments.

Return observations with observation_id, metadata_concept, evidence,
evidence_source, closest_schema_fields, fit_status, and fit_rationale.
evidence_source is snapshot, document, or both. fit_status is covered, weak_fit,
no_fit, out_of_scope, or uncertain. Use exact frozen field names. Covered and
weak-fit observations require a closest field.

Return candidate_new_fields only for in-scope observations with weak_fit or
no_fit. Each proposal contains proposed_name, definition,
supporting_observation_ids, why_existing_fields_are_insufficient,
operational_value, and source_level. Every support identifier must resolve to
a qualifying observation within that snapshot's assessment. Proposed names
must be reusable snake_case names absent from the frozen inventory.

The historical prompt allowed document-level candidates when they materially
enabled snapshot identification, retrieval, verification, or interpretation.
Human review subsequently applied the application's linked-document boundary.
Do not silently replace that broad discovery behavior with the narrower rule
used by the second instrument.

No proposal accepts a new field or revises the frozen schema.

## Confirmatory consequential-gap assessment

Consider the entire supplied schema, but do not inventory covered information.
Assess whether reusable snapshot metadata lacks adequate coverage and whether
the loss would materially impair interpretability, discoverability, or both.
Compare faithful representation of the evidence with the best representation
permitted by the closest existing fields. Do not invent an application or user
workflow to make a proposal appear consequential.

A critical gap requires explicit evidence, reusable snapshot metadata, inadequate
existing-field coverage, and a concrete material consequence. A possible gap
requires plausible reusable metadata and plausible material impact while
retaining uncertainty about evidence, coverage, or materiality. Additional
convenience or a possible search facet alone does not meet the threshold.

Parent-document administration remains on the linked document record unless
the evidence establishes a distinct snapshot-level role. An explicit explanatory
note may qualify as metadata even when it contains a number.

Return coverage_assessment and critical_or_possible_gaps. A no_critical_gap_found
assessment has an empty array. A possible_gap assessment has one or more
possible gaps and no critical gaps. A critical_gap_found assessment contains
at least one critical gap.

Each gap contains gap_id, gap_status, missing_metadata_concept,
proposed_field_name, evidence, why_snapshot_metadata, closest_schema_fields,
why_existing_fields_may_be_insufficient, material_impact, material_consequence,
and uncertainty_note. Use exact field names and preserve uncertainty. Do not
classify the finding as conceptual, documentation, or representation-related;
human adjudication makes that distinction. Do not accept or implement a field.

## Historical configuration and limits

Validation 1 supplied the complete initial v1.1 specification. Validation 2
supplied the complete documentation-revised v1.1.1 through a model-facing copy
that omitted validation history and prior outcomes while preserving substantive
scope, field definitions, examples, and guidance.

Both used GPT-5.5, medium reasoning, Flex service, prompt caching, an 8,000-token
output ceiling, and Pydantic Structured Outputs, one snapshot per request.
These document the historical execution. They are not current model or price
recommendations. Fresh execution requires separate integration, an explicit
budget, and new output locations; results may differ.

Technical output checks do not establish semantic correctness or sensitivity.
The two instruments reused the same 202 held-out snapshots. Calibration and
ablation results remain separate from substantive full-run findings.
