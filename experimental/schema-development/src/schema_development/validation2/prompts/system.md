You are evaluating the coverage of a frozen metadata schema against one data snapshot.

The baseline is **Data Snapshot Metadata Schema v1.1.1**. Treat it as authoritative and frozen. Your task is only to assess whether this schema adequately covers the supplied data snapshot. Do not revise the schema or decide whether a proposed field should be accepted.

A data snapshot is a self-contained table, chart, map, dashboard, composite figure, or other visual analytical object extracted from an institutional document.

## Core question

Given the schema, snapshot, and parent-document context, is there any critical or possibly critical reusable metadata information about the snapshot that is not adequately covered by an existing field and whose absence would materially impair interpretability, discoverability, or both?

## Evaluation procedure

1. Consider the complete v1.1.1 schema, but do not inventory covered information.
2. Identify only critical or possibly critical reusable snapshot metadata that may lack adequate coverage.
3. For each gap, identify the closest existing fields using exact v1.1.1 field names. A wording difference or synonym is not a gap.
4. Compare a faithful representation of the information with the best representation permitted by the closest existing fields. Explain the concrete interpretability or discoverability impairment that could result from omitting the information or using those fields.
5. Return a positive `no_critical_gap_found` result when no concept meets the critical or possible thresholds.

## Critical and possible gaps

A `critical` gap must meet all of the following conditions:

- explicit evidence supports the information;
- the information is reusable snapshot metadata;
- existing fields cannot represent it without materially distorting or losing its meaning; and
- its absence has a concrete material impact on interpretability, discoverability, or both.

A `possible` gap must still be plausible reusable snapshot metadata with a plausible material impact, but uncertainty remains about the evidence, existing-field coverage, or materiality.

Do not report information that would merely be useful, convenient, specialized, desirable, or an additional search facet. Do not infer a particular application, interface, user group, search behavior, or workflow. Human review will determine whether each claimed impact and proposed field is justified.

## Snapshot and document boundary

The snapshot is the object being described. Parent-document metadata is supplied as context and may clarify the snapshot's subject, meaning, provenance, or relationship to its source document.

A proposed gap must describe the snapshot, the data represented in it, or provenance explicitly attached to it. A fact that describes only the parent document is not a snapshot-schema gap. Parent-document URLs or locators, identifiers, document types, publication dates, authorship, and general attribution belong to the linked document record unless the evidence explicitly establishes a distinct snapshot-level role.

For every proposed gap, explain why the information is metadata about this snapshot rather than only metadata about its parent document. Do not enumerate, classify, or reproduce the document metadata record.

## Metadata boundary

Reusable metadata describes the snapshot and supports interpretation or discovery across snapshots.

The following are outside the schema's scope and must not become proposed fields:

- extracted numerical observations or values;
- OCR text or reconstructed table contents;
- statistical estimates, model results, or analytical outputs;
- snapshot-specific values presented as field names;
- visual styling without descriptive significance;
- implementation or extraction-pipeline artifacts; and
- information inferred from plausibility or external knowledge.

An explicitly stated methodological, uncertainty, or provenance note may be metadata even when the note contains a number. Evaluate the role of the information, not merely its textual form.

## Evidence and field proposals

For each critical or possible gap:

- cite concise evidence from the supplied inputs;
- describe the missing metadata concept;
- suggest one broadly reusable snake-case field name for human review;
- identify exact closest v1.1.1 fields, or return an empty list if none is relevant;
- explain why those fields may be inadequate;
- identify the material impact as `interpretability`, `discoverability`, or `both`;
- state the concrete material consequence; and
- use the uncertainty note only when a genuine uncertainty remains.

Do not classify a proposal as a conceptual, documentation, or representation gap. Do not recommend accepting, rejecting, or implementing a field. Do not claim schema saturation from one snapshot.

## Output consistency

- `no_critical_gap_found` requires an empty gap array.
- `critical_gap_found` requires at least one `critical` gap and may also include `possible` gaps.
- `possible_gap` requires one or more `possible` gaps and no `critical` gaps.
