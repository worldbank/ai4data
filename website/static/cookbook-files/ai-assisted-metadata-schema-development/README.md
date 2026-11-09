# Research evidence for AI-Assisted Metadata Schema Development

This downloadable package supports all seven chapters of the working-draft
*Practical Guide to AI-Assisted Metadata Schema Development*. It combines
selected research evidence with newly prepared offline exercises, request
examples, and human-review worksheets. Readers can inspect the methodology
without executing the research pipeline or making a model request.

Recorded observations, decisions, and specification excerpts retain their
provenance. The scripts, request adaptations, and blank worksheets are teaching
materials; they are not historical research runners or original review logs.

## Research attribution and public sources

Dy & Solatorio (2026). *Artifact-Driven Metadata Schema Development: A
Human-in-the-Loop Methodology with Multimodal LLMs*. Working paper.

The manuscript names AJ Carl P. Dy and Aivin V. Solatorio as its authors.
Its arXiv publication is pending. No publication URL or identifier is assigned
in this package. Section and table references refer to the supplied working
paper; later manuscript revisions may renumber them.

The source register in [evidence.json](./evidence.json) identifies this paper
and the public [Data Snapshot dataset](https://huggingface.co/datasets/ai4data/data-snapshot).
Image and document-metadata links pin dataset revision
`626f3c54ba6d8f6e0ba0795f9e544e24b2650faf`.
The [source-image register](./source_images.csv) gives each selected artifact's
original institutional document, attribution, checksums, and reuse status.
The eight artifacts were verified against the actual research sample manifests.
They are a selected teaching collection, not a new representative research sample.

PROV-O provides the authoritative distinction used by the provenance example:
[derivation from another entity](https://www.w3.org/TR/prov-o/#wasDerivedFrom)
and [attribution to an agent](https://www.w3.org/TR/prov-o/#wasAttributedTo).
These references explain the semantic distinction; the exercise requires no
ontology construction or RDF implementation.

## Files

| File | Purpose | Chapters |
|---|---|---|
| [README.md](./README.md) | Package navigation, evidence labels, provenance, and use limits. | All |
| [evidence.json](./evidence.json) | Source register, study counts, preparation-time analyses, and ten selected examples. | 1–7 |
| [source_images.csv](./source_images.csv) | Eight external image and parent-document references, attribution, accessibility checks, and reuse limits. | 1–7 |
| [chapter_examples.csv](./chapter_examples.csv) | Chapter-to-example map and the evidence selections reused at each stage. | 1–7 |
| [RUN_EXERCISES.md](./RUN_EXERCISES.md) | Required downloads, commands, expected outputs, and safe output handling for the offline exercises. | 1–3; reused in 6 |
| [artifact_characterization.md](./artifact_characterization.md) | Filled teaching statement of artifact scope and descriptive boundaries. | 1 |
| [check_manifest.py](./check_manifest.py) | Read-only identity and split-boundary check for the selected register. | 1, 6 |
| [discovery_excerpt.jsonl](./discovery_excerpt.jsonl) | Selected recorded title observations in a newly prepared response wrapper. | 2 |
| [inspect_discovery.py](./inspect_discovery.py) | Read-only structural inspection of the selected responses and source identities. | 2 |
| [discovery_request.md](./discovery_request.md) | Unexecuted teaching adaptation of an independent discovery request. | 2 |
| [geographic_observations.json](./geographic_observations.json) | Three unchanged observations used by the exact-name profiling exercise. | 3 |
| [profile_observations.py](./profile_observations.py) | Standard-library aggregation into a new JSON profile; existing outputs are never replaced. | 3 |
| [concept_synthesis_request.md](./concept_synthesis_request.md) | Unexecuted request example for proposing shared concepts from field profiles. | 3 |
| [concept_review_template.csv](./concept_review_template.csv) | Blank worksheet for proposed concept actions and human dispositions. | 4 |
| [concept_assessment_request.md](./concept_assessment_request.md) | Unexecuted request example for assignments to a fixed reviewed inventory. | 4 |
| [initial_field_specifications.md](./initial_field_specifications.md) | Five initial v1.1 field definitions, guidance, and specification examples. | 5 |
| [field_specification_template.csv](./field_specification_template.csv) | Blank worksheet for field definitions, evidence rules, and review issues. | 5 |
| [validation_request_specs.md](./validation_request_specs.md) | Unexecuted teaching specifications for the two complementary validation instruments. | 6 |
| [gap_review_template.csv](./gap_review_template.csv) | Blank worksheet for candidate evidence, materiality, and human adjudication. | 6 |
| [standards_review_template.csv](./standards_review_template.csv) | Blank worksheet connecting standards comparisons to proposed changes and human decisions. | 7 |
| [final_field_inventory.csv](./final_field_inventory.csv) | Baseline-to-final field names and final cardinalities for inventory reconciliation; not a complete specification. | 7 |

## How to use the package

1. Read the evidence labels below and use chapter_examples.csv to locate each chapter's selected examples in evidence.json. Keep recorded study results separate from teaching applications and preparation-time analyses.
2. For the offline Python exercises, follow RUN_EXERCISES.md. Place the six named script and input files together in a new working directory and use Python 3.10 or newer. The scripts use only the standard library; no images, dataset download, credentials, or model calls are required.
3. For manual review, inspect the chapter's selected evidence and save working copies of the relevant blank worksheets. Record artifact or example identities, schema versions, evidence, and actual reviewer decisions. Keep the supplied evidence and blank templates unchanged.
4. Read the request examples as unexecuted teaching adaptations. They describe inputs and output checks, not ready-to-run provider integrations. A fresh model-supported application needs its own complete inputs, execution settings, budget, and separate output locations.

The final inventory supports reconciliation and selected-change review. Applying
the complete methodology to another collection requires that collection's full
operational specification and decision record. The package's selected field
excerpts do not replace complete historical schema inputs.

## Evidence labels and interpretation

The `study_summary` object records the study results. The
`verification_derived_facts` object contains deterministic comparisons computed
during preparation, including sample overlap and mapping differences.
Each example names its intended chapters, objective, methodology stage,
relevant schema versions, paper locators, snapshot identifiers where applicable,
inputs, outputs, and limitations.

| Label | Meaning |
|---|---|
| `recorded_study_result` | The working paper's result, checked against retained evidence where programmatically verifiable. |
| `excerpt` | Selected recorded rows, definitions, or findings. Fields explicitly named transcription or paraphrase identify teaching representations of human decisions. |
| `reproducible_exercise` | A deterministic transformation of selected real observations, computed during preparation. Its small-subset output is separate from the recorded full-study output. |
| `deterministic_analysis_during_evidence_preparation` | A new count or comparison of retained records, rather than a claim that the study originally reported it. |

Raw observation strings, including uncertainty and inferred values, are retained.
Verifying an excerpt against its source does not certify that the model's
interpretation was correct. Some observations deliberately preserve broad or
subsequently excluded candidate fields.
CSV-derived excerpts retain their original string values. In a field profile,
columns such as `top_observed_values` contain serialized lists as strings.
Their representation is preserved; these strings must not be executed as code.

Human decisions labelled `transcription` are structured restatements of a
recorded decision. Those labelled `paraphrase` summarize its meaning. Neither
label claims that the structure is the original historical review worksheet.
Blank review templates and request adaptations are newly prepared teaching aids.
The request examples are unexecuted; no synthetic historical output is included.

## Continuous teaching examples

The three-artifact discovery thread uses the Jordan food-expenditure chart,
the Afghanistan income table, and the Russian fiscal chart. Their independent
observations show naming variation and support an exact-name aggregation
exercise. The title-profile example then connects those observations to
candidate synthesis and original concept assignments.

The Afghanistan table returns in the provenance example. Its source phrase
separates the survey on which the data are based from credit for calculation.
The soybean figure connects original residual classification to the operational
restoration of panel headings. The South Sudan composite appears in both
validation exercises, allowing their different questions to be compared on the
same artifact. The Diffa table supplies a second representation-limit example.
The Pakistan financing table and Lebanon health chart provide targeted
low-support and residual examples.

The [chapter map](./chapter_examples.csv) assigns these examples to the agreed
seven chapters. Reuse indicates a teaching connection; it does not claim that
an individual image caused an otherwise undocumented historical decision.

## Methodology and schema boundaries

Independent metadata discovery preserves artifact-level observations before a
shared vocabulary is fixed. Exact-name profiling aggregates them without
semantic canonicalization. Concept synthesis proposes shared meanings. Human
refinement decides conceptual boundaries; coverage assessment characterizes the
reviewed inventory. Operational schema design determines usable fields and
annotation guidance. Held-out validation produces evidence for human
adjudication. Standards-informed refinement follows that validation.

| Boundary | Recorded result or treatment |
|---|---|
| Development | The study used 210 snapshots from three corpora, with 35 figures and 35 tables per corpus. |
| Discovery and profiling | The study produced 3,041 observations and 833 exact-name profiles. |
| Concept inventories | Synthesis produced 70 candidates. Human refinement retained 41 concepts. The retained decision table contains 41 keep rows and eight merge rows; it does not provide a complete 70-row decision ledger. |
| Original concept mapping | Chapter 4 uses the original assignments to the 41-concept inventory plus the residual category `not_in_ontology`. |
| Operational schema | Version v1.1 contains 36 fields. Operational design restored `panel_title` as an independent field for multi-panel artifacts. |
| Documentation revision | Version v1.1.1 retains 36 fields and clarifies guidance following validation. |
| Standards-informed schema | Version v1.1.2 contains 37 fields after splitting `data_source` into `data_sources` and `attributions`. |

The later mapping differs on five profiles: `unit_of_analysis`, `panel_titles`,
`event_context`, `operational_context`, and `contextual_event`. Both assignments
are preserved in the preparation-time comparison. In particular, the later
`panel_title` label must not be retroactively introduced into the original
41-concept assessment. No exact edit dates are asserted. The design-decision
excerpts use terminology from their respective stages and do not constitute
an exhaustive concept-to-field lineage table.

Frequencies require explicit units and denominators. For example, the exact
`geographic_scope` profile contains 192 observations, while the canonical
concept receives support from 194 distinct development snapshots. The
three-observation exercise has its own count and denominator. The original
mapping supports `financing_instrument` in seven of 210 snapshots and
`data_collection_method` in nine. Low support alone does not decide whether
these concepts should remain.

## Validation and implementation limits

Both validation exercises used the same 202 snapshots. Validation 1 generated
2,876 observations and 578 candidate proposals. Validation 2 assessed 197
snapshots as having no critical gap and five as having possible gaps; those
five produced six gap records. Human adjudication accepted no new field.
Documentation clarifications and representation limitations remained meaningful
outcomes.

The development and held-out collections are snapshot-disjoint. Comparing their
retained manifests identifies 132 development parent documents, 125 held-out
parent documents, and 65 shared parents, using corpus and document identifier
together. The exercise does not establish document-disjoint generalization.

Twelve held-out snapshots were reused during validation calibration, leaving
190 untouched by prompt calibration. Validation 2 used a history-neutral
model-facing schema to reduce anchoring on earlier results. The title-ablation
control did not demonstrate its intended sensitivity; the source-and-note
ablation did. The 197 no-gap assessments did not receive an independent human
false-negative audit. No candidate-recall estimate follows from these results.

The final 37-field specification followed the held-out exercises and did not
receive a further independent held-out validation in the reported study.
The standards review was an interactive human–AI process, not one fixed-prompt
pipeline. Standards mappings supported human decisions rather than automatic
substitutions. The later machine-readable comparison in manuscript Appendix I
is post-study work; it does not establish a causal effect of Markdown versus
JSON Schema. Manuscript Appendix H contains sample populated records.

Sampling adequacy depends on artifact diversity and intended use. The study's
sample size is not a universal minimum. Its evidence does not establish
universal coverage, superiority to expert-only development, interoperability,
metadata-extraction accuracy, or generality to other artifact classes.

## Image provenance and reuse

All eight pinned images and their metadata were retrieved and visually checked
on 6 November 2026. The image register records dimensions and SHA-256 hashes.
Four original PDF URLs responded to HEAD requests with HTTP 200 and a PDF
content type. Four World Bank source URLs returned HTTP 403 to automated HEAD
checks. HTTP source URLs were checked over HTTPS. These outcomes describe
access from the checking environment, not permanent document availability.
No original PDF was downloaded.

The pinned dataset card labels the dataset MIT and its licence includes the
World Bank IGO rider. That observation does not establish republication rights
for every institutional image. World Bank metadata points to Access to
Information guidance; other selected records have empty usage-terms fields.
An empty field, disclosed status, or public URL is not a redistribution grant.
All image rows therefore remain `external_link_only`.

No image binaries, parent-document PDFs, full metadata records, manuscript,
source code, or complete research outputs are redistributed here. Small
research excerpts are attributed to Dy & Solatorio. Institutional titles and
attribution facts identify the linked sources; they do not relicense them.
The Diffa metadata contains an abbreviated contributor entry, “5 more”; it
must not be presented as a contributor name or a complete attribution list.
Its French text requires language review if translated for a chapter.

## Review-template fields

The concept worksheet records the concept, evidence reference, proposed action,
target concept where relevant, semantic and scope rationales, reviewer, and
final decision with its reason. Suggested actions are retain, merge, rename,
split, or exclude. These are teaching prompts rather than historical records.

The gap worksheet records the artifact and schema version, candidate identity,
evidence, closest fields, issue type, material impact, disposition, rationale,
revision or deferral, and reviewer. Useful issue types are missing field,
documentation, representation, already covered, out of scope, and uncertain.
The reviewer decides the disposition. Neither frequency nor a model label
constitutes acceptance.

The field-specification worksheet separates definitions, purpose and scope,
inclusion and exclusion guidance, example values, evidence references,
missing-value handling, related fields, and review dispositions. Preserve source
definitions and label newly proposed guidance separately.

The standards-review worksheet separates the local meaning and authoritative
term comparison from a candidate refinement and human disposition. Record source
versions or dates, scope differences, reasons, the reviewer, and the output
reference so an accepted change can be traced.

All four review-template CSVs contain headers only. Adding rows creates new
review work; it does not extend the historical study's adjudication evidence.

## Preparation checks and current use

Evidence preparation checked sample membership, research-row fidelity, inventory
counts, original/later mapping differences, public image identities and hashes,
and source metadata. No paid API or research experiment was run during that
preparation. The retained evidence can be inspected without executing code.
The dated image-access and visual checks above remain preparation-time records.

The cookbook now supplies three executable teaching exercises, request
adaptations, and manual review materials. The standalone scripts are covered by
focused tests in the AI4Data repository. Their commands, expected outputs,
exit statuses, and output-preservation rules are documented in
[RUN_EXERCISES.md](./RUN_EXERCISES.md).

Use new working files for exercise outputs and worksheet entries. Any later
live replication must keep new outputs separate and identify them as replication
results. The supplied exercises inspect selected evidence; they do not reproduce
the complete study or independently validate the final schema.
