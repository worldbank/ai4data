# Concept synthesis request: teaching adaptation

This independently written prompt explains a semantic task. It is not the
complete historical prompt and has not been executed as a new experiment.

## Input and instruction

Supply the exact-name field profiles, including their names, descriptions,
representative values, rationales, and occurrence statistics. State the artifact
class, intended descriptive use, diversity, and profile count. Compare the
meaning expressed by the accumulated evidence, not only the field names.

Propose reusable canonical metadata concepts. Merge clear synonyms while
retaining plausible distinctions when evidence is ambiguous. Keep low-frequency
but meaningful properties available for expert review. Distinguish descriptive
requirements from values and extraction-oriented contents. Do not design final
operational fields, relationships, or a formal ontology at this stage.

Return a table with canonical_name, definition, rationale, and sample_fields.
Each row is a proposal whose definition establishes the shared meaning.
sample_fields lists representative contributing names; it is not an exhaustive
profile-to-concept assignment table. Explain uncertain relationships separately
for human review if using this adaptation.

## Historical boundary

The study supplied all 833 profiles as a CSV in one GPT-5.5 request, using high
reasoning and a 12,000-token output ceiling. The prompt suggested approximately
50–70 concepts; the retained output contains 70 candidate rows. It returned a
Markdown table, not an enum-constrained classification or a final schema.
The three title profiles shown in the cookbook are only an excerpt of that
input. New synthesis from those three profiles cannot reproduce the complete
70-concept result. Checking input capacity is necessary before scaling an
adaptation; splitting profiles across requests would be a different workflow.

Source: Dy & Solatorio (2026). Artifact-Driven Metadata Schema Development:
A Human-in-the-Loop Methodology with Multimodal LLMs. Working paper. Appendix C.
