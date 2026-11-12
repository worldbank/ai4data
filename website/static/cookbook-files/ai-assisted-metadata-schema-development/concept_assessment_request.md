# Concept assessment request: teaching adaptation

This independently written, unexecuted specification illustrates classification
after human concept refinement. It is not the complete historical prompt or a
ready-to-send provider payload. No live model call is needed for the exercises.

## Reference and input

Supply the fixed reviewed concept inventory. Each entry contains its canonical
name, definition, and illustrative discovered field names. Include the examples
from concepts merged into a retained concept without making the merged names
additional classification targets.

Supply one exact-name field profile: metadata_field, top_description_values,
and top_observed_values. Keep its identity and the inventory version beside the
request. The historical classifier received these semantic profile attributes;
frequency analysis was a separate use of the resulting assignments.

## Classification instruction

Choose exactly one allowed canonical name. Compare the profile's intended
meaning with the definitions and examples. A similar name is insufficient.
Examples illustrate meanings and do not exhaust the permitted vocabulary of
discovered field names. Do not create concepts, propose merges, or modify the
reference inventory during classification.

Use the historical residual label not_in_ontology when no retained concept
adequately describes the profile. Intentionally excluded analytical values and
extraction-oriented contents may belong here. This label records a residual
classification; it does not assert that the reference is a formal ontology.

Return canonical_name and confidence. Constrain canonical_name to the reviewed
names plus not_in_ontology. Confidence is high, medium, or low. High means a clear
fit; medium means several concepts appear plausible; low means no fit is clear,
including a residual assignment. Confidence remains a model judgment.

## Inspection

Check the returned name against the allowed set, then inspect the semantic fit.
Retain the field identity, response, model and prompt settings, and execution
status. Flag ambiguous or surprising assignments for human review. Record a
human correction separately from the original model result.

## Historical boundary

The study used GPT-5.4 mini with medium reasoning through the Responses API.
The classification target was 41 retained concepts plus not_in_ontology.
A Pydantic enumeration constrained the output names; confidence also had a
fixed three-value vocabulary. Maximum output tokens and temperature used the
default API settings. These are historical settings, not a current model choice.

The original assessment mapping and later operational revisions are different
artifacts. A fresh replication must preserve that distinction and save new
outputs separately. Different models and prompts may return different labels.

Source: Dy & Solatorio (2026). Artifact-Driven Metadata Schema Development:
A Human-in-the-Loop Methodology with Multimodal LLMs. Working paper. Appendix D.
