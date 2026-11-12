# Independent discovery request: teaching adaptation

This independently written prompt illustrates the methodology. It is not the
complete historical prompt and has not been executed to produce new results.
The study supplied one snapshot image and available associated document
metadata per request. The examples below describe logical request contents;
they are not a provider-specific API payload.

For another artifact class, first select a representation and model that can
inspect its relevant evidence, and define allowable context and evidence-origin
labels. The image inputs and snapshot/document enums below remain specific to
this teaching adaptation; changing them requires a separately documented request.

## System instruction

Act as a metadata designer. From one analytical artifact and its supplied
document context, propose reusable descriptive properties that support
identification, interpretation, filtering, cataloging, and discovery.
Do not populate a predefined operational metadata schema. Distinguish reusable
field names from observed values and from numerical data contained in the
artifact. Prefer generic meanings. Keep potentially useful distinctions for
later review; no field proposed here is automatically accepted.

Ground every observed value in the image or supplied context. Identify the
source level as snapshot, document, or both. Preserve uncertainty; when no
value is identifiable, state that explicitly. Label inferred values as inferred.
Do not infer a specific value from vague contextual evidence. Do not add
administrative extraction fields as descriptive candidates.

Return an object with a fields array. Each item has nonempty string values for
metadata_field, observed_value, description, source_level, discovery_value,
and reasoning. Use snake_case field names. source_level is snapshot, document,
or both; discovery_value is high, medium, or low. The reasoning explains the
descriptive value and is evidence for later human review.

## User content

Supply the selected snapshot image and its exact snapshot identifier.
Supply a separately identified document-context object and its source record.
If context is unavailable, record that fact. Analyze this artifact independently
without a vocabulary or outputs accumulated from earlier requests.

## Historical boundary

The study asked for the 10–15 most useful fields; this was a prompt preference,
not a response-count guarantee. Discovery used GPT-5.5, medium reasoning, and
a 5,000-token output ceiling. These are historical settings, not current model
recommendations. The downloadable excerpts preserve selected results, not the
complete input messages. A document title shown in the source register must
not be represented as the complete historical document-context payload.

Source: Dy & Solatorio (2026). Artifact-Driven Metadata Schema Development:
A Human-in-the-Loop Methodology with Multimodal LLMs. Working paper. Appendix B.
