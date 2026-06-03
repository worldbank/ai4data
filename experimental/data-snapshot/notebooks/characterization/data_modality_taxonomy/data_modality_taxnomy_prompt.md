I am building a taxonomy of "data modalities" for document snapshots extracted from PDFs (tables, figures, charts, maps, etc.). My goal is to produce a stable, evaluation-ready classification scheme tailored to the dataset provided.

Your task is to analyze a list of short snapshot descriptions and:

1. Identify the common semantic data categories (“data modalities”)
2. Propose a clean, compact taxonomy
3. Define each modality clearly
4. Group related modalities under roll-up categories
5. Identify potential ambiguities or overlaps
6. Ensure it is stable, minimal, and extensible

Design Constraints:
- `data_modality` should capture what kind of information the snapshot conveys, not how it is rendered.
- The taxonomy should:
  - Have low cardinality (approximately 6–10 modalities)
  - Avoid overlap and ambiguous boundaries
  - Support human annotation and automated classification
- Roll-up groups are allowed for analytical convenience but should not be used as labels.

The taxonomy should:
- Be stable across heterogeneous document corpora
- Support multi-label classification if needed
- Be useful for downstream evaluation and error analysis
- Prefer compact, interpretable top-level categories
- Avoid overly granular modality definitions
- Keep visual representation separate from semantic content

Methodology:
Follow these steps explicitly:
1. Scan the dataset:
   - Identify recurring semantic patterns (e.g. finance, indicators, maps, governance, population).
2. Propose candidate modalities:
   - Each modality should represent a distinct analytical intent.
3. Resolve overlaps:
   - Merge or split categories only when it improves stability and clarity.
4. Name the modalities:
   - Prefer neutral, domain-appropriate terms.
   - Avoid overloaded ML terms unless clearly justified.
5. Validate against edge cases:
   - Ensure each snapshot can be assigned a dominant modality.

Output Requirements:
Produce a single table with the following columns:
rollup_group | data_modality | definition | typical_content | examples

Where:
- definition is precise and non-overlapping
- typical_content lists common elements
- examples are concrete snapshot-level examples

Finally:
- Identify edge cases or ambiguous examples
- Recommend refinements to the taxonomy if necessary

Tone & Quality Bar:
- Be explicit and structured
- Prefer clarity over cleverness
- Optimize for long-term stability, not short-term completeness
- Assume the taxonomy will be used for model evaluation