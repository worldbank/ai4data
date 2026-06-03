"""
Layout detection adapters for the data snapshot annotation project.

Each adapter module converts raw model outputs into the Unified Evaluation
Schema v1.3 format (``data-snapshot-eval-v1.3.schema.json``).

Adapter responsibilities
------------------------
- Convert coordinates into normalized xyxy ``[0, 1]`` format.
- Map model-specific labels to the canonical ``LABEL_MAP``.
- Generate consistent ``doc_id`` and ``page_id`` identifiers.
- Populate prediction metadata (``run_id``, ``model_name``, etc.).
- Ensure schema compliance prior to evaluation.
"""
