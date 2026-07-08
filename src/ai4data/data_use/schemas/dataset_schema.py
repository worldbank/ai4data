"""Dataset mention extraction schema.

Updated for v21-diversity model:
- `dataset_name`    -> `mention_name`
- `dataset_tag`     -> `specificity_tag`  (choices: named, descriptive, vague, na)
- `data_type`       -> `typology_tag`     (choices: survey, census, ...)
- `has_data`        removed (subsumed by `is_used`)
"""

from typing import Any, Dict


class DatasetSchema:
    """Schema builder for dataset mention extraction."""

    DEFAULT_THRESHOLD = 0.5

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        """Initialize schema with default threshold.

        Args:
            threshold: Default confidence threshold for extraction
        """
        self.threshold = threshold
        self._field_thresholds: Dict[str, float] = {}

    def set_threshold(self, field_name: str, threshold: float) -> "DatasetSchema":
        """Set custom threshold for a specific field.

        Args:
            field_name: Name of the field
            threshold: Confidence threshold (0.0 to 1.0)

        Returns:
            Self for method chaining
        """
        self._field_thresholds[field_name] = threshold
        return self

    def build(self, extractor, extract_provenance: bool = False) -> Any:
        """Build the GLiNER2 schema.

        Args:
            extractor: GLiNER2 extractor instance
            extract_provenance: If True, include provenance fields (author,
                producer, publication_year, reference_year, reference_population,
                geography, description, acronym) in the schema. These fields are
                informative but increase inference latency. Defaults to False.

        Returns:
            Configured schema object
        """
        t = self._field_thresholds  # shorthand
        default = self.threshold

        schema = (
            extractor.create_schema().structure("data_mention")
            # ── Core identity ──────────────────────────────────────────────────
            .field("mention_name", dtype="str", threshold=t.get("mention_name", default))
        )

        if extract_provenance:
            # Optional provenance fields — slower inference, richer output
            (
                schema.field("acronym", dtype="str", threshold=t.get("acronym", default))
                .field("producer", dtype="str")
                .field("reference_year", dtype="str")
                .field("geography", dtype="str")
            )

        # ── Classification fields (always extracted) ───────────────────────────
        # These are forced-choice fields (constrained by `choices`). The model
        # must pick one of the listed values, so thresholding adds no value and
        # only causes silent data loss. Use threshold=0 to never suppress them.
        # Per-field overrides via set_threshold() are still respected.
        # NOTE: "na" is excluded from choices at inference time to force the
        # model to commit to a concrete label.
        (
            schema.field(
                "specificity_tag",
                dtype="str",
                choices=["named", "descriptive", "vague"],
                threshold=t.get("specificity_tag", 0),
            )
            .field(
                "typology_tag",
                dtype="str",
                choices=[
                    "survey",
                    "census",
                    "database",
                    "administrative",
                    "indicator",
                    "geospatial",
                    "microdata",
                    "report",
                    "other",
                ],
                threshold=t.get("typology_tag", 0),
            )
            .field(
                "usage_context",
                dtype="str",
                choices=["primary", "supporting", "background"],
                threshold=t.get("usage_context", 0),
            )
            .field(
                "is_used",
                dtype="str",
                choices=["True", "False"],
                threshold=t.get("is_used", 0),
            )
        )

        return schema
