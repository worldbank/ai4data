"""Dataset mention extraction schema."""

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

    def build(self, extractor) -> Any:
        """Build the GLiNER2 schema.

        Args:
            extractor: GLiNER2 extractor instance

        Returns:
            Configured schema object
        """
        schema = (
            extractor.create_schema()
            .structure("dataset_mention")
            # Core dataset identity
            .field(
                "dataset_name",
                dtype="str",
                threshold=self._field_thresholds.get("dataset_name", self.threshold),
                # description=(
                #     "The extracted name of the dataset as mentioned in the text. "
                #     "May be a formal title (e.g., 'Demographic and Health Survey') or an informal reference "
                #     "(e.g., 'household survey data'), depending on the tagging."
                # ),
            )
            .field(
                "dataset_tag",
                dtype="str",
                choices=["named", "descriptive", "vague", "non-dataset"],
                # description=(
                #     "Classification of the dataset mention: "
                #     "'named' for formal dataset titles, "
                #     "'descriptive' for unnamed but clearly defined datasets, "
                #     "'vague' for ambiguous references to data sources, "
                #     "'non-dataset' when the term does not function as a dataset in context or empty."
                # ),
            )
            .field(
                "description",
                dtype="str",
                # description=(
                #     "A short description of the type of data contained in the dataset, "
                #     "such as 'household data', 'crime reports', 'satellite imagery', "
                #     "'employment indicators', or 'administrative microdata'. This describes "
                #     "the data content, not the dataset category."
                # ),
            )
            .field(
                "data_type",
                dtype="str",
                description=(
                    "The type or category of the dataset. e.g survey, report, system, etc."
                ),
            )
            # Metadata related to provenance
            .field(
                "acronym",
                dtype="str",
                threshold=self._field_thresholds.get("acronym", self.threshold),
                # description=(
                #     "The acronym associated with the dataset, if explicitly mentioned "
                #     "(e.g., 'HFS' for 'High-Frequency Survey')."
                # ),
            )
            .field(
                "author",
                dtype="str",
                # description=("The individual(s) or authors responsible for creating the dataset."),
            )
            .field(
                "producer",
                dtype="str",
                # description=(
                #     "The institution or organization that produced, collected, or published the dataset, "
                #     "such as a national statistics office, ministry, research institution, or international agency."
                # ),
            )
            .field(
                "geography",
                dtype="str",
                # description=("The geographical coverage of the dataset, "),
            )
            .field(
                "publication_year",
                dtype="str",
                # description=(
                #     "The year the dataset was released or published. "
                #     "This is distinct from the reference year, which refers to when the data were collected."
                # ),
            )
            .field(
                "reference_year",
                dtype="str",
                # description=(
                #     "The year or time period the data refer to. "
                #     "This is the year of data collection (e.g., 2018 survey year, 2020 census year), "
                #     "which is separate from the publication or release year."
                # ),
            )
            .field(
                "reference_population",
                dtype="str",
                # description=(
                #     "The target population covered by the dataset (e.g., 'households', "
                #     "'migrant workers', 'urban residents', 'women aged 15–49')."
                # ),
            )
            # Usage classification
            .field(
                "is_used",
                dtype="str",
                choices=["True", "False"],
                # choices=["used", "not_used"],
                # description=(
                #     "'True' if the dataset is used in the empirical analysis; "
                #     "'False' if not."
                # ),
            )
            # Context of the mention
            .field(
                "usage_context",
                dtype="str",
                choices=["primary", "background", "supporting"],
                # description=(
                #     "Describes how the dataset is used: "
                #     "'primary' if it is a main analytical dataset, "
                #     "'background' if cited for contextual or literature support, "
                #     "'supporting' if used as secondary or robustness-check data."
                # ),
            )
        )

        return schema
