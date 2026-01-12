from gliner2 import GLiNER2

# Load model once, use everywhere
MODEL_ID = "fastino/gliner2-large-v1"
extractor = GLiNER2.from_pretrained(MODEL_ID)  # can be switched into a finetuned one


schema = (
    extractor.create_schema()
    .structure("dataset_mention")
    # ──────────────────────────────
    # Core dataset identity
    # ──────────────────────────────
    .field(
        "dataset_name",
        dtype="str",
        threshold=0.85,
        description=(
            "The extracted name of the dataset as mentioned in the text. "
            "May be a formal title (e.g., 'Demographic and Health Survey') or an informal reference "
            "(e.g., 'household survey data'), depending on the tagging."
        ),
    )
    .field(
        "tag",
        dtype="str",
        choices=["named", "descriptive", "vague", "non-dataset"],
        description=(
            "Classification of the dataset mention: "
            "'named' for formal dataset titles, "
            "'descriptive' for unnamed but clearly defined datasets, "
            "'vague' for ambiguous references to data sources, "
            "'non-dataset' when the term does not function as a dataset in context."
        ),
    )
    # ──────────────────────────────
    # Data type vs. Data description
    # ──────────────────────────────
    .field(
        "data_type",
        dtype="str",
        description=(
            "The high-level category of the dataset based on its nature: "
            "'survey' for structured questionnaires, "
            "'report' for compiled statistical summaries, "
            "'program' for operational or monitoring systems, "
            "'census' for population-wide enumerations, "
            "'system' for administrative or information systems."
        ),
    )
    .field(
        "description",
        dtype="str",
        description=(
            "A short description of the type of data contained in the dataset, "
            "such as 'household data', 'crime reports', 'satellite imagery', "
            "'employment indicators', or 'administrative microdata'. This describes "
            "the data content, not the dataset category."
        ),
    )
    # ──────────────────────────────
    # Metadata related to provenance
    # ──────────────────────────────
    .field(
        "acronym",
        dtype="str",
        description=(
            "The acronym associated with the dataset, if explicitly mentioned "
            "(e.g., 'HFS' for 'High-Frequency Survey')."
        ),
    )
    .field(
        "producer",
        dtype="str",
        description=(
            "The institution or organization that produced, collected, or published the dataset, "
            "such as a national statistics office, ministry, research institution, or international agency."
        ),
    )
    .field(
        "publication_year",
        dtype="str",
        description=(
            "The year the dataset was released or published. "
            "This is distinct from the reference year, which refers to when the data were collected."
        ),
    )
    .field(
        "reference_year",
        dtype="str",
        description=(
            "The year or time period the data refer to. "
            "This is the year of data collection (e.g., 2018 survey year, 2020 census year), "
            "which is separate from the publication or release year."
        ),
    )
    .field(
        "reference_population",
        dtype="str",
        description=(
            "The target population covered by the dataset (e.g., 'households', "
            "'migrant workers', 'urban residents', 'women aged 15–49')."
        ),
    )
    # ──────────────────────────────
    # Usage classification
    # ──────────────────────────────
    .field(
        "is_used",
        dtype="str",
        choices=["true", "false"],
        description=(
            "'true' if the dataset is used in the empirical analysis; "
            "'false' if only mentioned for context, comparison, or narrative framing."
        ),
    )
    # ──────────────────────────────
    # Context of the mention
    # ──────────────────────────────
    .field(
        "mention_context",
        dtype="str",
        choices=["primary", "background", "supporting"],
        description=(
            "Describes how the dataset is used: "
            "'primary' if it is a main analytical dataset, "
            "'background' if cited for contextual or literature support, "
            "'supporting' if used as secondary or robustness-check data."
        ),
    )
)

text1 = """Our analysis uses the 2022 Demographic and Health Survey (DHS) conducted by the National Statistics Office collected for years 2010-2019 consists of demographic and employment indicators. The DHS provides nationally representative data for women aged 15–49, especially on health and fertility indicators. We complement the DHS with descriptive statistics from administrative systems, but only the DHS is used in the empirical models."""

results = extractor.extract(text1, schema, include_confidence=True)
