from gliner2 import GLiNER2
from huggingface_hub import snapshot_download

# Load the base model, then apply the fine-tuned LoRA adapter
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
adapter_path = snapshot_download("rafmacalaba/gliner2-datause-v1")
model.load_adapter(adapter_path)

schema = (
    model.create_schema()
    .structure("dataset_mention")
    # ── Core identity ──────────────────────────────────────────────────
    .field(
        "dataset_name",
        dtype="str",
        threshold=0.85,
        description=(
            "The extracted name of the dataset as mentioned in the text. "
            "May be a formal title (e.g., 'Demographic and Health Survey') or an informal "
            "reference (e.g., 'household survey data'), depending on the tag."
        ),
    )
    .field(
        "acronym",
        dtype="str",
        threshold=0.85,
        description=(
            "The acronym associated with the dataset, if explicitly mentioned "
            "(e.g., 'DHS' for 'Demographic and Health Survey')."
        ),
    )
    # ── Provenance ─────────────────────────────────────────────────────
    .field(
        "author",
        dtype="str",
        description=(
            "The individual author(s) or principal investigator(s) credited with "
            "creating or leading the dataset, if named in the text "
            "(e.g., 'Smith et al.', 'Jones and Chen'). Distinct from the producing institution."
        ),
    )
    .field(
        "producer",
        dtype="str",
        description=(
            "The institution or organization that produced, collected, or published the dataset, "
            "such as a national statistics office, ministry, research institution, "
            "or international agency."
        ),
    )
    .field(
        "publication_year",
        dtype="str",
        description=(
            "The year the dataset was released or published. "
            "Distinct from the reference year, which refers to when the data were collected."
        ),
    )
    .field(
        "reference_year",
        dtype="str",
        description=(
            "The year or time period the data refer to — i.e., the year of data collection "
            "(e.g., '2018 survey year', '2020 census year'), separate from the publication year."
        ),
    )
    .field(
        "reference_population",
        dtype="str",
        description=(
            "The target population covered by the dataset "
            "(e.g., 'households', 'migrant workers', 'urban residents', 'women aged 15–49')."
        ),
    )
    .field(
        "geography",
        dtype="str",
        description=(
            "The geographic scope or coverage of the dataset, such as a country, region, "
            "city, or subnational area (e.g., 'Ghana', 'Sub-Saharan Africa', 'rural districts')."
        ),
    )
    .field(
        "description",
        dtype="str",
        description=(
            "A short description of the data content, such as 'household survey data', "
            "'crime reports', 'satellite imagery', 'employment indicators', or "
            "'administrative microdata'. Describes what the data contain, not its category."
        ),
    )
    # ── Classification fields ──────────────────────────────────────────
    .field(
        "dataset_tag",
        dtype="str",
        choices=["named", "descriptive", "vague"],
        description=(
            "Classification of the dataset mention: "
            "'named' for formal dataset titles with a proper name, "
            "'descriptive' for unnamed but clearly defined datasets, "
            "'vague' for ambiguous or generic references to data sources."
        ),
    )
    .field(
        "usage_context",
        dtype="str",
        choices=["primary", "supporting", "background"],
        description=(
            "How the dataset is used in the paper: "
            "'primary' if it is the main analytical dataset, "
            "'supporting' if used as secondary or robustness-check data, "
            "'background' if only cited for context or literature framing."
        ),
    )
    .field(
        "is_used",
        dtype="str",
        choices=["True", "False"],
        description=(
            "'True' if the dataset is actually used in the empirical analysis; "
            "'False' if only mentioned for context, comparison, or narrative framing."
        ),
    )
)

text1 = """We use two complementary geocoded household data sets to analyze outcomes in Ghana:
the Demographic and Health Survey (DHS) in 2020 and the Ghana Living Standard Survey (GLSS)
in 2012, which provide information on a wide range of welfare outcomes. The paper contributes
to the growing literature on the local effects of mining."""

results = model.extract(text1, schema, include_confidence=True, include_spans=True)
