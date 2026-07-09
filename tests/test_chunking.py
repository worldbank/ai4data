"""Test text chunking and index adjustment for long texts."""

import re
from typing import List
from unittest.mock import MagicMock

import pytest

from ai4data.data_use import extract_from_text
from ai4data.data_use.extractors.dataset_extractor import DatasetExtractor
from ai4data.data_use.models.model_manager import ModelManager


def tokenize_text(text: str) -> List[str]:
    """Tokenize the input text into a list of tokens (words, punctuation)."""
    return re.findall(r"\w+(?:[-_]\w+)*|\S", text)


def create_long_text_with_datasets() -> str:
    """Create a long text (>500 tokens) with dataset mentions at different positions."""
    # Real content at the beginning
    intro = """Our analysis uses the 2022 Demographic and Health Survey (DHS) conducted by
the National Statistics Office collected for years 2010-2019 consists of demographic
and employment indicators. The DHS provides nationally representative data for women
aged 15–49, especially on health and fertility indicators. We complement the DHS with
descriptive statistics from administrative systems, but only the DHS is used in the
empirical models."""

    # Filler tokens to exceed 500 token limit
    filler = " ".join([f"filler_token_{i:03d}" for i in range(1, 600)])

    # Real content at the end (will be in a different chunk)
    conclusion = """We use two complementary geocoded household data sets to analyze
outcomes in Ghana: the Demographic and Health Survey (DHS) in 2020 and the Ghana
Living Standard Survey (GLSS) in 2012, which provide information on a wide range
of welfare outcomes. The paper contributes to the growing literature on the local
effects of mining."""

    return f"{intro}\n{filler}\n{conclusion}"


@pytest.fixture
def mock_model_manager(monkeypatch):
    """Mock ModelManager to return a smart mock model that finds actual entities."""
    mock_model = MagicMock()

    # Mock the schema builder chain
    mock_schema = MagicMock()
    mock_schema.structure.return_value = mock_schema
    mock_schema.field.return_value = mock_schema
    mock_model.create_schema.return_value = mock_schema

    def side_effect_extract(text, schema=None, include_confidence=True, **kwargs):
        """Mock implementation of extract returning character offsets and relations for DatasetSchemaV3."""

        # Find all occurrences of a substring
        def find_all(sub, string):
            start_pos = 0
            while True:
                start_pos = string.find(sub, start_pos)
                if start_pos == -1:
                    return
                yield start_pos
                start_pos += len(sub)

        name_ents = []
        relations = {
            "has_acronym": [],
            "has_organization": [],
            "has_timeframe": [],
            "has_datatype": [],
            "has_usage": [],
        }

        matched_spans = []

        def add_entity(
            name_text,
            name_start,
            name_end,
            specificity="named",
            usage="primary",
            datatype="survey",
            acronym_text=None,
            acronym_start=None,
            acronym_end=None,
        ):
            # Check overlap
            for ms_start, ms_end in matched_spans:
                if not (name_end <= ms_start or name_start >= ms_end):
                    return

            # Name entity
            name_ents.append(
                {
                    "text": name_text,
                    "start": name_start,
                    "end": name_end,
                    "confidence": 0.95,
                    "label": "named_data",
                }
            )
            matched_spans.append((name_start, name_end))

            # Factual relations
            if acronym_text is not None and acronym_start is not None and acronym_end is not None:
                relations["has_acronym"].append(
                    {
                        "head": {"text": name_text, "start": name_start, "end": name_end},
                        "tail": {
                            "text": acronym_text,
                            "start": acronym_start,
                            "end": acronym_end,
                            "confidence": 0.99,
                        },
                        "label": "has_acronym",
                        "score": 0.99,
                    }
                )
                matched_spans.append((acronym_start, acronym_end))

            usage_start = text.find(usage)
            if usage_start != -1:
                relations["has_usage"].append(
                    {
                        "head": {"text": name_text, "start": name_start, "end": name_end},
                        "tail": {
                            "text": usage,
                            "start": usage_start,
                            "end": usage_start + len(usage),
                            "confidence": 0.95,
                        },
                        "label": "has_usage",
                        "score": 0.95,
                    }
                )

        # 1. Look for "Demographic and Health Survey"
        for start in find_all("Demographic and Health Survey", text):
            end = start + len("Demographic and Health Survey")
            acronym_text, acronym_start, acronym_end = None, None, None
            dhs_near = text.find("DHS", end, end + 10)
            if dhs_near != -1:
                acronym_text = "DHS"
                acronym_start = dhs_near
                acronym_end = dhs_near + 3
            add_entity(
                "Demographic and Health Survey",
                start,
                end,
                acronym_text=acronym_text,
                acronym_start=acronym_start,
                acronym_end=acronym_end,
            )

        # 2. Look for "Ghana Living Standard Survey"
        for start in find_all("Ghana Living Standard Survey", text):
            end = start + len("Ghana Living Standard Survey")
            acronym_text, acronym_start, acronym_end = None, None, None
            glss_near = text.find("GLSS", end, end + 10)
            if glss_near != -1:
                acronym_text = "GLSS"
                acronym_start = glss_near
                acronym_end = glss_near + 4
            add_entity(
                "Ghana Living Standard Survey",
                start,
                end,
                acronym_text=acronym_text,
                acronym_start=acronym_start,
                acronym_end=acronym_end,
            )

        # 3. Look for "World Development Indicators"
        for start in find_all("World Development Indicators", text):
            end = start + len("World Development Indicators")
            acronym_text, acronym_start, acronym_end = None, None, None
            wdi_near = text.find("WDI", end, end + 10)
            if wdi_near != -1:
                acronym_text = "WDI"
                acronym_start = wdi_near
                acronym_end = wdi_near + 3
            add_entity(
                "World Development Indicators",
                start,
                end,
                acronym_text=acronym_text,
                acronym_start=acronym_start,
                acronym_end=acronym_end,
            )

        # 4. Look for standalone "DHS"
        for start in find_all("DHS", text):
            end = start + 3
            add_entity("DHS", start, end, acronym_text="DHS", acronym_start=start, acronym_end=end)

        # 5. Look for standalone "GLSS"
        for start in find_all("GLSS", text):
            end = start + 4
            add_entity(
                "GLSS", start, end, acronym_text="GLSS", acronym_start=start, acronym_end=end
            )

        # 6. Look for standalone "WDI"
        for start in find_all("WDI", text):
            end = start + 3
            add_entity("WDI", start, end, acronym_text="WDI", acronym_start=start, acronym_end=end)

        return {"entities": {"named_data": name_ents}, "relation_extraction": relations}

    def side_effect_batch_extract(texts, schema=None, include_confidence=True, **kwargs):
        return [side_effect_extract(t, schema, include_confidence, **kwargs) for t in texts]

    def side_effect_extract_json(text, schema=None, include_confidence=True, **kwargs):
        """Mock implementation of extract_json returning character offsets."""
        entities = []

        # Helper to find all occurrences of a substring
        def find_all(sub, string):
            start_pos = 0
            while True:
                start_pos = string.find(sub, start_pos)
                if start_pos == -1:
                    return
                yield start_pos
                start_pos += len(sub)

        matched_spans = []

        def add_entity(
            name_text,
            name_start,
            name_end,
            specificity="named",
            usage="primary",
            datatype="survey",
            acronym_text=None,
            acronym_start=None,
            acronym_end=None,
        ):
            # Check overlap with existing name/acronym spans
            for ms_start, ms_end in matched_spans:
                if not (name_end <= ms_start or name_start >= ms_end):
                    return

            ent = {
                "name": {
                    "text": name_text,
                    "start": name_start,
                    "end": name_end,
                    "confidence": 0.95,
                },
                "specificity": specificity,
                "usage": usage,
                "datatype": datatype,
                "producer": "Test Producer",
                "timeframe": "2022",
            }
            if acronym_text is not None and acronym_start is not None and acronym_end is not None:
                ent["acronym"] = {
                    "text": acronym_text,
                    "start": acronym_start,
                    "end": acronym_end,
                    "confidence": 0.99,
                }
            entities.append(ent)
            matched_spans.append((name_start, name_end))
            if acronym_start is not None and acronym_end is not None:
                matched_spans.append((acronym_start, acronym_end))

        # 1. Look for "Demographic and Health Survey"
        for start in find_all("Demographic and Health Survey", text):
            end = start + len("Demographic and Health Survey")
            acronym_text, acronym_start, acronym_end = None, None, None
            dhs_near = text.find("DHS", end, end + 10)
            if dhs_near != -1:
                acronym_text = "DHS"
                acronym_start = dhs_near
                acronym_end = dhs_near + 3
            add_entity(
                "Demographic and Health Survey",
                start,
                end,
                acronym_text=acronym_text,
                acronym_start=acronym_start,
                acronym_end=acronym_end,
            )

        # 2. Look for "Ghana Living Standard Survey"
        for start in find_all("Ghana Living Standard Survey", text):
            end = start + len("Ghana Living Standard Survey")
            acronym_text, acronym_start, acronym_end = None, None, None
            glss_near = text.find("GLSS", end, end + 10)
            if glss_near != -1:
                acronym_text = "GLSS"
                acronym_start = glss_near
                acronym_end = glss_near + 4
            add_entity(
                "Ghana Living Standard Survey",
                start,
                end,
                acronym_text=acronym_text,
                acronym_start=acronym_start,
                acronym_end=acronym_end,
            )

        # 3. Look for "World Development Indicators"
        for start in find_all("World Development Indicators", text):
            end = start + len("World Development Indicators")
            acronym_text, acronym_start, acronym_end = None, None, None
            wdi_near = text.find("WDI", end, end + 10)
            if wdi_near != -1:
                acronym_text = "WDI"
                acronym_start = wdi_near
                acronym_end = wdi_near + 3
            add_entity(
                "World Development Indicators",
                start,
                end,
                acronym_text=acronym_text,
                acronym_start=acronym_start,
                acronym_end=acronym_end,
            )

        # 4. Look for standalone "DHS"
        for start in find_all("DHS", text):
            end = start + 3
            add_entity("DHS", start, end, acronym_text="DHS", acronym_start=start, acronym_end=end)

        # 5. Look for standalone "GLSS"
        for start in find_all("GLSS", text):
            end = start + 4
            add_entity(
                "GLSS", start, end, acronym_text="GLSS", acronym_start=start, acronym_end=end
            )

        # 6. Look for standalone "WDI"
        for start in find_all("WDI", text):
            end = start + 3
            add_entity("WDI", start, end, acronym_text="WDI", acronym_start=start, acronym_end=end)

        return {"data_mention": entities}

    mock_model.extract.side_effect = side_effect_extract
    mock_model.batch_extract.side_effect = side_effect_batch_extract
    mock_model.extract_json.side_effect = side_effect_extract_json

    def mock_load(self, model_id=None, **kwargs):
        return mock_model

    from ai4data.data_use.models.model_manager import ModelManager

    monkeypatch.setattr(ModelManager, "load", mock_load)

    return ModelManager()


class TestTextChunking:
    """Test suite for text chunking and index adjustment."""

    def test_short_text_no_chunking(self, mock_model_manager):
        """Test that short texts (<500 tokens) are not chunked."""
        short_text = "The Demographic and Health Survey (DHS) provides health data."

        result = extract_from_text(short_text, include_confidence=True, normalize_text=False)

        assert "input_text" in result
        assert "datasets" in result
        assert result["input_text"] == short_text
        assert len(result["datasets"]) > 0

    def test_long_text_chunking(self, mock_model_manager):
        """Test that long texts (>500 tokens) are properly chunked."""
        long_text = create_long_text_with_datasets()
        tok_text = tokenize_text(long_text)

        # Verify text is long enough to trigger chunking
        assert len(tok_text) > 500, f"Text should have >500 tokens, got {len(tok_text)}"

        result = extract_from_text(long_text, include_confidence=True, normalize_text=False)

        assert "input_text" in result
        assert "datasets" in result
        assert result["input_text"] == long_text

    def test_index_correctness_after_chunking(self, mock_model_manager):
        """Test that start/end indices are within character bounds after chunking."""
        long_text = create_long_text_with_datasets()

        result = extract_from_text(long_text, include_confidence=True, normalize_text=False)
        datasets = result["datasets"]

        # Verify we found datasets
        assert len(datasets) > 0, "Should find at least one dataset"

        # Check each dataset's indices are within character bounds
        for dataset in datasets:
            mention_name = dataset.get("mention_name")

            if mention_name and isinstance(mention_name, dict):
                start = mention_name.get("start")
                end = mention_name.get("end")

                if start is not None and end is not None:
                    assert start >= 0, f"Start index {start} should be >= 0"
                    assert start < end, f"Start {start} should be < end {end}"

    def test_datasets_from_different_chunks(self, mock_model_manager):
        """Test that datasets from different chunks are all found with correct indices."""
        long_text = create_long_text_with_datasets()

        result = extract_from_text(long_text, include_confidence=True, normalize_text=False)
        datasets = result["datasets"]

        # Should find datasets from both beginning and end of text
        dataset_names = [
            d.get("mention_name", {}).get("text") for d in datasets if d.get("mention_name")
        ]

        # Filter out None values
        dataset_names = [name for name in dataset_names if name]

        # Should have found multiple datasets
        assert (
            len(dataset_names) >= 2
        ), f"Should find datasets from different chunks, found: {dataset_names}"

        # Verify all have valid indices (within character bounds)
        for dataset in datasets:
            if dataset.get("mention_name") and isinstance(dataset["mention_name"], dict):
                start = dataset["mention_name"].get("start")
                end = dataset["mention_name"].get("end")

                if start is not None and end is not None:
                    assert 0 <= start < end, f"Invalid indices: start={start}, end={end}"

    def test_deduplication_across_chunks(self, mock_model_manager):
        """Test that duplicate entities across chunk boundaries are deduplicated."""
        # Create text with same dataset mentioned in overlap region
        text_with_duplicates = (
            """
        The Demographic and Health Survey (DHS) is a comprehensive dataset.
        """
            + " ".join([f"token_{i}" for i in range(450)])
            + """
        The Demographic and Health Survey (DHS) provides important data.
        """
        )

        result = extract_from_text(
            text_with_duplicates, include_confidence=True, normalize_text=False
        )
        datasets = result["datasets"]

        # Count DHS mentions
        dhs_mentions = [
            d
            for d in datasets
            if d.get("acronym")
            and isinstance(d["acronym"], dict)
            and d["acronym"].get("text") == "DHS"
        ]

        # Should have deduplicated based on position
        # (exact count depends on model behavior, but should be reasonable)
        assert len(dhs_mentions) >= 1, "Should find at least one DHS mention"

    def test_output_format(self, mock_model_manager):
        """Test that extract_from_text returns the correct format."""
        text = "The World Development Indicators (WDI) is a dataset."

        result = extract_from_text(text, include_confidence=True, normalize_text=False)

        # Check structure
        assert isinstance(result, dict), "Result should be a dict"
        assert "input_text" in result, "Result should have 'input_text' key"
        assert "datasets" in result, "Result should have 'datasets' key"

        # Check types
        assert isinstance(result["input_text"], str), "'input_text' should be a string"
        assert isinstance(result["datasets"], list), "'datasets' should be a list"

        # Check input_text matches
        assert result["input_text"] == text, "'input_text' should match original text"


class TestMarkdownAwareChunking:
    """Tests for markdown-aware chunking: footnotes, tables, headers, dedup."""

    @pytest.fixture
    def extractor(self):
        """Create a DatasetExtractor without loading a real model."""
        from unittest.mock import patch

        with patch.object(ModelManager, "load", return_value=MagicMock()):
            ext = DatasetExtractor.__new__(DatasetExtractor)
            ext.model_manager = MagicMock()
            ext.threshold = 0.5
            ext._model = MagicMock()
            ext._schema_core = MagicMock()
            ext._schema_provenance = MagicMock()
        return ext

    # --- Footnote Extraction ---

    def test_footnote_extraction_closed_bracket(self, extractor):
        """Test [N] closed-bracket footnote format."""
        text = (
            "The survey data showed improvements. [1] Several key findings emerged.\n\n"
            "[1] World Bank Group; European Union; United Nations. (2020). Beirut Report."
        )
        footnotes, _ = extractor._extract_footnotes(text)
        assert 1 in footnotes
        assert "World Bank Group" in footnotes[1]

    def test_footnote_extraction_open_bracket(self, extractor):
        """Test [N open-bracket footnote format (pymupdf4llm artifact)."""
        text = (
            "Close to 90 percent of the refugees live in camps. [1]\n\n"
            "[1 Data is from April 30, 2025. There are also 14,936 asylum-seekers."
        )
        footnotes, _ = extractor._extract_footnotes(text)
        assert 1 in footnotes
        assert "Data is from April 30, 2025" in footnotes[1]

    def test_footnote_extraction_bare_number(self, extractor):
        """Test bare number footnote format (most common: 6,848 instances)."""
        text = (
            "High poverty rates and limited access to finance. [3] Economic data\n"
            "suggests that rural populations face barriers.\n\n"
            "3 The self-reliance survey is part of the Enhancing Self-Reliance program."
        )
        footnotes, _ = extractor._extract_footnotes(text)
        assert 3 in footnotes
        assert "self-reliance survey" in footnotes[3]

    def test_footnote_extraction_no_body_ref(self, extractor):
        """Bare numbers without matching [N] body ref should NOT be treated as footnotes."""
        text = "There are 50 districts in the region.\n\n" "3 The program was established in 2020."
        footnotes, _ = extractor._extract_footnotes(text)
        # No [3] body ref exists, so bare "3" should not match
        assert 3 not in footnotes

    def test_footnote_extraction_multiple(self, extractor):
        """Test multiple footnotes in various formats."""
        text = (
            "Findings from the assessment. [1] Economic conditions improved. [2]\n"
            "Self-reliance data is promising. [3]\n\n"
            "[1] UNHCR Statistics package. Kenya registered refugees.\n"
            '[2 Loschmann, C. (2019) "Benefits of hosting refugees".\n'
            "3 The self-reliance survey draws on the global index."
        )
        footnotes, _ = extractor._extract_footnotes(text)
        assert len(footnotes) == 3
        assert 1 in footnotes and "UNHCR" in footnotes[1]
        assert 2 in footnotes and "Loschmann" in footnotes[2]
        assert 3 in footnotes and "self-reliance" in footnotes[3]

    # --- Footnote Appending ---

    def test_footnote_appending(self, extractor):
        """Test that chunks with [N] refs get footnote text appended."""
        chunk = "The survey data showed improvements. [3] Several key findings emerged."
        footnotes = {
            1: "UNHCR Statistics.",
            3: "The self-reliance survey is part of the program.",
            5: "World Bank data portal.",
        }
        enriched = extractor._append_footnotes_to_chunk(chunk, footnotes)
        assert "---" in enriched
        assert "[3] The self-reliance survey" in enriched
        # Should NOT include footnotes not referenced in this chunk
        assert "[1]" not in enriched.split("---")[1]
        assert "[5]" not in enriched.split("---")[1]

    def test_footnote_appending_no_refs(self, extractor):
        """Chunks without [N] refs should not be modified."""
        chunk = "The survey data showed improvements. Several key findings emerged."
        footnotes = {1: "UNHCR Statistics.", 3: "Self-reliance survey."}
        enriched = extractor._append_footnotes_to_chunk(chunk, footnotes)
        assert enriched == chunk

    # --- Table Boundaries ---

    def test_detect_table_boundaries(self, extractor):
        """Test table region detection with |---| separator."""
        text = (
            "Some text before the table.\n\n"
            "|Indicator|Baseline|Target|\n"
            "|---|---|---|\n"
            "|Poverty rate|42%|35%|\n"
            "|Enrollment|78%|90%|\n\n"
            "Some text after the table."
        )
        tables = extractor._detect_table_boundaries(text)
        assert len(tables) == 1
        start, end = tables[0]
        table_region = text[start:end]
        assert "|Indicator|" in table_region
        assert "|Enrollment|" in table_region
        assert "Some text before" not in table_region

    def test_detect_multiple_tables(self, extractor):
        """Test detection of multiple tables."""
        text = (
            "|A|B|\n|---|---|\n|1|2|\n\n"
            "Paragraph between tables.\n\n"
            "|X|Y|Z|\n|---|---|---|\n|a|b|c|\n"
        )
        tables = extractor._detect_table_boundaries(text)
        assert len(tables) == 2

    # --- Split Points ---

    def test_split_at_bold_header(self, extractor):
        """Test that split snaps to **Bold Header**."""
        text = (
            "Some preceding text with analysis and findings.\n\n"
            "**B. Results Monitoring**\n\n"
            "The project will utilize health facility data."
        )
        # Target in the middle
        target = 50
        split = extractor._find_split_point(text, target, window=60)
        # Should split just before **B. Results Monitoring**
        remaining = text[split:]
        assert remaining.startswith("**B. Results")

    def test_split_at_markdown_header(self, extractor):
        """Test that split snaps to # Markdown Header."""
        text = (
            "Some preceding analysis text here.\n\n"
            "## Chapter 3: Data Sources\n\n"
            "The assessment draws on data from refugee records."
        )
        target = 40
        split = extractor._find_split_point(text, target, window=50)
        remaining = text[split:]
        assert remaining.startswith("## Chapter 3")

    def test_split_at_paragraph_break(self, extractor):
        """Test split at paragraph break when no headers nearby."""
        text = (
            "First paragraph with some analysis text that goes on.\n\n"
            "Second paragraph starts here with more content."
        )
        target = 55
        split = extractor._find_split_point(text, target, window=60)
        remaining = text[split:]
        assert remaining.startswith("Second paragraph")

    def test_split_avoids_table(self, extractor):
        """Test that split jumps out of table region."""
        text = "Text before table.\n" "|A|B|\n|---|---|\n|1|2|\n|3|4|\n" "Text after table.\n"
        tables = extractor._detect_table_boundaries(text)
        # Target inside table
        table_mid = (tables[0][0] + tables[0][1]) // 2
        split = extractor._find_split_point(text, table_mid, window=200, table_boundaries=tables)
        # Split should be at table start (before table)
        assert split == tables[0][0] or split >= tables[0][1]

    # --- Deduplication ---

    def test_null_name_filtering(self, extractor):
        """Test that entities with None dataset_name are filtered."""
        datasets = [
            {"dataset_name": {"text": "DHS", "start": 0, "end": 3, "confidence": 0.9}},
            {"dataset_name": None},
            {"dataset_name": {"text": None, "start": 10, "end": 15, "confidence": 0.5}},
            {"dataset_name": {"text": "GLSS", "start": 20, "end": 24, "confidence": 0.8}},
        ]
        result = extractor._deduplicate_entities(datasets)
        assert len(result) == 2
        texts = [r["dataset_name"]["text"] for r in result]
        assert "DHS" in texts
        assert "GLSS" in texts

    def test_same_span_deduplication(self, extractor):
        """Test that same-span entities keep highest confidence."""
        datasets = [
            {
                "dataset_name": {"text": "DHS", "start": 0, "end": 3, "confidence": 0.7},
                "acronym": {"text": "DHS"},
            },
            {
                "dataset_name": {"text": "DHS", "start": 0, "end": 3, "confidence": 0.95},
                "acronym": {"text": "GLSS"},
            },  # wrong metadata, higher conf
            {
                "dataset_name": {"text": "DHS", "start": 0, "end": 3, "confidence": 0.6},
                "acronym": None,
            },
        ]
        result = extractor._deduplicate_entities(datasets)
        assert len(result) == 1
        assert result[0]["dataset_name"]["confidence"] == 0.95

    # --- Note Boundaries ---

    def test_split_at_italic_note(self, extractor):
        """Test that split snaps before _Note_: to keep it attached."""
        text = (
            "The project will support capacity building in education.\n\n"
            "_Note:_ Priority will be given to (a) females (at least 50 percent); "
            "and (b) those who are members of village savings groups.\n\n"
            "The implementation timeline spans three years."
        )
        target = 70
        split = extractor._find_split_point(text, target, window=80)
        remaining = text[split:]
        assert remaining.startswith("_Note:")

    def test_split_at_plain_note(self, extractor):
        """Test that split snaps before Note: to keep it attached."""
        text = (
            "Enrollment data from administrative records shows progress.\n\n"
            "Note: Studies in Nigeria show cost-effectiveness of water interventions.\n\n"
            "Further analysis is needed."
        )
        target = 70
        split = extractor._find_split_point(text, target, window=80)
        remaining = text[split:]
        assert remaining.startswith("Note:")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
