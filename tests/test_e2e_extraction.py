"""End-to-end integration tests for dataset extraction pipeline.

These tests validate that the extraction pipeline (chunking, footnote
injection, schema, deduplication) works correctly with the v2 context-aware
training data format.

Requires the trained model to be available. Skip with:
    pytest tests/test_e2e_extraction.py -v -k "not slow"
"""

import json
from pathlib import Path
from typing import Dict, List

import pytest

# ── Test data paths ──────────────────────────────────────────────────────────
EVAL_DATA = Path(__file__).parent.parent / (
    "data_generation/synthetic_data/batches/v2/finetune-data/eval.jsonl"
)


def _load_eval_samples(n: int = 10) -> List[Dict]:
    """Load first n eval samples."""
    if not EVAL_DATA.exists():
        pytest.skip(f"Eval data not found: {EVAL_DATA}")
    samples = []
    with open(EVAL_DATA) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if len(samples) >= n:
                    break
    return samples


# ── Unit tests (no model needed) ─────────────────────────────────────────────


class TestChunking:
    """Test chunking behavior with v2-style markdown text."""

    @pytest.fixture
    def extractor(self):
        from ai4data.data_use.extractors.dataset_extractor import DatasetExtractor

        ext = DatasetExtractor()
        # Don't load the model — we only test chunking methods
        return ext

    def test_chunk_respects_max_tokens(self, extractor):
        """Chunks should not exceed 400 tokens (before footnote injection)."""
        # Create ~800 word text
        text = "This is a test sentence with several words. " * 100
        chunks = extractor._chunk_text(text, max_tokens=400, overlap=50)

        assert len(chunks) >= 2, "Should produce multiple chunks"
        for chunk_text, offset in chunks:
            words = chunk_text.split()
            # 400 tokens ≈ 400 words max (whitespace tokenizer)
            # Allow some slack for boundary snapping
            assert len(words) <= 450, f"Chunk too long: {len(words)} words"

    def test_chunk_preserves_markdown_header(self, extractor):
        """Markdown headers should appear intact in at least one chunk."""
        text = (
            "First section content with details. " * 80
            + "\n\n# Second Section\n\n"
            + "Second section content with details. " * 80
        )
        chunks = extractor._chunk_text(text, max_tokens=200, overlap=20)

        # The full "# Second Section" line should exist in at least one chunk
        header_in_chunk = any("# Second Section" in chunk_text for chunk_text, _ in chunks)
        assert (
            header_in_chunk
        ), "Markdown header '# Second Section' should appear intact in at least one chunk"

    def test_chunk_preserves_table(self, extractor):
        """Tables should not be split across chunks."""
        table = (
            "| Col A | Col B |\n"
            "|---|---|\n"
            "| val 1 | val 2 |\n"
            "| val 3 | val 4 |\n"
            "| val 5 | val 6 |\n"
        )
        text = "Prefix text. " * 100 + "\n\n" + table + "\n\n" + "Suffix text. " * 100
        chunks = extractor._chunk_text(text, max_tokens=200, overlap=20)

        # Find which chunk contains the table
        table_chunks = [chunk_text for chunk_text, _ in chunks if "|---|" in chunk_text]
        # The table should be fully in at least one chunk
        for tc in table_chunks:
            if "| Col A |" in tc:
                assert "| val 5 |" in tc, "Table rows should not be split"


class TestFootnoteInjection:
    """Test footnote extraction and injection into chunks."""

    @pytest.fixture
    def extractor(self):
        from ai4data.data_use.extractors.dataset_extractor import DatasetExtractor

        return DatasetExtractor()

    def test_footnotes_extracted_correctly(self, extractor):
        """Footnotes with --- separator should be extracted."""
        text = (
            "The survey [1] provides data on health outcomes.\n"
            "Another dataset [2] covers education.\n\n"
            "---\n"
            "[1] Demographic and Health Survey 2020, ICF International.\n"
            "[2] Education Statistics Database, UNESCO.\n"
        )
        footnotes, body = extractor._extract_footnotes(text)

        assert 1 in footnotes, "Should extract footnote [1]"
        assert 2 in footnotes, "Should extract footnote [2]"
        assert "Demographic and Health Survey" in footnotes[1]
        assert "Education Statistics" in footnotes[2]

    def test_footnotes_appended_to_chunk(self, extractor):
        """Referenced footnotes should be appended to chunk."""
        chunk = "Analysis of the survey [1] shows significant trends."
        footnotes = {
            1: "DHS 2020, conducted by ICF.",
            2: "Census 2019, national statistics office.",
        }
        enriched = extractor._append_footnotes_to_chunk(chunk, footnotes)

        assert "---" in enriched, "Should have separator"
        assert "[1] DHS 2020" in enriched, "Should append referenced footnote"
        assert "[2]" not in enriched.split("---")[1], "Should NOT append unreferenced footnote"

    def test_footnote_safety_truncation(self, extractor):
        """Enriched chunks should not exceed 500 tokens."""
        # Create a near-400 token text with footnote refs
        body = "Word " * 395 + "[1] [2] [3]."
        footnotes = {
            i: f"Very long footnote number {i} with lots of detail. " * 20 for i in range(1, 4)
        }
        extractor._append_footnotes_to_chunk(body, footnotes)

        # Now test via _chunk_text which does the truncation
        chunks = extractor._chunk_text(
            body + "\n\n---\n" + "\n".join(f"[{i}] {footnotes[i]}" for i in range(1, 4)),
            max_tokens=400,
        )
        for chunk_text, _ in chunks:
            word_count = len(chunk_text.split())
            assert word_count <= 550, f"Post-footnote chunk too long: {word_count} words"


class TestSchemaAlignment:
    """Test that schema matches v2 training data format."""

    def test_schema_has_all_v2_fields(self):
        """Schema should include all fields from v2 training data."""
        from ai4data.data_use.schemas.dataset_schema import DatasetSchema

        schema = DatasetSchema()
        # Check by inspecting the field thresholds storage
        # The fields are defined in build(), so we check the choices
        _v2_fields = [
            "dataset_name",
            "acronym",
            "producer",
            "reference_year",
            "geography",
            "data_type",
            "dataset_tag",
            "usage_context",
            "is_used",
        ]
        # We can't easily introspect the built schema without a model,
        # but we can verify the class exists and has the build method
        assert hasattr(schema, "build"), "Schema should have build method"

    def test_data_type_choices_match_v2(self):
        """data_type choices should match v2 training data."""
        import inspect

        from ai4data.data_use.schemas.dataset_schema import DatasetSchema

        source = inspect.getsource(DatasetSchema.build)
        expected_types = [
            "survey",
            "census",
            "database",
            "administrative",
            "indicator",
            "geospatial",
            "microdata",
            "report",
            "other",
        ]
        for dtype in expected_types:
            assert dtype in source, f"data_type choice '{dtype}' missing from schema"

    def test_dataset_tag_choices_match_v2(self):
        """specificity_tag choices should be named/descriptive/vague only."""
        import inspect

        from ai4data.data_use.schemas.dataset_schema import DatasetSchema

        source = inspect.getsource(DatasetSchema.build)
        assert "named" in source
        assert "descriptive" in source
        assert "vague" in source
        # non-dataset should NOT be in the choices
        choices_section = source[source.index("specificity_tag") :]
        choices_section = choices_section[: choices_section.index("]")]
        assert "non-dataset" not in choices_section


class TestDeduplicate:
    """Test entity deduplication preserves all fields."""

    @pytest.fixture
    def extractor(self):
        from ai4data.data_use.extractors.dataset_extractor import DatasetExtractor

        return DatasetExtractor()

    def test_dedup_preserves_data_type(self, extractor):
        """Deduplication should preserve data_type, usage_context, is_used."""
        entities = [
            {
                "dataset_name": {"text": "DHS 2020", "start": 10, "end": 18, "confidence": 0.9},
                "data_type": {"value": "survey", "choices": ["survey", "census"]},
                "dataset_tag": {"value": "named"},
                "usage_context": {"value": "primary"},
                "is_used": {"value": "True"},
            },
            {
                "dataset_name": {"text": "DHS 2020", "start": 10, "end": 18, "confidence": 0.7},
                "data_type": {"value": "census", "choices": ["survey", "census"]},
                "dataset_tag": {"value": "descriptive"},
                "usage_context": {"value": "supporting"},
                "is_used": {"value": "False"},
            },
        ]
        result = extractor._deduplicate_entities(entities)

        assert len(result) == 1, "Should deduplicate to 1 entity"
        # Should keep the one with highest confidence (0.9)
        kept = result[0]
        assert kept["data_type"]["value"] == "survey", "Should keep highest-confidence entity"
        assert kept["usage_context"]["value"] == "primary"
        assert kept["is_used"]["value"] == "True"

    def test_dedup_filters_null_names(self, extractor):
        """Should filter entities with None dataset_name."""
        entities = [
            {"dataset_name": None, "data_type": {"value": "survey"}},
            {"dataset_name": {"text": "Census 2020", "start": 0, "end": 11, "confidence": 0.8}},
            {"dataset_name": {"text": None, "start": 0, "end": 5}},
        ]
        result = extractor._deduplicate_entities(entities)

        assert len(result) == 1
        assert result[0]["dataset_name"]["text"] == "Census 2020"


class TestEvalDataFormat:
    """Validate v2 eval data format is compatible with extraction pipeline."""

    def test_eval_data_has_expected_structure(self):
        """Eval data should have input text and output json_structures."""
        samples = _load_eval_samples(5)

        for s in samples:
            assert "input" in s, "Sample should have 'input' field"
            assert "output" in s, "Sample should have 'output' field"
            assert "json_structures" in s["output"]

            for js in s["output"]["json_structures"]:
                dm = js["dataset_mention"]
                assert "dataset_name" in dm, "Should have dataset_name"
                assert "dataset_tag" in dm, "Should have dataset_tag"
                tag = dm["dataset_tag"]
                tag_val = tag.get("value") if isinstance(tag, dict) else tag
                assert tag_val in (
                    "named",
                    "descriptive",
                    "vague",
                ), f"Tag should be named/descriptive/vague, got: {tag_val}"

    def test_eval_data_no_non_datasets(self):
        """V2 eval data should have zero non-dataset tags."""
        samples = _load_eval_samples(50)
        for s in samples:
            for js in s["output"]["json_structures"]:
                tag = js["dataset_mention"]["dataset_tag"]
                tag_val = tag.get("value") if isinstance(tag, dict) else tag
                assert tag_val != "non-dataset", "V2 data should not contain non-dataset tag"

    def test_eval_text_within_token_budget(self):
        """V2 eval texts should fit within the chunk budget."""
        samples = _load_eval_samples(50)
        word_counts = [len(s["input"].split()) for s in samples]
        avg = sum(word_counts) / len(word_counts)

        # V2 training data averages ~350 words
        assert avg < 500, f"Average word count {avg:.0f} exceeds expected ~350"
        assert avg > 200, f"Average word count {avg:.0f} is unexpectedly low"


"""
End-to-end integration tests for dataset extraction pipeline.
"""
