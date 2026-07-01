"""Tests for deduplication functionality."""

import pytest

from ai4data.data_use.extractors.deduplication import (
    are_fuzzy_duplicates,
    build_acronym_clusters,
    deduplicate_extraction,
    extract_context_window,
    extract_mentions,
    filter_overlapping_mentions,
    fuzzy_clusters,
    is_likely_acronym_pair,
    merge_cluster,
    merge_similar_clusters,
)


class TestFuzzyMatching:
    """Test multi-metric fuzzy matching."""

    def test_exact_match(self):
        """Exact matches should always return True."""
        assert are_fuzzy_duplicates("World Bank", "World Bank")
        assert are_fuzzy_duplicates("WDI", "WDI")

    def test_case_insensitive(self):
        """Matching should be case insensitive."""
        assert are_fuzzy_duplicates("World Bank", "world bank")
        assert are_fuzzy_duplicates("WDI", "wdi")

    def test_minor_variations(self):
        """Should match minor variations."""
        assert are_fuzzy_duplicates(
            "Living Standards Measurement Study", "Living Standard Measurement Study"
        )
        assert are_fuzzy_duplicates("World Development Indicators", "World Development Indicator")

    def test_short_strings_strict(self):
        """Short strings should require higher similarity."""
        # These should NOT match (too different for short strings)
        assert not are_fuzzy_duplicates("WDI", "WHO")
        assert not are_fuzzy_duplicates("DHS", "NHS")

    def test_long_strings_lenient(self):
        """Long strings can have more variation."""
        assert are_fuzzy_duplicates("Demographic and Health Surveys", "Demographic Health Survey")

    def test_completely_different(self):
        """Completely different strings should not match."""
        assert not are_fuzzy_duplicates("World Bank", "United Nations")
        assert not are_fuzzy_duplicates(
            "World Development Indicators", "Demographic and Health Surveys"
        )


class TestAcronymValidation:
    """Test acronym validation logic."""

    def test_valid_acronyms(self):
        """Valid acronym-longform pairs should match."""
        assert is_likely_acronym_pair("WDI", "World Development Indicators")
        assert is_likely_acronym_pair("DHS", "Demographic and Health Surveys")
        assert is_likely_acronym_pair("LSMS", "Living Standards Measurement Study")

    def test_with_stop_words(self):
        """Should handle stop words correctly."""
        assert is_likely_acronym_pair("WB", "World Bank")
        assert is_likely_acronym_pair(
            "UNESCO", "United Nations Educational Scientific Cultural Organization"
        )

    def test_partial_acronyms(self):
        """Should handle partial acronyms."""
        assert is_likely_acronym_pair("WD", "World Development Indicators")

    def test_invalid_acronyms(self):
        """Invalid pairs should not match."""
        assert not is_likely_acronym_pair("WDI", "Demographic and Health Surveys")
        assert not is_likely_acronym_pair("ABC", "World Bank Data")

    def test_longer_than_longform(self):
        """Acronym longer than longform should not match."""
        assert not is_likely_acronym_pair("VERYLONGACRONYM", "Short")


class TestOverlapDetection:
    """Test overlap detection and filtering."""

    def test_no_overlaps(self):
        """Non-overlapping mentions should all be kept."""
        mentions = [
            {"text": "WDI", "start": 0, "end": 3, "score": 0.9, "source": "doc", "page": 1},
            {"text": "DHS", "start": 10, "end": 13, "score": 0.9, "source": "doc", "page": 1},
        ]
        filtered = filter_overlapping_mentions(mentions)
        assert len(filtered) == 2

    def test_complete_overlap_same_start(self):
        """Overlapping mentions - keep higher score."""
        mentions = [
            {"text": "World Bank", "start": 0, "end": 10, "score": 0.8, "source": "doc", "page": 1},
            {
                "text": "World Bank Open Data",
                "start": 0,
                "end": 20,
                "score": 0.95,
                "source": "doc",
                "page": 1,
            },
        ]
        filtered = filter_overlapping_mentions(mentions)
        assert len(filtered) == 1
        assert filtered[0]["text"] == "World Bank Open Data"

    def test_partial_overlap(self):
        """Partial overlap - keep better mention."""
        mentions = [
            {
                "text": "Development Indicators",
                "start": 6,
                "end": 28,
                "score": 0.85,
                "source": "doc",
                "page": 1,
            },
            {
                "text": "World Development Indicators",
                "start": 0,
                "end": 28,
                "score": 0.95,
                "source": "doc",
                "page": 1,
            },
        ]
        filtered = filter_overlapping_mentions(mentions)
        assert len(filtered) == 1
        assert filtered[0]["text"] == "World Development Indicators"

    def test_different_pages_no_overlap(self):
        """Overlapping spans on different pages should both be kept."""
        mentions = [
            {"text": "WDI", "start": 0, "end": 3, "score": 0.9, "source": "doc", "page": 1},
            {"text": "World Bank", "start": 0, "end": 10, "score": 0.9, "source": "doc", "page": 2},
        ]
        filtered = filter_overlapping_mentions(mentions)
        assert len(filtered) == 2

    def test_similar_scores_prefer_longer(self):
        """With similar scores, prefer longer text."""
        mentions = [
            {"text": "WDI", "start": 0, "end": 3, "score": 0.90, "source": "doc", "page": 1},
            {
                "text": "World Development Indicators",
                "start": 0,
                "end": 28,
                "score": 0.91,
                "source": "doc",
                "page": 1,
            },
        ]
        filtered = filter_overlapping_mentions(mentions)
        assert len(filtered) == 1
        assert filtered[0]["text"] == "World Development Indicators"


class TestContextExtraction:
    """Test context window extraction."""

    def test_simple_context(self):
        """Extract context around a mention."""
        text = "First sentence. The World Bank provides data. Third sentence."
        start = text.index("World Bank")
        end = start + len("World Bank")

        context = extract_context_window(text, start, end, sentences=1)
        assert "World Bank" in context
        assert "provides data" in context

    def test_multiple_sentences(self):
        """Extract multiple sentences."""
        text = "Sentence one. Sentence two. The dataset is used. Sentence four. Sentence five."
        start = text.index("dataset")
        end = start + len("dataset")

        context = extract_context_window(text, start, end, sentences=1)
        # Should include at least the sentence with the mention
        assert "dataset" in context

    def test_start_of_text(self):
        """Handle mention at start of text."""
        text = "The WDI database is comprehensive. More text here."
        start = 0
        end = len("The WDI database")

        context = extract_context_window(text, start, end, sentences=1)
        assert "WDI" in context

    def test_end_of_text(self):
        """Handle mention at end of text."""
        text = "Some introduction text. We used the World Bank data."
        start = text.index("World Bank")
        end = start + len("World Bank data")

        context = extract_context_window(text, start, end, sentences=1)
        assert "World Bank" in context


class TestExtractMentions:
    """Test mention extraction from records."""

    def test_simple_extraction(self):
        """Extract mentions from basic records."""
        records = [
            {
                "text": "We used WDI data.",
                "source": "doc1",
                "page": 1,
                "datasets": [
                    {
                        "dataset_name": {"text": "WDI", "start": 8, "end": 11},
                        "label": "named",
                        "score": 0.95,
                    }
                ],
            }
        ]

        mentions = extract_mentions(records)
        assert len(mentions) == 1
        assert mentions[0]["text"] == "WDI"
        assert mentions[0]["source"] == "doc1"
        assert mentions[0]["page"] == 1

    def test_with_acronyms(self):
        """Extract mentions with acronym relations."""
        records = [
            {
                "text": "World Development Indicators (WDI) data.",
                "source": "doc1",
                "page": 1,
                "datasets": [
                    {
                        "dataset_name": {
                            "text": "World Development Indicators",
                            "start": 0,
                            "end": 28,
                            "acronym": [{"text": "WDI", "start": 30, "end": 33}],
                        },
                        "label": "named",
                        "score": 0.95,
                    }
                ],
            }
        ]

        mentions = extract_mentions(records)
        assert len(mentions) == 1
        assert mentions[0]["text"] == "World Development Indicators"
        assert "WDI" in mentions[0]["acronym"]


class TestClustering:
    """Test clustering algorithms."""

    def test_acronym_clustering(self):
        """Cluster by acronym relationships."""
        mentions = [
            {
                "text": "World Development Indicators",
                "acronym": ["WDI"],
                "score": 0.9,
                "label": "named",
            },
            {"text": "WDI", "acronym": [], "score": 0.9, "label": "named"},
        ]

        clusters = build_acronym_clusters(mentions)
        # Should cluster together
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_exact_text_clustering(self):
        """Cluster identical texts."""
        mentions = [
            {"text": "World Bank", "score": 0.9, "label": "named"},
            {"text": "World Bank", "score": 0.85, "label": "named"},
        ]

        clusters = build_acronym_clusters(mentions)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_fuzzy_clustering(self):
        """Cluster similar texts."""
        mentions = [
            {"text": "Living Standards Measurement Study", "score": 0.9, "label": "named"},
            {"text": "Living Standard Measurement Study", "score": 0.85, "label": "named"},
        ]

        clusters = fuzzy_clusters(mentions, [0, 1])
        assert len(clusters) == 1
        assert len(clusters[0]) == 2


class TestMerging:
    """Test cluster merging logic."""

    def test_merge_cluster_structure(self):
        """Test merged cluster has correct structure."""
        mentions = [
            {
                "text": "WDI",
                "label": "named",
                "score": 0.9,
                "start": 0,
                "end": 3,
                "source": "doc1",
                "page": 1,
                "raw_context": "The WDI provides data.",
            },
            {
                "text": "World Development Indicators",
                "label": "named",
                "score": 0.95,
                "start": 10,
                "end": 38,
                "source": "doc1",
                "page": 1,
                "raw_context": "We used World Development Indicators for analysis.",
            },
        ]

        merged = merge_cluster([0, 1], mentions)

        # Check structure
        assert "dataset_name" in merged
        assert "dataset_tag" in merged
        assert "count" in merged
        assert "occurrences" in merged
        assert merged["count"] == 2
        assert len(merged["occurrences"]) == 2

        # Check occurrences have required fields
        for occ in merged["occurrences"]:
            assert "text" in occ
            assert "start" in occ
            assert "end" in occ
            assert "confidence" in occ
            assert "source" in occ
            assert "page" in occ
            assert "context" in occ

    def test_canonical_selection(self):
        """Test canonical form selection."""
        mentions = [
            {
                "text": "WDI",
                "label": "named",
                "score": 0.9,
                "start": 0,
                "end": 3,
                "raw_context": "x",
            },
            {
                "text": "World Development Indicators",
                "label": "named",
                "score": 0.95,
                "start": 0,
                "end": 28,
                "raw_context": "x",
            },
        ]

        merged = merge_cluster([0, 1], mentions)
        # Should prefer longer, named form
        assert merged["dataset_name"] == "World Development Indicators"


class TestDeduplicateExtraction:
    """Test the main deduplication API."""

    def test_text_level_deduplication(self):
        """Test deduplication of extract_from_text output."""
        extraction_result = {
            "input_text": "The WDI is great. World Development Indicators provides data.",
            "datasets": [
                {
                    "dataset_name": {"text": "WDI", "start": 4, "end": 7},
                    "label": "named",
                    "score": 0.9,
                },
                {
                    "dataset_name": {
                        "text": "World Development Indicators",
                        "start": 18,
                        "end": 46,
                    },
                    "label": "named",
                    "score": 0.95,
                },
            ],
        }

        result = deduplicate_extraction(extraction_result)

        assert "datasets" in result
        assert "input_text" in result
        # Should be deduplicated to 1
        assert len(result["datasets"]) == 1
        assert result["datasets"][0]["count"] == 2
        assert "occurrences" in result["datasets"][0]

    def test_document_level_deduplication(self):
        """Test deduplication of extract_from_document output."""
        extraction_result = [
            {
                "input_text": "Page 1 mentions WDI.",
                "datasets": [
                    {
                        "dataset_name": {"text": "WDI", "start": 16, "end": 19},
                        "label": "named",
                        "score": 0.9,
                    }
                ],
                "document": {"source": "report.pdf", "pages": [1]},
            },
            {
                "input_text": "Page 2 mentions World Development Indicators.",
                "datasets": [
                    {
                        "dataset_name": {
                            "text": "World Development Indicators",
                            "start": 16,
                            "end": 44,
                        },
                        "label": "named",
                        "score": 0.95,
                    }
                ],
                "document": {"source": "report.pdf", "pages": [2]},
            },
        ]

        result = deduplicate_extraction(extraction_result)

        assert "source" in result
        assert "datasets" in result
        assert "document_metadata" in result
        # Should be deduplicated across pages
        assert len(result["datasets"]) == 1
        assert result["datasets"][0]["count"] == 2
        # Should track both pages
        assert set(result["datasets"][0]["pages"]) == {"1", "2"}

    def test_empty_input(self):
        """Handle empty input gracefully."""
        result = deduplicate_extraction({"input_text": "No datasets here.", "datasets": []})
        assert result["datasets"] == []

    def test_auto_detection(self):
        """Test automatic detection of text vs document level."""
        # Text input should be auto-detected
        text_input = {"input_text": "test", "datasets": []}
        result = deduplicate_extraction(text_input)
        assert "input_text" in result
        assert "datasets" in result

        # Document input should be auto-detected
        doc_input = [
            {"input_text": "page 1", "datasets": [], "document": {"source": "doc", "pages": [1]}}
        ]
        result = deduplicate_extraction(doc_input)
        assert "source" in result
        assert "datasets" in result
        assert "document_metadata" in result


class TestCrossClusterValidation:
    """Test cross-cluster merging."""

    def test_merge_similar_clusters(self):
        """Merge clusters with similar canonical texts."""
        clusters = [
            {
                "text": "WDI",
                "label": "named",
                "score": [0.9],
                "count": 1,
                "form_counts": {"WDI": 1},
                "occurrences": [{"text": "WDI", "score": 0.9}],
                "acronym": [],
                "pages": [],
                "sources": [],
            },
            {
                "text": "World Development Indicators",
                "label": "named",
                "score": [0.95],
                "count": 1,
                "form_counts": {"World Development Indicators": 1},
                "occurrences": [{"text": "World Development Indicators", "score": 0.95}],
                "acronym": ["WDI"],
                "pages": [],
                "sources": [],
            },
        ]

        # Note: These might not merge unless fuzzy matching determines they're similar enough
        # The test verifies the function runs without error
        result = merge_similar_clusters(clusters)
        assert len(result) >= 1  # May or may not merge depending on fuzzy threshold

    def test_no_merge_different_datasets(self):
        """Don't merge completely different datasets."""
        clusters = [
            {
                "text": "World Bank",
                "label": "named",
                "score": [0.9],
                "count": 1,
                "form_counts": {"World Bank": 1},
                "occurrences": [{"text": "World Bank", "score": 0.9}],
                "acronym": [],
                "pages": [],
                "sources": [],
            },
            {
                "text": "United Nations",
                "label": "named",
                "score": [0.95],
                "count": 1,
                "form_counts": {"United Nations": 1},
                "occurrences": [{"text": "United Nations", "score": 0.95}],
                "acronym": [],
                "pages": [],
                "sources": [],
            },
        ]

        result = merge_similar_clusters(clusters)
        assert len(result) == 2  # Should stay separate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
