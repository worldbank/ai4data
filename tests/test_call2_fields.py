"""Call 2 (LFM usage + attributes) fields: schema vocabulary, extractor output, dedup merge."""

from ai4data.data_use import deduplicate_extraction
from ai4data.data_use.extractors.dataset_extractor import DatasetExtractor
from ai4data.data_use.models.metadata_model import _parse_json
from ai4data.data_use.schemas import (
    ENTITY_LABELS,
    THEME_TAGS,
    USAGE_ACTION_TO_PURPOSE,
    USAGE_ACTIONS,
)
from ai4data.data_use.schemas.dataset_schema import PURPOSE_ACTION_LABELS


class TestVocabulary:
    def test_entity_labels(self):
        assert ENTITY_LABELS == {
            "NAMED_DATA": "named",
            "DESCRIPTIVE_DATA": "descriptive",
            "VAGUE_DATA": "vague",
        }

    def test_every_usage_action_maps_to_a_purpose_label(self):
        assert set(USAGE_ACTION_TO_PURPOSE) == set(USAGE_ACTIONS)
        assert set(USAGE_ACTION_TO_PURPOSE.values()) <= set(PURPOSE_ACTION_LABELS)


class TestParseJson:
    def test_plain_and_thinking_output(self):
        assert _parse_json('{"a": 1}') == {"a": 1}
        assert _parse_json('reasoning<|im_start|>{"a": 1}') == {"a": 1}

    def test_invalid_returns_none(self):
        assert _parse_json("no json") is None
        assert _parse_json("{broken") is None


class TestExtractorOutput:
    def test_named_mention_carries_call2_fields(self, mock_model_manager):
        out = DatasetExtractor().extract_from_text("DHS provides data.", use_classifier=False)
        ds = out["datasets"][0]
        assert ds["usage_context"]["text"] == "primary"
        assert ds["usage_action"]["text"] == "analyze"
        assert ds["usage_summary"]["text"] == "Used."
        assert ds["purpose_action"]["text"] == "quantitative_analysis"
        assert ds["is_used"]["text"] == "True"
        assert ds["theme_tags"] == ["health"]
        # attributes are nested under "attributes" in the model JSON
        assert ds["source_attributes"]["year"] == "2022"
        assert ds["source_attributes"]["data_type"] == "survey"
        assert ds["source_attributes"]["producer"] is None
        assert set(ds["source_attributes"]) == {
            "geography", "year", "version", "producer", "acronym", "data_type"
        }
        assert set(ds["theme_tags"]) <= set(THEME_TAGS)


def _rec(page, name, action, tags, attrs, summary="s"):
    f = lambda t: {"text": t, "confidence": 1.0, "start": 0, "end": len(name)}  # noqa: E731
    return {
        "document": {"source": "doc.pdf", "pages": [page]},
        "input_text": name,
        "datasets": [
            {
                "mention_name": f(name),
                "specificity_tag": "named",
                "usage_context": f("primary"),
                "is_used": f("True"),
                "usage_action": f(action),
                "usage_summary": f(summary),
                "purpose_action": f("quantitative_analysis"),
                "theme_tags": tags,
                "source_attributes": attrs,
            }
        ],
    }


class TestDeduplicationMerge:
    def test_new_fields_survive_and_merge(self):
        res = [
            _rec(1, "Household Survey", "source", ["poverty"], {"year": "2024", "producer": None}),
            _rec(2, "Household Survey", "source", ["employment"], {"year": "2024", "producer": "NSO"}),
            _rec(3, "Household Survey", "analyze", ["poverty"], {"year": "2023", "producer": None}),
        ]
        merged = deduplicate_extraction(res)["datasets"]
        assert len(merged) == 1
        m = merged[0]
        assert m["usage_action"] == "source"  # most frequent
        assert m["purpose_action"] == "quantitative_analysis"
        assert m["theme_tags"] == ["employment", "poverty"]  # union
        assert m["source_attributes"] == {"year": "2024", "producer": "NSO"}
        assert m["usage_summary"] == "s"
        assert {o["usage_action"] for o in m["occurrences"]} == {"source", "analyze"}
