"""Tests for schema builders."""

from ai4data.data_use.schemas.dataset_schema import DatasetSchema


class TestDatasetSchemaEdgeCases:
    """Test suite for edge cases of DatasetSchema."""

    def test_typology_tag_whitelist_v3(self):
        """Test that typology_tag is coerced to whitelist values or 'other' in V3 schema."""
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.extract.return_value = {
            "entities": {
                "named_data": [
                    {"text": "Demographic Survey", "confidence": 0.9, "start": 0, "end": 18},
                    {"text": "Census Data", "confidence": 0.8, "start": 20, "end": 31},
                    {"text": "Geospatial Data", "confidence": 0.85, "start": 33, "end": 48},
                ]
            },
            "relation_extraction": {
                "has_datatype": [
                    {
                        "head": {"start": 0, "end": 18},
                        "tail": {"text": "  SURVEY  ", "confidence": 0.95, "start": 0, "end": 18},
                    },
                    {
                        "head": {"start": 20, "end": 31},
                        "tail": {"text": "estimations", "confidence": 0.90, "start": 20, "end": 31},
                    },
                    {
                        "head": {"start": 33, "end": 48},
                        "tail": {
                            "text": "spatial analysis",
                            "confidence": 0.85,
                            "start": 33,
                            "end": 48,
                        },
                    },
                ],
                "has_specificity": [
                    {
                        "head": {"start": 0, "end": 18},
                        "tail": {"text": "named", "confidence": 0.95, "start": 0, "end": 18},
                    },
                    {
                        "head": {"start": 20, "end": 31},
                        "tail": {"text": "named", "confidence": 0.95, "start": 20, "end": 31},
                    },
                    {
                        "head": {"start": 33, "end": 48},
                        "tail": {"text": "named", "confidence": 0.95, "start": 33, "end": 48},
                    },
                ],
                "has_usage": [
                    {
                        "head": {"start": 0, "end": 18},
                        "tail": {"text": "primary", "confidence": 0.95, "start": 0, "end": 18},
                    },
                    {
                        "head": {"start": 20, "end": 31},
                        "tail": {"text": "primary", "confidence": 0.95, "start": 20, "end": 31},
                    },
                    {
                        "head": {"start": 33, "end": 48},
                        "tail": {"text": "primary", "confidence": 0.95, "start": 33, "end": 48},
                    },
                ],
            },
        }

        schema = DatasetSchema()
        mock_model.batch_extract.return_value = [
            {
                "typology": {"label": "survey", "confidence": 0.95},
                "usage": {"label": "primary", "confidence": 0.95},
            },
            {
                "typology": {"label": "geospatial", "confidence": 0.95},
                "usage": {"label": "primary", "confidence": 0.95},
            },
            {
                "typology": {"label": "estimates", "confidence": 0.95},
                "usage": {"label": "primary", "confidence": 0.95},
            },
        ]

        results = schema.extract_with_classification(
            "Demographic Survey. Census Data. Geospatial Data.", mock_model
        )

        assert len(results) == 3
        sorted_results = sorted(results, key=lambda r: r["mention_name"]["text"])
        assert sorted_results[0]["typology_tag"]["text"] == "estimates"
        assert sorted_results[1]["typology_tag"]["text"] == "survey"
        assert sorted_results[2]["typology_tag"]["text"] == "geospatial"
