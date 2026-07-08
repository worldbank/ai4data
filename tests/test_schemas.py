"""Tests for schema builders."""

from ai4data.data_use.schemas.dataset_schema import DatasetSchema


class TestDatasetSchema:
    """Test suite for DatasetSchema class."""

    def test_initialization_default(self):
        """Test schema initialization with default threshold."""
        schema = DatasetSchema()
        assert schema.threshold == 0.5
        assert schema._field_thresholds == {}

    def test_initialization_custom_threshold(self):
        """Test schema initialization with custom threshold."""
        schema = DatasetSchema(threshold=0.9)
        assert schema.threshold == 0.9

    def test_set_threshold(self):
        """Test setting field-specific threshold."""
        schema = DatasetSchema()
        result = schema.set_threshold("dataset_name", 0.95)

        # Should return self for chaining
        assert result is schema
        assert schema._field_thresholds["dataset_name"] == 0.95

    def test_set_multiple_thresholds(self):
        """Test setting multiple field thresholds."""
        schema = DatasetSchema()
        schema.set_threshold("dataset_name", 0.95)
        schema.set_threshold("acronym", 0.99)

        assert schema._field_thresholds["dataset_name"] == 0.95
        assert schema._field_thresholds["acronym"] == 0.99

    def test_method_chaining(self):
        """Test that set_threshold supports method chaining."""
        schema = DatasetSchema().set_threshold("dataset_name", 0.95).set_threshold("acronym", 0.99)

        assert schema._field_thresholds["dataset_name"] == 0.95
        assert schema._field_thresholds["acronym"] == 0.99

    def test_build_with_mock_extractor(self, mock_gliner_model):
        """Test building schema with mock extractor."""
        schema = DatasetSchema()
        built_schema = schema.build(mock_gliner_model)

        # Verify create_schema was called
        mock_gliner_model.create_schema.assert_called_once()
        assert built_schema is not None

    def test_default_threshold_constant(self):
        """Test that DEFAULT_THRESHOLD constant is accessible."""
        assert DatasetSchema.DEFAULT_THRESHOLD == 0.5


class TestDatasetSchemaV2EdgeCases:
    """Test suite for edge cases of single-pass V5 DatasetSchemaV2."""

    def test_missing_or_null_fields(self):
        """Test handling of missing or null fields in model output."""
        from unittest.mock import MagicMock

        from ai4data.data_use.schemas.dataset_schema_v2 import DatasetSchemaV2

        mock_model = MagicMock()
        mock_model.extract_json.return_value = {
            "data_mention": [
                {"name": {"text": "Demographic Survey", "confidence": 0.9, "start": 99, "end": 117}}
            ]
        }

        schema = DatasetSchemaV2()
        results = schema.extract_with_classification("Some text", mock_model)

        assert len(results) == 1
        res = results[0]
        assert res["mention_name"]["text"] == "Demographic Survey"
        assert res["specificity_tag"]["text"] == "named"
        assert res["usage_context"]["text"] == "primary"
        assert res["is_used"]["text"] == "True"

    def test_invalid_negative_offsets(self):
        """Test start and end offsets bounding when indices are smaller than label prefix."""
        from unittest.mock import MagicMock

        from ai4data.data_use.schemas.dataset_schema_v2 import DatasetSchemaV2

        mock_model = MagicMock()
        mock_model.extract_json.return_value = {
            "data_mention": [{"name": {"text": "DHS", "confidence": 0.95, "start": 10, "end": 13}}]
        }

        schema = DatasetSchemaV2()
        results = schema.extract_with_classification("DHS text", mock_model)

        assert len(results) == 1
        res = results[0]
        assert res["mention_name"]["start"] == 0
        assert res["mention_name"]["end"] == 0

    def test_string_name_field_type(self):
        """Test name field when returned as a raw string instead of a dictionary."""
        from unittest.mock import MagicMock

        from ai4data.data_use.schemas.dataset_schema_v2 import DatasetSchemaV2

        mock_model = MagicMock()
        mock_model.extract_json.return_value = {
            "data_mention": [
                {"name": "Demographic Survey", "specificity": "named", "usage": "primary"}
            ]
        }

        schema = DatasetSchemaV2()
        results = schema.extract_with_classification("Demographic Survey", mock_model)

        assert len(results) == 1
        res = results[0]
        assert res["mention_name"]["text"] == "Demographic Survey"
        assert res["mention_name"]["confidence"] == 1.0
        assert res["mention_name"]["start"] is None
        assert res["mention_name"]["end"] is None

    def test_mixed_casing_and_spaces_in_usage(self):
        """Test mixed casing and spaces in the usage tag mapping to is_used."""
        from unittest.mock import MagicMock

        from ai4data.data_use.schemas.dataset_schema_v2 import DatasetSchemaV2

        mock_model = MagicMock()
        mock_model.extract_json.return_value = {
            "data_mention": [
                {
                    "name": {"text": "DHS", "confidence": 0.9, "start": 99, "end": 102},
                    "usage": "  Background  ",
                }
            ]
        }

        schema = DatasetSchemaV2()
        results = schema.extract_with_classification("DHS", mock_model)

        assert len(results) == 1
        res = results[0]
        assert res["usage_context"]["text"] == "Background"
        assert res["is_used"]["text"] == "False"

    def test_typology_tag_whitelist_v2(self):
        """Test that typology_tag is coerced to whitelist values or 'other' in V2 schema."""
        from unittest.mock import MagicMock

        from ai4data.data_use.schemas.dataset_schema_v2 import DatasetSchemaV2

        mock_model = MagicMock()
        mock_model.extract_json.return_value = {
            "data_mention": [
                {
                    "name": {
                        "text": "Demographic Survey",
                        "confidence": 0.9,
                        "start": 99,
                        "end": 117,
                    },
                    "datatype": "SURVEY",  # mixed case, should map to 'survey'
                },
                {
                    "name": {"text": "Census Data", "confidence": 0.8, "start": 120, "end": 131},
                    "datatype": "estimation",  # should map to 'estimates'
                },
                {
                    "name": {"text": "Admin system", "confidence": 0.75, "start": 140, "end": 152},
                    "datatype": "administrative records",  # should map to 'administrative'
                },
                {
                    "name": {"text": "Index Data", "confidence": 0.75, "start": 160, "end": 170},
                    "datatype": "indices",  # should map to 'indicator'
                },
                {
                    "name": {"text": "Platform Data", "confidence": 0.70, "start": 180, "end": 195},
                    "datatype": "information platform",  # should map to 'administrative'
                },
                {
                    "name": {"text": "Random Data", "confidence": 0.70, "start": 200, "end": 211},
                    "datatype": "unknown_type",  # invalid type, should map to 'other'
                },
            ]
        }

        schema = DatasetSchemaV2()
        results = schema.extract_with_classification("Some text", mock_model)

        assert len(results) == 6
        assert results[0]["typology_tag"]["text"] == "survey"
        assert results[1]["typology_tag"]["text"] == "estimates"
        assert results[2]["typology_tag"]["text"] == "administrative"
        assert results[3]["typology_tag"]["text"] == "indicator"
        assert results[4]["typology_tag"]["text"] == "administrative"
        assert results[5]["typology_tag"]["text"] == "other"


class TestDatasetSchemaV3EdgeCases:
    """Test suite for edge cases of DatasetSchemaV3."""

    def test_typology_tag_whitelist_v3(self):
        """Test that typology_tag is coerced to whitelist values or 'other' in V3 schema."""
        from unittest.mock import MagicMock

        from ai4data.data_use.schemas.dataset_schema_v2 import DatasetSchemaV3

        mock_model = MagicMock()
        mock_model.extract.return_value = {
            "entities": {
                "name": [
                    {"text": "Demographic Survey", "confidence": 0.9, "start": 50, "end": 68},
                    {"text": "Census Data", "confidence": 0.8, "start": 80, "end": 91},
                    {"text": "Geospatial Data", "confidence": 0.85, "start": 110, "end": 125},
                ]
            },
            "relation_extraction": {
                "has_datatype": [
                    {
                        "head": {"start": 50, "end": 68},
                        "tail": {
                            "text": "  SURVEY  ",
                            "confidence": 0.95,
                            "start": 100,
                            "end": 108,
                        },
                    },
                    {
                        "head": {"start": 80, "end": 91},
                        "tail": {
                            "text": "estimations",
                            "confidence": 0.90,
                            "start": 120,
                            "end": 131,
                        },
                    },
                    {
                        "head": {"start": 110, "end": 125},
                        "tail": {
                            "text": "spatial analysis",
                            "confidence": 0.85,
                            "start": 140,
                            "end": 156,
                        },
                    },
                ],
                "has_specificity": [
                    {
                        "head": {"start": 50, "end": 68},
                        "tail": {"text": "named", "confidence": 0.95, "start": 100, "end": 105},
                    },
                    {
                        "head": {"start": 80, "end": 91},
                        "tail": {"text": "named", "confidence": 0.95, "start": 100, "end": 105},
                    },
                    {
                        "head": {"start": 110, "end": 125},
                        "tail": {"text": "named", "confidence": 0.95, "start": 100, "end": 105},
                    },
                ],
                "has_usage": [
                    {
                        "head": {"start": 50, "end": 68},
                        "tail": {"text": "primary", "confidence": 0.95, "start": 100, "end": 107},
                    },
                    {
                        "head": {"start": 80, "end": 91},
                        "tail": {"text": "primary", "confidence": 0.95, "start": 100, "end": 107},
                    },
                    {
                        "head": {"start": 110, "end": 125},
                        "tail": {"text": "primary", "confidence": 0.95, "start": 100, "end": 107},
                    },
                ],
            },
        }

        schema = DatasetSchemaV3()
        mock_model.batch_extract.return_value = []

        results = schema.extract_with_classification(
            "specificity: named | usage: primary | Some text...", mock_model
        )

        assert len(results) == 3
        # Sort results by mention name text to make assertions independent of confidence-based sorting
        sorted_results = sorted(results, key=lambda r: r["mention_name"]["text"])
        assert sorted_results[0]["typology_tag"]["text"] == "estimates"
        assert sorted_results[1]["typology_tag"]["text"] == "survey"
        assert sorted_results[2]["typology_tag"]["text"] == "geospatial"
