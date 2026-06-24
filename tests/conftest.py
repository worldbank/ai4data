"""Pytest configuration and fixtures."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_text():
    """Sample text for testing extraction."""
    return """Our analysis uses the 2022 Demographic and Health Survey (DHS) conducted by
the National Statistics Office collected for years 2010-2019 consists of demographic
and employment indicators. The DHS provides nationally representative data for women
aged 15–49, especially on health and fertility indicators."""


@pytest.fixture
def sample_extraction_result():
    """Expected extraction result structure."""
    return {
        "dataset_mention": [
            {
                "dataset_name": {
                    "text": "2022 Demographic and Health Survey",
                    "confidence": 0.91,
                    "start": 4,
                    "end": 9,
                },
                "dataset_tag": "named",
                "acronym": {"text": "DHS", "confidence": 0.99, "start": 10, "end": 11},
                "producer": {
                    "text": "National Statistics Office",
                    "confidence": 0.99,
                    "start": 15,
                    "end": 18,
                },
                "is_used": "True",
                "usage_context": "primary",
            }
        ]
    }


@pytest.fixture
def mock_gliner_model():
    """Mock GLiNER2 model to avoid downloading during tests."""
    mock_model = MagicMock()

    # Mock the create_schema method chain
    mock_schema_builder = MagicMock()
    MagicMock()

    # Setup method chaining
    mock_schema_builder.structure.return_value = mock_schema_builder
    mock_schema_builder.field.return_value = mock_schema_builder

    mock_model.create_schema.return_value = mock_schema_builder

    # Mock extract method
    mock_model.extract.return_value = {
        "entities": {
            "name": [
                {
                    "text": "Test Dataset",
                    "confidence": 0.95,
                    "start": 83,
                    "end": 95,
                    "label": "name",
                }
            ]
        },
        "relation_extraction": {
            "has_acronym": [
                {
                    "head": {"text": "Test Dataset", "start": 83, "end": 95},
                    "tail": {"text": "TD", "start": 96, "end": 98, "confidence": 0.95},
                    "label": "has_acronym",
                    "score": 0.95,
                }
            ]
        },
    }

    # Mock batch_extract method
    mock_model.batch_extract.return_value = [
        {
            "entities": {
                "specificity": [{"text": "named", "confidence": 0.95, "start": 13, "end": 18}],
                "usage": [{"text": "primary", "confidence": 0.95, "start": 47, "end": 54}],
            }
        }
    ]

    # Mock extract_json method
    mock_model.extract_json.return_value = {
        "data_mention": [
            {
                "name": {"text": "Test Dataset", "confidence": 0.95, "start": 99, "end": 104},
                "specificity": "named",
                "usage": "primary",
                "datatype": "survey",
                "acronym": {"text": "TD", "confidence": 0.95, "start": 105, "end": 107},
                "producer": "Test Producer",
                "timeframe": "2022",
            }
        ]
    }

    # Allow load_adapter to be called without errors
    mock_model.load_adapter = MagicMock()

    return mock_model


@pytest.fixture
def mock_classifier_pipeline():
    """Mock HuggingFace text-classification pipeline for the BERT page classifier.

    Returns a callable that behaves like pipeline(text) -> [{"label": ..., "score": ...}].
    Default returns WITH_DATA so tests that expect extraction to proceed do so by default.
    """
    mock_clf = MagicMock()
    mock_clf.return_value = [{"label": "WITH_DATA", "score": 0.97}]
    return mock_clf


@pytest.fixture
def mock_model_manager(monkeypatch, mock_gliner_model, mock_classifier_pipeline):
    """Mock ModelManager to return mock GLiNER2 model and mock BERT classifier."""

    def mock_load(self, model_id=None, **kwargs):
        return mock_gliner_model

    def mock_load_classifier(self, model_id=None):
        return mock_classifier_pipeline

    from ai4data.data_use.models.model_manager import ModelManager

    monkeypatch.setattr(ModelManager, "load", mock_load)
    monkeypatch.setattr(ModelManager, "load_classifier", mock_load_classifier)

    return ModelManager()
