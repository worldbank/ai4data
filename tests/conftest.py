"""Pytest configuration and fixtures."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _auto_mock_classifier(monkeypatch, mock_classifier_pipeline, request):
    """Mock load_classifier for every test (skipped for slow/e2e tests)."""
    if request.node.get_closest_marker("slow"):
        return
    if "test_models" in str(request.node.fspath):
        return
    from ai4data.data_use.models.model_manager import ModelManager

    monkeypatch.setattr(
        ModelManager, "load_classifier",
        lambda self, model_id=None: mock_classifier_pipeline,
    )


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
    """Mock GLiNER2 model that dispatches batch_extract by schema kind.

    The three-model swarm pipeline invokes ``batch_extract`` with distinct
    schema objects built via ``model.create_schema()``:

    1. **entity** schema (``.entities()``) → Call 1, entity-only pass for
       mention spans.
    2. **pass1** schema (``.entities()`` + ``.relations()``) → Call 1b,
       relation extraction pass.
    3. **fallback** schema (``.classification()``) → Call 2, usage +
       typology + purpose_action.

    To let ``batch_extract`` dispatch statelessly (critical for
    ``extract_batch`` which runs the pipeline multiple times), each
    ``create_schema()`` call returns a fresh MagicMock tagged with a
    ``_kind`` attribute by its builder side-effects.
    """
    mock_model = MagicMock()

    # Each create_schema() returns a fresh tagged mock so the three schema
    # builders produce distinct objects whose _kind tag survives.
    def _make_schema_builder():
        s = MagicMock()

        def tag_entities(*a, **kw):
            s._kind = "entity"
            return s

        def tag_relations(*a, **kw):
            s._kind = "pass1"
            return s

        def tag_classification(*a, **kw):
            s._kind = "fallback"
            return s

        s.entities.side_effect = tag_entities
        s.relations.side_effect = tag_relations
        s.classification.side_effect = tag_classification
        return s

    mock_model.create_schema.side_effect = _make_schema_builder

    # Stateless dispatcher keyed on schema._kind.
    def _batch_extract(texts, schema, **kwargs):
        kind = getattr(schema, "_kind", "entity")
        n = len(texts) if isinstance(texts, list) else 1
        if kind == "entity":
            # Entity-only schema (scratch benchmarks / variant A).
            out = [{"entities": {}} for _ in range(n)]
            if n >= 1:
                out[0] = {
                    "entities": {
                        "named_data": [
                            {
                                "text": "DHS",
                                "confidence": 0.95,
                                "start": 0,
                                "end": 3,
                            }
                        ]
                    }
                }
            return out
        elif kind == "pass1":
            # Combined entity + relation schema — Call 1 of the Variant B'
            # pipeline. One call returns mention spans AND relations.
            out = [{"entities": {}, "relation_extraction": {}} for _ in range(n)]
            if n >= 1:
                out[0] = {
                    "entities": {
                        "named_data": [
                            {
                                "text": "DHS",
                                "confidence": 0.95,
                                "start": 0,
                                "end": 3,
                            }
                        ]
                    },
                    "relation_extraction": {},
                }
            return out
        else:  # fallback
            return [
                {
                    "typology": {"label": "survey", "confidence": 0.95},
                    "usage": {"label": "primary", "confidence": 0.95},
                }
                for _ in range(n)
            ]

    mock_model.batch_extract.side_effect = _batch_extract

    # Legacy single-text extract/extract_json kept for any direct callers.
    mock_model.extract.return_value = {
        "entities": {
            "named_data": [{"text": "Test Dataset", "confidence": 0.95, "start": 0, "end": 12}]
        },
        "relation_extraction": {},
    }
    mock_model.extract_json.return_value = {"data_mention": []}

    # Allow adapter loading without errors.
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
    """Mock ModelManager to share a single base model across adapters.

    Production loads one GLiNER2 base model and loads the fine-tuned LoRA
    adapter on it via gliner2's native ``model.load_adapter(path)`` inside
    ``adapter_scope``. The mock mirrors this: ``load_base`` returns the same
    ``mock_gliner_model``, ``adapter_scope`` yields that model without actual
    adapter loading.
    """

    from contextlib import contextmanager

    import threading

    from ai4data.data_use.models.model_manager import ModelManager

    @contextmanager
    def mock_adapter_scope(self, model, adapter_name, adapter_id, model_id=None):
        yield model

    monkeypatch.setattr(ModelManager, "load_base", lambda self, model_id=None: mock_gliner_model)
    monkeypatch.setattr(ModelManager, "adapter_scope", mock_adapter_scope)
    monkeypatch.setattr(ModelManager, "load_classifier", lambda self, model_id=None: mock_classifier_pipeline)

    manager = ModelManager()
    manager._lock = threading.Lock()
    return manager
