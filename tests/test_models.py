"""Tests for ModelManager."""

import pytest

from ai4data.data_use.models.model_manager import ModelManager


class TestModelManager:
    """Test suite for ModelManager class."""

    def setup_method(self):
        """Clear the class-level model cache before each test."""
        ModelManager._model_cache.clear()

    def test_initialization_default(self):
        """Test manager initialization with default parameters."""
        manager = ModelManager()
        assert manager.cache_dir is None
        assert manager.adapter_id == ModelManager.DEFAULT_ADAPTER_ID
        assert manager._model_cache == {}

    def test_initialization_with_cache_dir(self):
        """Test manager initialization with custom cache directory."""
        manager = ModelManager(cache_dir="./test_cache")
        assert manager.cache_dir == "./test_cache"

    def test_initialization_no_adapter(self):
        """Test manager initialization with adapter disabled."""
        manager = ModelManager(adapter_id=None)
        assert manager.adapter_id is None

    def test_default_model_id(self):
        """Test that default model ID is set."""
        assert ModelManager.DEFAULT_MODEL_ID == "fastino/gliner2-large-v1"

    def test_default_adapter_id(self):
        """Test that default adapter ID is set."""
        assert ModelManager.DEFAULT_ADAPTER_ID == "ai4data/datause-extraction"

    def test_load_with_adapter(self, monkeypatch, mock_gliner_model):
        """Test that snapshot_download and load_adapter are called when adapter_id is set."""
        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", lambda model_id, **kw: mock_gliner_model)

        fake_adapter_path = "/tmp/fake_adapter"
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id: fake_adapter_path,
        )

        manager = ModelManager(adapter_id="rafmacalaba/gliner2-datause-v1")
        model = manager.load("fastino/gliner2-base-v1")

        mock_gliner_model.load_adapter.assert_called_once_with(fake_adapter_path)
        assert model is mock_gliner_model

    def test_load_without_adapter(self, monkeypatch, mock_gliner_model):
        """Test that load_adapter is NOT called when adapter_id is None."""
        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", lambda model_id, **kw: mock_gliner_model)

        snapshot_calls = []
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id: snapshot_calls.append(repo_id),
        )

        manager = ModelManager(adapter_id=None)
        manager.load("fastino/gliner2-base-v1")

        assert (
            snapshot_calls == []
        ), "snapshot_download should not be called when adapter_id is None"
        mock_gliner_model.load_adapter.assert_not_called()

    def test_model_caching(self, monkeypatch, mock_gliner_model):
        """Test that models are cached after first load."""
        load_count = {"count": 0}

        def mock_from_pretrained(model_id, **kwargs):
            load_count["count"] += 1
            return mock_gliner_model

        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", mock_from_pretrained)
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id: "/tmp/fake_adapter",
        )

        manager = ModelManager()

        model1 = manager.load("test-model")
        assert load_count["count"] == 1

        model2 = manager.load("test-model")
        assert load_count["count"] == 1  # Should not increment
        assert model1 is model2

    def test_cache_key_includes_adapter(self, monkeypatch, mock_gliner_model):
        """Test that same base model with different adapters creates separate cache entries."""
        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", lambda model_id, **kw: mock_gliner_model)
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id: "/tmp/fake_adapter",
        )

        manager = ModelManager(adapter_id=None)  # start with no default adapter

        manager.load("fastino/gliner2-base-v1", adapter_id="adapter-a")
        manager.load("fastino/gliner2-base-v1", adapter_id="adapter-b")

        assert len(manager._model_cache) == 2
        assert ("fastino/gliner2-base-v1", "adapter-a") in manager._model_cache
        assert ("fastino/gliner2-base-v1", "adapter-b") in manager._model_cache

    def test_different_models_cached_separately(self, monkeypatch, mock_gliner_model):
        """Test that different models are cached separately."""

        def mock_from_pretrained(model_id, **kwargs):
            mock = MagicMock()
            mock.model_id = model_id
            mock.load_adapter = MagicMock()
            return mock

        from unittest.mock import MagicMock

        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", mock_from_pretrained)
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id: "/tmp/fake_adapter",
        )

        manager = ModelManager()

        model1 = manager.load("model-1")
        model2 = manager.load("model-2")

        assert model1 is not model2
        assert len(manager._model_cache) == 2

    def test_clear_cache(self, monkeypatch, mock_gliner_model):
        """Test cache clearing."""

        def mock_from_pretrained(model_id, **kwargs):
            return mock_gliner_model

        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", mock_from_pretrained)
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id: "/tmp/fake_adapter",
        )

        manager = ModelManager()
        manager.load("test-model")

        assert len(manager._model_cache) == 1

        manager.clear_cache()
        assert len(manager._model_cache) == 0

    def test_load_with_none_uses_default(self, monkeypatch, mock_gliner_model):
        """Test that load(None) uses default model ID."""
        loaded_model_id = {"id": None}

        def mock_from_pretrained(model_id, **kwargs):
            loaded_model_id["id"] = model_id
            return mock_gliner_model

        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", mock_from_pretrained)
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id: "/tmp/fake_adapter",
        )

        manager = ModelManager()
        manager.load(None)

        assert loaded_model_id["id"] == ModelManager.DEFAULT_MODEL_ID

    def test_load_error_handling(self, monkeypatch):
        """Test error handling when model loading fails."""

        def mock_from_pretrained(model_id, **kwargs):
            raise Exception("Model not found")

        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", mock_from_pretrained)

        manager = ModelManager()

        with pytest.raises(RuntimeError, match="Failed to load model"):
            manager.load("invalid-model")

    def test_load_classifier(self, monkeypatch):
        """Test load_classifier handles cache_dir properly."""
        clf_calls = []
        tokenizer_calls = []

        class MockTokenizer:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                tokenizer_calls.append((model_id, kwargs))
                return "mock_tokenizer"

        def mock_pipeline(task, model, tokenizer, device, truncation, max_length, model_kwargs):
            clf_calls.append((task, model, tokenizer, device, truncation, max_length, model_kwargs))
            return "mock_pipeline"

        monkeypatch.setattr("transformers.AutoTokenizer", MockTokenizer)
        monkeypatch.setattr("transformers.pipeline", mock_pipeline)

        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        # Test with custom cache dir
        manager = ModelManager(cache_dir="./custom_cache")
        clf = manager.load_classifier()

        assert clf == "mock_pipeline"
        assert len(tokenizer_calls) == 1
        assert tokenizer_calls[0] == (
            ModelManager.DEFAULT_CLASSIFIER_ID,
            {"cache_dir": "./custom_cache"},
        )
        assert len(clf_calls) == 1
        assert clf_calls[0] == (
            "text-classification",
            ModelManager.DEFAULT_CLASSIFIER_ID,
            "mock_tokenizer",
            -1,
            True,
            512,
            {"cache_dir": "./custom_cache"},
        )
