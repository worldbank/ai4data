"""Tests for ModelManager."""

from unittest.mock import MagicMock

import pytest

from ai4data.data_use.models.model_manager import ModelManager


class TestModelManager:
    """Test suite for ModelManager class."""

    def setup_method(self):
        """Clear the class-level caches before each test."""
        ModelManager._base_cache.clear()
        ModelManager._adapter_path_cache.clear()
        ModelManager._active_adapter.clear()
        ModelManager._classifier_cache.clear()

    def test_initialization_default(self):
        """Test manager initialization with default parameters."""
        manager = ModelManager()
        assert manager.cache_dir is None
        assert manager.adapter_id == ModelManager.DEFAULT_ADAPTER_ID
        assert manager._base_cache == {}

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
        """Test that load_adapter is called when adapter_id is set."""
        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", lambda model_id, **kw: mock_gliner_model)
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id, **kw: "/tmp/fake_adapter",
        )

        manager = ModelManager(adapter_id="rafmacalaba/gliner2-datause-v1")
        model = manager.load("fastino/gliner2-base-v1")

        assert model is mock_gliner_model
        mock_gliner_model.load_adapter.assert_called_once_with("/tmp/fake_adapter")

    def test_load_without_adapter(self, monkeypatch, mock_gliner_model):
        """Test that load_adapter is NOT called when adapter_id is None."""
        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", lambda model_id, **kw: mock_gliner_model)
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id: [],
        )

        manager = ModelManager(adapter_id=None)
        manager.load("fastino/gliner2-base-v1")

        mock_gliner_model.load_adapter.assert_not_called()

    def test_model_caching(self, monkeypatch, mock_gliner_model):
        """Test that the base model is loaded only once."""
        load_count = {"count": 0}

        def mock_from_pretrained(model_id, **kwargs):
            load_count["count"] += 1
            return mock_gliner_model

        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", mock_from_pretrained)

        manager = ModelManager()

        model1 = manager.load("test-model")
        assert load_count["count"] == 1

        model2 = manager.load("test-model")
        assert load_count["count"] == 1
        assert model1 is model2

    def test_adapter_scope_idempotent(self, monkeypatch, mock_gliner_model):
        """Test that adapter_scope skips load_adapter if same adapter already active."""
        def fake_download(repo, **kw):
            return f"/tmp/{repo.split('/')[-1]}"

        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download", fake_download
        )

        manager = ModelManager()
        model = mock_gliner_model

        with manager.adapter_scope(model, "entity", "adapter-a"):
            pass
        assert mock_gliner_model.load_adapter.call_count == 1

        with manager.adapter_scope(model, "entity", "adapter-a"):
            pass
        assert mock_gliner_model.load_adapter.call_count == 1

    def test_adapter_scope_switches_on_different_adapter(
        self, monkeypatch, mock_gliner_model
    ):
        """Test that adapter_scope reloads when a different adapter is requested."""
        calls = []

        def fake_download(repo, **kw):
            path = f"/tmp/{repo.split('/')[-1]}"
            calls.append(repo)
            return path

        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download", fake_download
        )

        manager = ModelManager()
        model = mock_gliner_model

        with manager.adapter_scope(model, "entity", "adapter-a"):
            pass
        with manager.adapter_scope(model, "relation", "adapter-b"):
            pass
        with manager.adapter_scope(model, "impact", "adapter-c"):
            pass

        assert mock_gliner_model.load_adapter.call_count == 3
        # Verify each adapter was downloaded
        assert "adapter-a" in calls[0]
        assert "adapter-b" in calls[1]
        assert "adapter-c" in calls[2]

    def test_adapter_scope_empty_adapter_is_noop(self, monkeypatch, mock_gliner_model):
        """Test that adapter_scope with no adapter_id does nothing."""
        manager = ModelManager()
        model = mock_gliner_model

        with manager.adapter_scope(model, "none", None):
            passthrough = model
        assert passthrough is model
        mock_gliner_model.load_adapter.assert_not_called()

    def test_one_base_shared_across_adapters(self, monkeypatch, mock_gliner_model):
        """Test that the same base model serves multiple adapters."""
        load_count = {"count": 0}

        def mock_from_pretrained(model_id, **kwargs):
            load_count["count"] += 1
            return mock_gliner_model

        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", mock_from_pretrained)
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id, **kw: f"/tmp/{repo_id.split('/')[-1]}",
        )

        manager = ModelManager()
        model_a = manager.load("fastino/gliner2-base-v1", adapter_id="adapter-a")
        model_b = manager.load("fastino/gliner2-base-v1", adapter_id="adapter-b")

        assert load_count["count"] == 1
        assert model_a is model_b
        assert len(manager._base_cache) == 1

    def test_different_models_cached_separately(self, monkeypatch):
        """Test that different base models are cached separately."""

        def mock_from_pretrained(model_id, **kwargs):
            mock = MagicMock()
            mock.model_id = model_id
            return mock

        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", mock_from_pretrained)

        manager = ModelManager()

        model1 = manager.load("model-1")
        model2 = manager.load("model-2")

        assert model1 is not model2
        assert len(manager._base_cache) == 2

    def test_clear_cache(self, monkeypatch, mock_gliner_model):
        """Test clearing model caches."""
        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", lambda model_id, **kw: mock_gliner_model)

        manager = ModelManager(adapter_id=None)
        manager.load()
        assert len(ModelManager._base_cache) == 1
        manager.clear_cache()
        assert len(ModelManager._base_cache) == 0

    def test_load_with_none_uses_default(self, monkeypatch, mock_gliner_model):
        """Test that load(model_id=None) falls back to default model ID."""
        loaded_model_id = []

        def mock_from_pretrained(model_id, **kwargs):
            loaded_model_id.append(model_id)
            return mock_gliner_model

        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", mock_from_pretrained)

        manager = ModelManager(adapter_id=None)
        manager.load(model_id=None)
        assert loaded_model_id[0] == ModelManager.DEFAULT_MODEL_ID

    def test_load_error_handling(self, monkeypatch):
        """Test error handling when from_pretrained raises exception."""
        from gliner2 import GLiNER2

        def mock_from_pretrained(model_id, **kwargs):
            raise ValueError("Invalid model")

        monkeypatch.setattr(GLiNER2, "from_pretrained", mock_from_pretrained)

        manager = ModelManager()

        with pytest.raises(RuntimeError, match="Failed to load model"):
            manager.load("invalid-model")

    def test_load_classifier(self, monkeypatch, mock_gliner_model):
        """Test load_classifier resolves correct wrapper/pipeline instances."""
        # 1. Test default GLiNER-based classifier.
        monkeypatch.setattr(
            "ai4data.data_use.models.model_manager.snapshot_download",
            lambda repo_id, **kw: "/tmp/fake_adapter",
        )
        from gliner2 import GLiNER2

        monkeypatch.setattr(GLiNER2, "from_pretrained", lambda model_id, **kw: mock_gliner_model)

        manager = ModelManager()
        clf = manager.load_classifier()

        from ai4data.data_use.models.model_manager import GLiNERClassifierWrapper

        assert isinstance(clf, GLiNERClassifierWrapper)
        assert clf.model is mock_gliner_model
        assert callable(clf._scope_fn)
        assert len(manager._base_cache) == 1

        # 2. Test fallback transformers classifier (when custom model_id is used)
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

        manager_custom = ModelManager(cache_dir="./custom_cache")
        clf_custom = manager_custom.load_classifier(model_id="custom-bert-model")

        assert clf_custom == "mock_pipeline"
        assert len(tokenizer_calls) == 1
        assert tokenizer_calls[0] == ("custom-bert-model", {"cache_dir": "./custom_cache"})
        assert len(clf_calls) == 1
        assert clf_calls[0] == (
            "text-classification",
            "custom-bert-model",
            "mock_tokenizer",
            -1,
            True,
            512,
            {"cache_dir": "./custom_cache"},
        )
