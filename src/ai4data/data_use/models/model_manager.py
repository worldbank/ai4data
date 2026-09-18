"""Model loading and caching for GLiNER2.

Loads the base GLiNER2 model **once** and loads the fine-tuned LoRA adapter on
it around each inference call via ``model.load_adapter(path)`` (gliner2's
native method). This avoids loading N full base-model copies (one per
adapter), cutting both startup time and memory ~4x for the three-model
swarm + classifier pipeline.
"""

import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch
from gliner2 import GLiNER2
from huggingface_hub import snapshot_download


class GLiNERClassifierWrapper:
    """Wrapper that runs GLiNER2 sequence classification under an adapter scope."""

    def __init__(self, model, scope_fn):
        self.model = model
        self._scope_fn = scope_fn
        self.tasks = {"has_data_mention": ["has_mention", "no_mention"]}

    def __call__(self, text: str):
        if not text.strip():
            return [{"label": "NO_DATA", "score": 1.0}]
        with self._scope_fn():
            res = self.model.classify_text(text, self.tasks, threshold=0.0, include_confidence=True)
        info = res.get("has_data_mention", {})
        label = info.get("label", "no_mention")
        score = info.get("confidence", 0.0)
        mapped_label = "WITH_DATA" if label == "has_mention" else "NO_DATA"
        return [{"label": mapped_label, "score": score}]


class ModelManager:
    """Manages GLiNER2 model loading and adapter activation.

    One base model per ``model_id`` is cached at the class level. Fine-tuned
    LoRA adapters are loaded on-demand into that shared base via gliner2's
    native ``model.load_adapter(path)``, which replaces the currently-active
    adapter weights in-place. All adapter loads go through ``adapter_scope``
    which holds a class-wide lock, so concurrent pages/extractors sharing the
    base model never race.
    """

    DEFAULT_MODEL_ID = "fastino/gliner2-large-v1"
    DEFAULT_ADAPTER_ID = "ai4data/datause-extraction"
    DEFAULT_CLASSIFIER_ID = "ai4data/datause-classifier"
    _base_cache: Dict[str, GLiNER2] = {}
    _adapter_path_cache: Dict[str, str] = {}
    _active_adapter: Dict[int, Optional[str]] = {}
    _classifier_cache: Dict[str, object] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        adapter_id: Optional[str] = DEFAULT_ADAPTER_ID,
    ):
        self.cache_dir = cache_dir
        self.adapter_id = adapter_id

    def _map_location(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_base(self, model_id: Optional[str] = None) -> GLiNER2:
        """Load (once) and cache the base GLiNER2 model without any adapter."""
        model_id = model_id or self.DEFAULT_MODEL_ID
        with self._lock:
            if model_id not in self._base_cache:
                kwargs = {"map_location": self._map_location()}
                if self.cache_dir:
                    kwargs["cache_dir"] = self.cache_dir
                try:
                    self._base_cache[model_id] = GLiNER2.from_pretrained(model_id, **kwargs)
                except Exception as exc:
                    raise RuntimeError(f"Failed to load model '{model_id}': {exc}") from exc
            return self._base_cache[model_id]

    def _adapter_path(self, adapter_id: str) -> str:
        """Download (once) and cache the adapter repo path."""
        if adapter_id not in self._adapter_path_cache:
            kwargs = {}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            self._adapter_path_cache[adapter_id] = snapshot_download(adapter_id, **kwargs)
        return self._adapter_path_cache[adapter_id]

    @contextmanager
    def adapter_scope(
        self,
        model: GLiNER2,
        adapter_name: str,
        adapter_id: Optional[str],
        model_id: Optional[str] = None,
    ) -> Iterator[GLiNER2]:
        """Context manager that ensures ``adapter_id`` is active on ``model``.

        Acquires the class-wide lock, downloads the adapter (cached), and calls
        gliner2's native ``model.load_adapter(path)`` to swap the active LoRA
        weights in-place. If the requested adapter is already the active one,
        the load is skipped.

        Inference MUST happen inside the ``with`` block so no other thread can
        reload a different adapter mid-call.

        Args:
            model: The shared base model.
            adapter_name: Logical name for this adapter scope (unused; kept
                          for interface symmetry).
            adapter_id: HuggingFace adapter repo ID. Empty/None is a no-op
                        (yields the model as-is).
            model_id: Unused; kept for interface symmetry.
        """
        with self._lock:
            if adapter_id:
                model_key = id(model)
                if self._active_adapter.get(model_key) != adapter_id:
                    path = self._adapter_path(adapter_id)
                    model.load_adapter(path)
                    self._active_adapter[model_key] = adapter_id
            yield model

    def _scope_factory(
        self,
        model: GLiNER2,
        adapter_name: str,
        adapter_id: str,
    ):
        """Return a zero-arg callable that provides a fresh adapter_scope."""
        return self.adapter_scope(model, adapter_name, adapter_id)

    def load(
        self,
        model_id: Optional[str] = None,
        adapter_id: Optional[str] = None,
    ) -> GLiNER2:
        """Load a GLiNER2 model with an optional adapter, with caching.

        Backward-compatible convenience: returns the shared cached base model
        with ``adapter_id`` loaded (replacing any previously loaded adapter).

        Args:
            model_id: HuggingFace model ID or path to local model.
            adapter_id: HuggingFace adapter repo ID. If None, falls back to
                        the manager's default ``adapter_id``.
        Returns:
            Loaded GLiNER2 model
        """
        model_id = model_id or self.DEFAULT_MODEL_ID
        resolved_adapter = adapter_id if adapter_id is not None else self.adapter_id
        model = self.load_base(model_id)
        if resolved_adapter:
            with self.adapter_scope(model, "default", resolved_adapter, model_id):
                pass
        return model

    def load_classifier(self, model_id: Optional[str] = None):
        """Load the page-relevance classifier, with caching.

        For ``datause-classifier`` adapters, uses the shared base model with
        the classifier adapter loaded before each ``classify_text`` call.

        Falls back to a HuggingFace text-classification pipeline for other
        model IDs.
        """
        model_id = model_id or self.DEFAULT_CLASSIFIER_ID

        if "datause-classifier" in model_id:
            cache_key = ("classifier", model_id)
            if cache_key in self._classifier_cache:
                return self._classifier_cache[cache_key]
            base = self.load_base()
            def _clf_scope():
                return self.adapter_scope(base, "classifier", model_id)
            clf = GLiNERClassifierWrapper(base, _clf_scope)
            self._classifier_cache[cache_key] = clf
            return clf

        cache_key = ("hf_classifier", model_id)
        if cache_key in self._classifier_cache:
            return self._classifier_cache[cache_key]

        from transformers import AutoTokenizer
        from transformers import pipeline as hf_pipeline

        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1
        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        clf = hf_pipeline(
            "text-classification",
            model=model_id,
            tokenizer=tokenizer,
            device=device,
            truncation=True,
            max_length=512,
            model_kwargs=kwargs,
        )
        self._classifier_cache[cache_key] = clf
        return clf

    def clear_cache(self):
        """Clear the model and adapter caches."""
        with self._lock:
            self._base_cache.clear()
            self._adapter_path_cache.clear()
            self._active_adapter.clear()
            self._classifier_cache.clear()
