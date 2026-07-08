"""Model loading and caching for GLiNER2."""

from typing import Optional

import torch
from gliner2 import GLiNER2
from huggingface_hub import snapshot_download


class GLiNERClassifierWrapper:
    """Wrapper to make GLiNER2-based sequence classification act like a HF pipeline."""

    def __init__(self, model):
        self.model = model
        self.schema = model.create_schema()
        self.schema.classification("data_use", ["WITH_DATA", "NO_DATA"], multi_label=False)

    def __call__(self, text: str):
        if not text.strip():
            return [{"label": "NO_DATA", "score": 1.0}]
        res = self.model.batch_extract([text], self.schema, include_confidence=True)[0]
        info = res.get("data_use", {})
        return [{"label": info.get("label", "NO_DATA"), "score": info.get("confidence", 0.0)}]


class ModelManager:
    """Manages GLiNER2 model loading and caching."""

    DEFAULT_MODEL_ID = "fastino/gliner2-large-v1"
    DEFAULT_ADAPTER_ID = "ai4data/datause-extraction"
    DEFAULT_CLASSIFIER_ID = "rafmacalaba/datause-classifier-v1"
    _model_cache = {}

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        adapter_id: Optional[str] = DEFAULT_ADAPTER_ID,
    ):
        """Initialize model manager.

        Args:
            cache_dir: Directory to cache models. If None, uses default HuggingFace cache.
            adapter_id: HuggingFace adapter repo ID to load after the base model.
                       Defaults to the datause fine-tuned adapter. Set to None to skip.
        """
        self.cache_dir = cache_dir
        self.adapter_id = adapter_id

    def load(
        self,
        model_id: Optional[str] = None,
        adapter_id: Optional[str] = None,
    ) -> GLiNER2:
        """Load a GLiNER2 model with an optional adapter, with caching.

        Args:
            model_id: HuggingFace model ID or path to local model.
                     If None, uses default model.
            adapter_id: HuggingFace adapter repo ID to apply after loading the base model.
                       If not provided, falls back to the adapter_id set on __init__
                       (DEFAULT_ADAPTER_ID by default). Pass an empty string or explicitly
                       pass ``None`` via ``load(adapter_id=None)`` to skip adapter loading.

        Returns:
            Loaded GLiNER2 model (with adapter applied if specified)

        Raises:
            RuntimeError: If model or adapter loading fails
        """
        model_id = model_id or self.DEFAULT_MODEL_ID
        # Resolve adapter: explicit argument wins; fall back to instance default
        resolved_adapter = adapter_id if adapter_id is not None else self.adapter_id

        # Use (model_id, adapter_id) as cache key so different adapters are cached separately
        cache_key = (model_id, resolved_adapter)

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        try:
            kwargs = {}
            if torch.cuda.is_available():
                kwargs["map_location"] = "cuda"
            elif torch.backends.mps.is_available():
                kwargs["map_location"] = "mps"
            else:
                kwargs["map_location"] = "cpu"

            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir

            # Load the base model
            model = GLiNER2.from_pretrained(model_id, **kwargs)

            # Apply the adapter if specified
            if resolved_adapter:
                download_kwargs = {}
                if self.cache_dir:
                    download_kwargs["cache_dir"] = self.cache_dir
                adapter_path = snapshot_download(resolved_adapter, **download_kwargs)
                model.load_adapter(adapter_path)

            self._model_cache[cache_key] = model
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_id}': {str(e)}") from e

    def load_classifier(self, model_id: Optional[str] = None):
        """Load the BERT page-relevance classifier, with caching.

        Returns a HuggingFace text-classification pipeline that predicts
        ``WITH_DATA`` (page likely contains a dataset mention) or ``NO_DATA``.
        Token truncation at 512 is done by the pipeline's tokeniser so the
        full token budget is used rather than a character approximation.

        Args:
            model_id: HuggingFace model ID. If None, uses DEFAULT_CLASSIFIER_ID.

        Returns:
            Loaded HuggingFace pipeline instance

        Raises:
            RuntimeError: If model loading fails
        """
        model_id = model_id or self.DEFAULT_CLASSIFIER_ID
        cache_key = ("classifier", model_id)

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        try:
            # If the classifier is a GLiNER2-based LoRA adapter
            if "datause-classifier" in model_id:
                model = self.load(model_id="fastino/gliner2-large-v1", adapter_id=model_id)
                clf = GLiNERClassifierWrapper(model)
                self._model_cache[cache_key] = clf
                return clf

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
            self._model_cache[cache_key] = clf
            return clf

        except Exception as e:
            raise RuntimeError(f"Failed to load classifier '{model_id}': {str(e)}") from e

    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
