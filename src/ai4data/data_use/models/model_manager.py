"""Model loading and caching for GLiNER2."""

from typing import Optional

import torch
from gliner2 import GLiNER2


class ModelManager:
    """Manages GLiNER2 model loading and caching."""

    # DEFAULT_MODEL_ID = "fastino/gliner2-large-v1"
    # DEFAULT_MODEL_ID = "rafmacalaba/datause-extraction-v4-finetuned"
    # DEFAULT_MODEL_ID = "rafmacalaba/datause-extraction-synthetic-finetuned"
    # DEFAULT_MODEL_ID = "rafmacalaba/datause-extraction-v6-finetuned"
    DEFAULT_MODEL_ID = "rafmacalaba/datause-extraction-v6-judge-finetuned"

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize model manager.

        Args:
            cache_dir: Directory to cache models. If None, uses default HuggingFace cache.
        """
        self.cache_dir = cache_dir
        self._model_cache = {}

    def load(self, model_id: Optional[str] = None) -> GLiNER2:
        """Load a GLiNER2 model with caching.

        Args:
            model_id: HuggingFace model ID or path to local model.
                     If None, uses default model.

        Returns:
            Loaded GLiNER2 model

        Raises:
            RuntimeError: If model loading fails
        """
        model_id = model_id or self.DEFAULT_MODEL_ID

        # Return cached model if available
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        try:
            # Load model
            kwargs = {}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir

            model = GLiNER2.from_pretrained(model_id, **kwargs)

            # Cache the model
            self._model_cache[model_id] = model

            # Move model to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_id}': {str(e)}") from e

    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
