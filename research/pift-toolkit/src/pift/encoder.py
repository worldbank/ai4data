"""
Encoder wrapper: load a SentenceTransformer and encode with the right prefixes.

Some encoder families (E5, BGE, GTE) expect a query/document prefix. The prefix
is part of the model's contract, so it is carried in the config (for a base
model) or in the fine-tuned model's ``pift_config.json`` (written at the end of
training). This wrapper applies the correct prefix automatically, so callers
never have to remember it, and embeddings are L2-normalized so a dot product is
cosine similarity.

A fine-tuned model directory is detected by the presence of ``pift_config.json``;
its recorded prefixes override anything passed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder:
    def __init__(self, model_id_or_path: str, query_prefix: str = "",
                 doc_prefix: str = "", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        # A fine-tuned directory records its own prefixes; trust those.
        cfg_path = Path(model_id_or_path) / "pift_config.json"
        if cfg_path.exists():
            saved = json.loads(cfg_path.read_text())
            query_prefix = saved.get("query_prefix") or ""
            doc_prefix = saved.get("doc_prefix") or ""

        self.device = pick_device(device)
        self.model = SentenceTransformer(model_id_or_path, device=self.device)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def _encode(self, texts, prefix, batch_size, show):
        texts = [prefix + t for t in texts]
        emb = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=show,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_documents(self, texts, batch_size: int = 64, show: bool = False):
        return self._encode(texts, self.doc_prefix, batch_size, show)

    def encode_queries(self, texts, batch_size: int = 64, show: bool = False):
        return self._encode(texts, self.query_prefix, batch_size, show)


def resolve_model(model_id_or_path: str | None, config):
    """Return (model_id, query_prefix, doc_prefix), defaulting to the config base."""
    if model_id_or_path is None:
        bm = config.base_model
        return bm["hf_id"], bm["query_prefix"], bm["doc_prefix"]
    return model_id_or_path, config.base_model["query_prefix"], config.base_model["doc_prefix"]
