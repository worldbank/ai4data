"""Sliding-window HuggingFace embeddings and token splitters for discovery processors.

Uses ``lru_cache`` for process-wide reuse instead of singleton classes. Call
:func:`ai4data.discovery.wiring.register_discovery_processors` before
:func:`ai4data.discovery.processors.document.embed_documents` /
:func:`ai4data.discovery.processors.document.get_doc_reps`.
"""

from __future__ import annotations

import gc
from functools import lru_cache

import numpy as np
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

from .config import embedding_inference


def get_device(preferred_device: str | None) -> str:
    """
    Resolve device for embedding inference: explicit preference, else cuda, mps, or cpu.

    Imports ``torch`` lazily so importing discovery without embedding stays lighter.
    """
    if preferred_device:
        return preferred_device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@lru_cache(maxsize=1)
def _cached_huggingface_embeddings(
    model_name: str,
    batch_size: int,
    device_resolved: str,
    show_progress: bool,
) -> HuggingFaceEmbeddings:
    model_kwargs = {"device": device_resolved}
    encode_kwargs = {"normalize_embeddings": True, "batch_size": batch_size}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=show_progress,
    )


def default_embedding_factory() -> HuggingFaceEmbeddings:
    """Build (cached) HuggingFaceEmbeddings from :data:`embedding_inference` settings."""
    ei = embedding_inference
    device = get_device(ei.device)
    return _cached_huggingface_embeddings(ei.model, ei.batch_size, device, ei.show_progress)


@lru_cache(maxsize=1)
def _cached_sentence_transformers_splitter(model_name: str, chunk_overlap: int) -> SentenceTransformersTokenTextSplitter:
    return SentenceTransformersTokenTextSplitter(model_name=model_name, chunk_overlap=chunk_overlap)


def default_text_splitter_factory() -> SentenceTransformersTokenTextSplitter:
    """Token-bounded splitter aligned with the embedding model name; ``chunk_overlap`` defaults to 0."""
    return _cached_sentence_transformers_splitter(embedding_inference.model, 0)


def clear_embedding_caches() -> None:
    """Drop cached embedder/splitter and release GPU memory when CUDA is available."""
    _cached_huggingface_embeddings.cache_clear()
    _cached_sentence_transformers_splitter.cache_clear()

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


def get_collection_name(embedding_model: HuggingFaceEmbeddings, metadata_type: str | None = None) -> str:
    """Build a stable collection name from model id and optional metadata type suffix."""
    collection_name = embedding_model.model_name.replace("/", "__")

    if metadata_type:
        collection_name = f"{collection_name}__{metadata_type}"

    return collection_name


def get_embeddings(
    docs: list[str],
    embedding_model: HuggingFaceEmbeddings,
    *,
    as_array: bool = True,
) -> np.ndarray | list:
    """Embed strings; return a NumPy array when ``as_array`` is True."""
    embeddings = embedding_model.embed_documents(docs)

    if as_array:
        return np.array(embeddings)

    return embeddings
