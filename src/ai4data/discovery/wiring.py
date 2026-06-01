"""Wire default embedding model, token splitter, and clustering into discovery processors."""

from __future__ import annotations

_registered = False


def register_discovery_processors() -> None:
    """Idempotent: register defaults for :func:`~ai4data.discovery.processors.document.embed_documents` / ``get_doc_reps``."""
    global _registered
    if _registered:
        return

    from .embeddings import default_embedding_factory, default_text_splitter_factory
    from .processors.document import configure_discovery_processors
    from .semantic_cluster_embedding import semantic_cluster_embedding

    configure_discovery_processors(
        embedding_factory=default_embedding_factory,
        text_splitter_factory=default_text_splitter_factory,
        cluster_fn=semantic_cluster_embedding,
    )
    _registered = True
