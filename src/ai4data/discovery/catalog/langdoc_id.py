"""
Stable document id for LangChain langdocs (used when indexing chunks).

Copied from ``ai4data.discovery.metadata.handler`` / ``ai4data.discovery.metadata.utils`` for the
``ai4data.discovery.catalog`` surface.
"""

from __future__ import annotations

import hashlib
import uuid

from langchain_core.documents import Document as LangchainDocument

from ..metadata.utils import get_idno


def create_uuid_from_string(val: str) -> uuid.UUID:
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def get_langdoc_uuid(doc: LangchainDocument) -> str:
    """Deterministic id from type, idno, qfield, and page content."""
    meta = doc.metadata
    metadata_type = meta["type"]
    metadata_idno = meta.get("idno") or get_idno(meta, metadata_type)
    metadata_qfield = meta.get("qfield", "")

    key = f"{metadata_type}__{metadata_idno}__{metadata_qfield}__{doc.page_content}"

    return str(create_uuid_from_string(key))
