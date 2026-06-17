"""
Stable integration surface for **NADA** metadata used by downstream
packages (e.g. ``nada-opensearch``).

**In this package**

- :mod:`ai4data.discovery.catalog.http` — catalog HTTP API (fetch JSON, search, list ids).

- :mod:`ai4data.discovery.catalog.langdoc_id` — ``get_langdoc_uuid`` and UUID helper.

**Related (import from** :mod:`ai4data.discovery.metadata.handler` **):** ``MetadataLoader``, ``get_metadata_langdocs``. Those are not re-exported here to avoid a circular import with :mod:`ai4data.discovery.metadata.handler` (which uses :func:`ai4data.discovery.catalog.http.get_metadata_json`).

``MetadataLoader`` uses :func:`ai4data.discovery.catalog.http.get_metadata_json` for fetches.
"""

from __future__ import annotations

from . import extract as catalog_extract
from .http import get_ids_type, get_metadata_ids, get_metadata_json, is_extract_mode, search_metadata
from .langdoc_id import get_langdoc_uuid

__all__ = [
    "catalog_extract",
    "get_metadata_json",
    "get_metadata_ids",
    "get_ids_type",
    "search_metadata",
    "get_langdoc_uuid",
    "is_extract_mode",
]
