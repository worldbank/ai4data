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

from .http import get_ids_type, get_metadata_ids, get_metadata_json, search_metadata
from .langdoc_id import get_langdoc_uuid

__all__ = [
    "get_metadata_json",
    "get_metadata_ids",
    "get_ids_type",
    "search_metadata",
    "get_langdoc_uuid",
]
