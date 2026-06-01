"""
Authenticated GET helper for NADA / IHSN JSON APIs (``x-api-key``).

Separate from :mod:`ai4data.discovery.catalog.http` (public catalog JSON + search).
"""

from __future__ import annotations

import requests

from ..config import metadata_catalog


def get_resource(url: str, verify: bool = True, timeout: float = 2.0):
    headers = {
        "x-api-key": metadata_catalog.x_api_key,
    }
    response = requests.request("get", url, verify=verify, timeout=timeout, headers=headers)
    response.raise_for_status()

    return response.json()["data"]
