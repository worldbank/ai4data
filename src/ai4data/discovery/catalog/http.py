"""
HTTP client for the NADA metadata catalog (search + JSON by idno).

Canonical implementation; ``ai4data.scraper.metadata`` re-exports batch helpers from
:mod:`ai4data.discovery.catalog.batch` and these functions for backward compatibility.
"""

from __future__ import annotations

import json

import httpx
from tqdm.auto import tqdm

from ..auth import get_catalog_auth_headers, get_catalog_cookies
from ..config import METADATA_CATALOG_URL
from ..paths import get_metadata_cache_path


def get_metadata_json(
    idno: str,
    metadata_type: str = None,
    force: bool = False,
    load_exists: bool = True,
    include_resources: bool = False,
) -> dict:
    """
    Retrieve metadata from a cache or fetch it from the catalog if not cached or if forced.

    Args:
        idno (str): The ID number of the metadata to retrieve.
        metadata_type (str, optional): Used for caching. If None, caching is bypassed.
        force (bool, optional): If True, forces a fresh retrieval.
        load_exists (bool, optional): If True, load from cache when present.
        include_resources (bool, optional): If True, include the resources in the metadata. Defaults to False.
    Returns:
        dict: The metadata associated with the given ID number.
    """
    cache_path = (
        get_metadata_cache_path(
            idno, metadata_type, include_resources=include_resources
        )
        if metadata_type
        else None
    )
    if cache_path and not force and cache_path.exists():
        try:
            if load_exists:
                with cache_path.open("r") as cache_file:
                    return json.load(cache_file)
            else:
                return None
        except OSError as e:
            print(f"Failed to read cache file {cache_path}: {e}")

    try:
        # Add query parameters to the request
        params = {
            "include_resources": include_resources,
        }
        response = httpx.get(
            f"{METADATA_CATALOG_URL}/api/catalog/json/{idno}",
            params=params,
            headers=get_catalog_auth_headers(),
            cookies=get_catalog_cookies(),
        )
        response.raise_for_status()
        metadata: dict = response.json()

        if metadata.get("type", None) == "timeseries":
            metadata["type"] = "indicator"
        if metadata.get("type", None) == "survey":
            metadata["type"] = "microdata"

        if metadata_type:
            if metadata_type and metadata.get("type", None) != metadata_type:
                raise ValueError(
                    f"The metadata type {metadata.get('type')} does not match the requested type: {metadata_type}"
                )

    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Failed to fetch metadata for ID {idno}: {e}") from e

    if cache_path:
        try:
            with cache_path.open("w") as cache_file:
                json.dump(metadata, cache_file, indent=2)
        except OSError as e:
            print(f"Failed to write cache file {cache_path}: {e}")

    return metadata


def search_metadata(params: dict = None) -> dict:
    """Search metadata in the metadata catalog using provided search parameters."""
    response = httpx.get(
        f"{METADATA_CATALOG_URL}/api/catalog/search",
        params=params,
        headers=get_catalog_auth_headers(),
        cookies=get_catalog_cookies(),
    )
    response.raise_for_status()
    data = response.json().get("result", {})
    return data


def get_ids_type(result: dict = None) -> dict:
    """Extract id, idno, and type from a single catalog search row."""
    if result is None:
        return dict(id=None, idno=None, type=None)

    metadata_type = result.get("type", None)
    if metadata_type == "timeseries":
        metadata_type = "indicator"
    elif metadata_type == "survey":
        metadata_type = "microdata"

    return dict(
        id=result.get("id", None), idno=result.get("idno", None), type=metadata_type
    )


def get_metadata_ids(
    params: dict = None,
    search_metadata_func=search_metadata,
    max_items: int | None = None,
) -> list:
    """
    Retrieve metadata IDs from the metadata catalog based on search parameters.

    Paginates through every page unless ``max_items`` is set, in which case pagination
    stops once that many rows have been collected (fewer HTTP calls for smoke tests).
    """
    default_params = dict(
        sk="", ps=100, type="timeseries", sort_by="year", sort_order="asc"
    )
    params = {**default_params, **(params or {})}

    if max_items is not None and max_items <= 0:
        return []

    all_metadata_ids: list = []

    params["page"] = 1
    num_per_page = int(params["ps"])

    data = search_metadata_func(params)
    all_metadata_ids.extend([get_ids_type(row) for row in data.get("rows", [])])
    if max_items is not None and len(all_metadata_ids) >= max_items:
        return all_metadata_ids[:max_items]

    all_pages = int(data.get("found", 0)) // num_per_page + 1

    for page in tqdm(range(2, all_pages + 1)):
        params["page"] = page
        data = search_metadata_func(params)
        all_metadata_ids.extend([get_ids_type(row) for row in data.get("rows", [])])
        if max_items is not None and len(all_metadata_ids) >= max_items:
            return all_metadata_ids[:max_items]

    return all_metadata_ids if max_items is None else all_metadata_ids[:max_items]
