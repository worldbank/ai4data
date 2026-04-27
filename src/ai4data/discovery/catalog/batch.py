"""
Batch jobs and Fire CLI for syncing metadata IDs and JSON from the catalog.

Prefer importing from here instead of the legacy ``ai4data.scraper.metadata`` shim.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from fire import Fire
from tqdm.auto import tqdm

from ..paths import get_metadata_ids_path
from .http import get_metadata_ids, get_metadata_json, search_metadata


def save_metadata_ids(metadata_ids: list, dtype: str) -> None:
    """
    Save the metadata IDs to a JSON file.

    Args:
        metadata_ids (list): A list of metadata ID dictionaries to save.
        dtype (str): The type of metadata (e.g., 'indicator', 'document').
    """
    if metadata_ids is None or len(metadata_ids) == 0:
        return

    fpath = get_metadata_ids_path(dtype)
    fpath.parent.mkdir(parents=True, exist_ok=True)

    with open(fpath, "w") as f:
        json.dump(metadata_ids, f, indent=2)


def scrape_all_ids(
    get_metadata_ids_func: Callable = get_metadata_ids,
    search_metadata_func: Callable = search_metadata,
    save_metadata_ids_func: Callable = save_metadata_ids,
    **kwargs,
):
    """
    Scrape all metadata IDs from the metadata catalog based on the provided parameters.

    Args:
        get_metadata_ids_func (Callable, optional): Function or task to fetch metadata IDs.
        search_metadata_func (Callable, optional): Function or task to search metadata.
        save_metadata_ids_func (Callable, optional): Function or task to save metadata IDs.
        **kwargs: Arbitrary keyword arguments representing search parameters.
    """
    params = kwargs

    assert "ps" in params, "The number of items per page is required"
    assert "type" in params, "The type of metadata is required, e.g., timeseries, document, geospatial, etc."

    if params["type"] == "indicator":
        params["type"] = "timeseries"
    if params["type"] == "microdata":
        params["type"] = "survey"

    dtype = params["type"]
    if dtype == "timeseries":
        dtype = "indicator"
    elif dtype == "survey":
        dtype = "microdata"

    metadata_ids = get_metadata_ids_func(params, search_metadata_func=search_metadata_func)
    print(f"Total metadata ids: {len(metadata_ids)}")

    save_metadata_ids_func(metadata_ids, dtype)


def scrape_all_metadata(
    get_metadata_json_func: Callable = get_metadata_json,
    get_metadata_ids_func: Callable = get_metadata_ids,
    search_metadata_func: Callable = search_metadata,
    save_metadata_ids_func: Callable = save_metadata_ids,
    **kwargs,
):
    """
    Scrape all metadata from the metadata catalog based on the provided parameters.

    Args:
        get_metadata_func (Callable, optional): Function or task to fetch metadata.
        get_metadata_ids_func (Callable, optional): Function or task to fetch metadata IDs.
        search_metadata_func (Callable, optional): Function or task to search metadata.
        save_metadata_ids_func (Callable, optional): Function or task to save metadata IDs.
        **kwargs: Arbitrary keyword arguments representing search parameters.
    """

    skip_errors = kwargs.get("skip_errors", False)

    scrape_all_ids(
        get_metadata_ids_func=get_metadata_ids_func,
        search_metadata_func=search_metadata_func,
        save_metadata_ids_func=save_metadata_ids_func,
        **kwargs,
    )

    dtype = kwargs.get("type", "timeseries")
    force = kwargs.get("force", False)

    if dtype == "indicator":
        dtype = "timeseries"

    fpath = get_metadata_ids_path(dtype)

    with open(fpath) as f:
        metadata_ids = json.load(f)

    print(f"Total metadata ids: {len(metadata_ids)}")

    for metadata_id in tqdm(metadata_ids):
        idno = metadata_id["idno"]
        metadata_type = metadata_id["type"]

        try:
            _ = get_metadata_json_func(idno=idno, metadata_type=metadata_type, force=force, load_exists=False)
        except Exception as e:
            print(f"Error fetching metadata for {idno}: {e}")

            if not skip_errors:
                raise e


def main(action: str, **kwargs):
    if action == "scrape_all_metadata":
        scrape_all_metadata(**kwargs)
    elif action == "scrape_all_ids":
        scrape_all_ids(**kwargs)
    else:
        raise ValueError(f"Invalid action: {action}")


if __name__ == "__main__":
    # uv run python -m ai4data.discovery.catalog.batch --action=scrape_all_ids type=indicator ps=100
    Fire(main)
