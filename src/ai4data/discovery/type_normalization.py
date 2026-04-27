"""Catalog metadata type strings used in filenames (aligned with NADA API aliases)."""


def normalize_catalog_metadata_type(type: str) -> str:
    """Map API-facing types to normalized storage names (e.g. timeseries -> indicator)."""
    if type == "timeseries":
        type = "indicator"

    if type == "survey":
        type = "microdata"

    return type
