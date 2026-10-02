"""Catalog metadata type strings used in filenames (aligned with NADA API aliases)."""

# NADA API type -> normalized storage/handler name. Types not listed here (document,
# geospatial, table, script, image, video) use the same name on both sides.
_API_TO_METADATA_TYPE = {
    "timeseries": "indicator",
    "survey": "microdata",
    "timeseriesdb": "indicator-db",
    "timeseries-db": "indicator-db",
}

# Normalized name -> the type string NADA's search API filters on.
_METADATA_TO_API_TYPE = {
    "indicator": "timeseries",
    "microdata": "survey",
    "indicator-db": "timeseriesdb",
}


def normalize_catalog_metadata_type(type: str) -> str:
    """Map API-facing types to normalized storage names (e.g. timeseries -> indicator)."""
    return _API_TO_METADATA_TYPE.get(type, type)


def to_catalog_api_type(type: str) -> str:
    """Map normalized names back to NADA API types (e.g. indicator -> timeseries)."""
    return _METADATA_TO_API_TYPE.get(type, type)
