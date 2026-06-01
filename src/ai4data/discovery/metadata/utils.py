# Description: Utility functions for metadata
import hashlib
import json
import re
import uuid

from ..config import metadata_catalog
from ..paths import get_metadata_ids_path

IDNO_KEYS = dict(
    indicator="series_description.idno",
    document="document_description.title_statement.idno",
    microdata="study_desc.title_statement.idno",
    geospatial="description.idno",
)


def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()

    return uuid.UUID(hex=hex_string)


def get_idno_key(metadata_type: str, prefix: str = None) -> str:
    """
    Get the key for the idno for the given metadata type. This is useful as input to the group_search if the metadata is not normalized.

    Args:
        metadata_type (str): The metadata type.
        prefix (str, optional): The prefix to add to the key. Defaults to None.

    Returns:
        str: The key for the idno.
    """
    assert metadata_type in IDNO_KEYS, f"Metadata type {metadata_type} not supported"

    key = IDNO_KEYS.get(metadata_type)

    if prefix:
        key = f"{prefix}.{key}"

    return key


def get_idno(metadata: dict, metadata_type: str) -> str:
    if metadata_type == "indicator":
        idno = metadata["series_description"]["idno"]
    elif metadata_type == "document":
        idno = metadata["document_description"]["title_statement"]["idno"]
    elif metadata_type == "microdata":
        idno = metadata["study_desc"]["title_statement"]["idno"]
    elif metadata_type == "geospatial":
        idno = metadata["description"]["idno"]
    else:
        raise ValueError(f"Type {metadata_type} not supported")

    return idno


def get_metadata_ids(metadata_type: str) -> list[dict]:
    """
    Get the metadata ids collected from the metadata catalog for the given type.
    """

    fpath = get_metadata_ids_path(metadata_type=metadata_type)

    with open(fpath) as f:
        metadata_ids = json.load(f)

    return metadata_ids


def parse_dimension_label_values(metadata: dict) -> dict[str, list]:
    pattern = re.compile(r"\[(.*?)\]")

    series: dict = metadata.get("series_description", {})
    dimensions: list[dict] = series.get("dimensions", [])
    dimension_label_values: dict[str, list] = {}

    for dimension in dimensions:
        dimension_label = dimension["label"]
        m = pattern.match(dimension_label)
        if m:
            label = m.group(1)
            value = pattern.sub("", dimension_label, count=1).strip()
            if label in dimension_label_values:
                dimension_label_values[label].append(value)
            else:
                dimension_label_values[label] = [value]

    if len(dimensions) > 0:
        assert len(dimension_label_values) > 0, (
            "No dimension label values found despite having dimensions"
        )

    return dimension_label_values


IDNO_ID_MAP = {}

_document_ids_path = get_metadata_ids_path(metadata_type="document")
if _document_ids_path.is_file():
    metadata_ids = get_metadata_ids(metadata_type="document")
    for metadata_id in metadata_ids:
        IDNO_ID_MAP[metadata_id["idno"]] = metadata_id["id"]


def get_thumbnail_url(idno: str) -> str | None:
    if idno not in IDNO_ID_MAP:
        return None

    db_id = IDNO_ID_MAP[idno]

    return metadata_catalog.thumbnail_url.format(db_id=db_id)
