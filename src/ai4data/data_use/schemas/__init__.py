"""Schema definitions for entity extraction.

DatasetSchema is the single canonical schema for the three-model pipeline
(Call 1 entity, Call 1b relation, Call 2 usage/attributes), together with the
label vocabularies of the GLiNER and LFM models.
"""

from .dataset_schema import (
    DATA_TYPES,
    ENTITY_LABELS,
    SOURCE_ATTRIBUTE_KEYS,
    THEME_TAGS,
    USAGE_ACTION_TO_PURPOSE,
    USAGE_ACTIONS,
    USAGE_ROLES,
    DatasetSchema,
)

__all__ = [
    "DATA_TYPES",
    "ENTITY_LABELS",
    "SOURCE_ATTRIBUTE_KEYS",
    "THEME_TAGS",
    "USAGE_ACTION_TO_PURPOSE",
    "USAGE_ACTIONS",
    "USAGE_ROLES",
    "DatasetSchema",
]
