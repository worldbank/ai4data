"""Schema definitions for entity extraction.

DatasetSchema is the single canonical schema for the three-model swarm
pipeline (Call 1 entity, Call 1b relation, Call 2 classification).
"""

from .dataset_schema import DatasetSchema

__all__ = ["DatasetSchema"]
