"""Entity extractors."""

from .dataset_extractor import DatasetExtractor
from .deduplication import deduplicate_extraction

__all__ = ["DatasetExtractor", "deduplicate_extraction"]
