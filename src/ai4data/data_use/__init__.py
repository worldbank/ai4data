"""ai4data.data_use - Dataset mention extraction library.

This module provides tools for extracting dataset mentions from text and documents.
It requires optional dependencies to be installed:

    uv pip install ai4data[datause]

For harmonization features:

    uv pip install ai4data[harmonization]
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Lazy imports to avoid requiring dependencies at import time
_DatasetExtractor = None
_deduplicate_extraction = None
_ModelManager = None
_DatasetSchema = None


def _check_datause_deps():
    """Check if datause dependencies are installed."""
    try:
        import gliner2  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _get_dataset_extractor():
    """Get the DatasetExtractor class, importing lazily."""
    global _DatasetExtractor
    if _DatasetExtractor is None:
        if not _check_datause_deps():
            raise ImportError(
                "DatasetExtractor requires datause dependencies. "
                'Install with: uv pip install "ai4data[datause]"'
            )
        from .extractors.dataset_extractor import DatasetExtractor
        _DatasetExtractor = DatasetExtractor
    return _DatasetExtractor


def _get_deduplicate_extraction():
    """Get the deduplicate_extraction function, importing lazily."""
    global _deduplicate_extraction
    if _deduplicate_extraction is None:
        from .extractors.deduplication import deduplicate_extraction
        _deduplicate_extraction = deduplicate_extraction
    return _deduplicate_extraction


def _get_model_manager():
    """Get the ModelManager class, importing lazily."""
    global _ModelManager
    if _ModelManager is None:
        if not _check_datause_deps():
            raise ImportError(
                "ModelManager requires datause dependencies. "
                'Install with: uv pip install "ai4data[datause]"'
            )
        from .models.model_manager import ModelManager
        _ModelManager = ModelManager
    return _ModelManager


def _get_dataset_schema():
    """Get the DatasetSchema class, importing lazily."""
    global _DatasetSchema
    if _DatasetSchema is None:
        from .schemas.dataset_schema import DatasetSchema
        _DatasetSchema = DatasetSchema
    return _DatasetSchema


# Public API - these are functions that return the actual classes
# to maintain backwards compatibility
class DatasetExtractor:
    """Proxy class for lazy loading DatasetExtractor."""
    def __new__(cls, *args, **kwargs):
        RealClass = _get_dataset_extractor()
        return RealClass(*args, **kwargs)


class ModelManager:
    """Proxy class for lazy loading ModelManager."""
    def __new__(cls, *args, **kwargs):
        RealClass = _get_model_manager()
        return RealClass(*args, **kwargs)


class DatasetSchema:
    """Proxy class for lazy loading DatasetSchema."""
    def __new__(cls, *args, **kwargs):
        RealClass = _get_dataset_schema()
        return RealClass(*args, **kwargs)


def deduplicate_extraction(*args, **kwargs):
    """Deduplicate dataset extraction results."""
    func = _get_deduplicate_extraction()
    return func(*args, **kwargs)


__version__ = "0.1.0"

# Convenience functions for simple usage
_default_extractor = None


def _get_default_extractor():
    """Get or create the default extractor instance."""
    global _default_extractor
    if _default_extractor is None:
        RealExtractor = _get_dataset_extractor()
        _default_extractor = RealExtractor()
    return _default_extractor


def extract_from_text(
    text: str,
    include_confidence: bool = False,
    custom_schema: Optional[Any] = None,
    exclude_non_datasets: bool = True,
    dataset_threshold: Optional[float] = None,
    max_tokens: int = 200,
    model_id: Optional[str] = None,
    enable_chunking: bool = True,
    use_classifier: bool = False,
) -> Dict[str, Any]:
    """Extract dataset mentions from text.

    Args:
        text: Input text to extract from
        include_confidence: Whether to include confidence scores
        custom_schema: Optional custom schema to use instead of default
        exclude_non_datasets: If True, filter out datasets with dataset_tag="non-dataset"
        dataset_threshold: Optional confidence threshold for dataset_name field (0.0-1.0)
        max_tokens: Maximum tokens per chunk for long texts (default: 500)
        model_id: Optional model ID to use for this specific extraction
        enable_chunking: Whether to split long text into chunks (default: True)
        use_classifier: Whether to use pre-filtering classifier (default: False)

    Returns:
        Dict with 'input_text' and 'datasets' keys containing the original text
        and list of extracted dataset mentions. Datasets with all None values
        (excluding classification fields) are automatically filtered out.

    Example:
        >>> from ai4data import extract_from_text
        >>> text = "We used the 2022 DHS survey data..."
        >>> result = extract_from_text(text, include_confidence=True)
        >>> print(result['input_text'])
        >>> print(result['datasets'])
    """
    extractor = _get_default_extractor()
    return extractor.extract_from_text(
        text,
        include_confidence=include_confidence,
        custom_schema=custom_schema,
        exclude_non_datasets=exclude_non_datasets,
        dataset_threshold=dataset_threshold,
        max_tokens=max_tokens,
        model_id=model_id,
        enable_chunking=enable_chunking,
        use_classifier=use_classifier,
    )


def extract_from_document(
    source: Union[str, Path],
    include_confidence: bool = True,
    custom_schema: Optional[Any] = None,
    n_pages: int = 1,
    include_metadata: bool = True,
    exclude_non_datasets: bool = True,
    dataset_threshold: Optional[float] = None,
    max_tokens: int = 200,
    use_classifier: bool = True,
    skip_references: bool = True,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Extract dataset mentions from a PDF document.

    Args:
        source: Path to PDF file or URL
        include_confidence: Whether to include confidence scores
        custom_schema: Optional custom schema to use instead of default
        n_pages: Number of pages per chunk
        include_metadata: Whether to include metadata
        exclude_non_datasets: If True, filter out datasets with dataset_tag="non-dataset"
        dataset_threshold: Optional confidence threshold for dataset_name field (0.0-1.0)
        max_tokens: Maximum tokens per chunk for long texts (default: 500)
        use_classifier: Whether to use pre-filtering classifier (default: True)
        skip_references: If True, skip pages in references/appendix sections (default: True)
        verbose: If True, print logging when references are detected and skipped
        >>> results = extract_from_document(
        ...     "https://example.com/report.pdf",
        ...     include_confidence=True,
        ...     include_metadata=True
        ... )
        >>> for result in results:
        ...     print(f"Page {result['pages']}: {result.get('dataset_name')}")
    """
    extractor = _get_default_extractor()
    return extractor.extract_from_document(
        source,
        include_confidence=include_confidence,
        custom_schema=custom_schema,
        n_pages=n_pages,
        include_metadata=include_metadata,
        exclude_non_datasets=exclude_non_datasets,
        dataset_threshold=dataset_threshold,
        max_tokens=max_tokens,
        use_classifier=use_classifier,
        skip_references=skip_references,
        verbose=verbose,
    )


def initialize_harmonization_env(
    country_map_path=None,
    min_population=500_000,
    embedder_model="avsolatorio/GIST-all-MiniLM-L6-v2",
):
    """
    Initialize the harmonization environment.

    This preloads all necessary resources (country maps, embeddings, etc.)
    for the harmonization pipeline. Reuse the returned environment across
    multiple harmonization runs for better performance.

    Args:
        country_map_path: Path to country_map.json (optional)
        min_population: Minimum city population threshold
        embedder_model: Sentence transformer model name

    Returns:
        Environment dictionary for use with harmonize_datasets()

    Raises:
        ImportError: If harmonization dependencies are not installed
    """
    try:
        from .extractors.harmonization import initialize_environment
    except ImportError as e:
        raise ImportError(
            "Harmonization dependencies not installed. "
            'Install with: uv pip install "ai4data[harmonization]"'
        ) from e

    return initialize_environment(
        country_map_path=country_map_path,
        min_population=min_population,
        embedder_model=embedder_model,
    )


def harmonize_datasets(
    input_folder: str,
    output_dir: str = None,
    env=None,
    initial_wave: int = 50,
    incremental_wave: int = 10,
    sim_threshold: float = 0.5,
    data_format: str = "dedup",
    dataset_tags: List[str] = None,
):
    """
    Harmonize dataset names from dedup outputs into canonical families.

    This function processes deduplicated dataset extractions and groups
    similar dataset names into canonical families with aliases and
    country-specific prototypes.

    Args:
        input_folder: Root folder containing dedup outputs
                     Expected structure: input_folder/project_*/dedup/*.json
        output_dir: Where to save harmonization results (optional)
        env: Pre-initialized environment from initialize_harmonization_env() (optional)
        initial_wave: Number of folders to process in initial wave
        incremental_wave: Number of folders per incremental wave
        sim_threshold: Similarity threshold for clustering (0-1)
        data_format: "dedup" for dedup outputs or "s2orc" for S2ORC format
        dataset_tags: List of dataset tags to include (e.g., ['named'], ['named', 'descriptive']).
                     If None, defaults to ['named'] only. 'non-dataset' tags are always excluded.
                     Only applies when data_format="dedup".

    Returns:
        None (writes output files to output_dir)

    Raises:
        ImportError: If harmonization dependencies are not installed

    Example:
        >>> # Initialize environment once
        >>> env = initialize_harmonization_env()
        >>>
        >>> # Run harmonization with only named datasets (default)
        >>> harmonize_datasets(
        ...     input_folder="./revalidation/project_documents_outputs/",
        ...     output_dir="./harmonization_outputs/",
        ...     env=env,
        ...     initial_wave=40,
        ...     incremental_wave=10,
        ...     sim_threshold=0.5
        ... )
        >>>
        >>> # Run harmonization with named and descriptive datasets
        >>> harmonize_datasets(
        ...     input_folder="./revalidation/project_documents_outputs/",
        ...     output_dir="./harmonization_outputs/",
        ...     env=env,
        ...     dataset_tags=['named', 'descriptive']
        ... )
    """
    try:
        from .extractors.harmonization import run_incremental_pipeline
    except ImportError as e:
        raise ImportError(
            "Harmonization dependencies not installed. "
            'Install with: uv pip install "ai4data[harmonization]"'
        ) from e

    return run_incremental_pipeline(
        initial_folder=input_folder,
        output_dir=output_dir,
        env=env,
        initial_wave=initial_wave,
        incremental_wave=incremental_wave,
        sim_threshold=sim_threshold,
        data_format=data_format,
        dataset_tags=dataset_tags,
    )


__all__ = [
    "DatasetExtractor",
    "DatasetSchema",
    "ModelManager",
    "extract_from_text",
    "extract_from_document",
    "deduplicate_extraction",
    "harmonize_datasets",
    "initialize_harmonization_env",
]
