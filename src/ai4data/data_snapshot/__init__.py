"""Extract figure/table snapshots and metadata with optional model dependencies."""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "Detection": "detection",
    "DataSnapshotMetadata": "metadata_schema",
    "ExtractionResult": "metadata_extraction",
    "SnapshotDetector": "detection",
    "build_snapshot_stem": "naming",
    "discover_pdfs": "inputs",
    "discover_snapshot_images": "inputs",
    "extract_snapshots": "extraction",
    "extract_metadata": "metadata_extraction",
    "iter_pdf_pages": "rendering",
    "run_extraction": "pipeline",
    "run_pipeline": "pipeline",
    "submit_snapshot": "submission",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Resolve and cache public objects without importing optional runtimes.

    Parameters
    ----------
    name : str
        Public attribute requested from the module.

    Returns
    -------
    Any
        The corresponding public function or class.

    Raises
    ------
    AttributeError
        If the name is not a public export.
    """
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORT_MODULES[name]}"), name)
    globals()[name] = value
    return value
