"""
On-disk layout for discovery workflows (catalog cache, scraped id lists, PDF cache, contextual dimensions).

The data root is set via :func:`init_discovery_paths` and/or env ``AI4DATA_DISCOVERY_DATA_PATH``; see ``README.md``.

**Precedence:** explicit path passed to :func:`init_discovery_paths` wins; otherwise ``AI4DATA_DISCOVERY_DATA_PATH``
if set; otherwise ``<package>/ai4data/data`` relative to this file.
"""

from __future__ import annotations

from pathlib import Path

from .config import discovery_data
from .type_normalization import normalize_catalog_metadata_type

# Populated by init_discovery_paths (and module load default).
_data_root: Path | None = None
METADATA_IDS_DIR: Path
METADATA_CACHE_DIR: Path
DOCUMENT_CACHE_DIR: Path


def _default_data_root() -> Path:
    """Package-relative ``ai4data/data``, or ``discovery_data.data_path`` when set."""
    if discovery_data.data_path is not None:
        return discovery_data.data_path.expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data"


def init_discovery_paths(data_path: Path | None = None) -> None:
    """
    Set the root directory for discovery cache and id-list paths.

    Pass ``data_path`` to force a root regardless of ``AI4DATA_DISCOVERY_DATA_PATH``.
    Pass ``None`` to use the env-based default from :func:`_default_data_root`.
    Safe to call multiple times (e.g. in tests).
    """
    global _data_root, METADATA_IDS_DIR, METADATA_CACHE_DIR, DOCUMENT_CACHE_DIR
    root = data_path if data_path is not None else _default_data_root()
    _data_root = root
    METADATA_IDS_DIR = root / "metadata_ids"
    METADATA_CACHE_DIR = root / "metadata_cache"
    DOCUMENT_CACHE_DIR = root / "document_cache"


def get_discovery_data_root() -> Path:
    """Return the current discovery data root (after :func:`init_discovery_paths`)."""
    if _data_root is None:
        init_discovery_paths()
    assert _data_root is not None
    return _data_root


init_discovery_paths()


def ensure_dir_exists(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path


def construct_path(directory: Path, filename: str) -> Path:
    ensure_dir_exists(directory)
    return directory / filename


def get_metadata_ids_path(metadata_type: str) -> Path:
    metadata_type = normalize_catalog_metadata_type(metadata_type)

    return construct_path(METADATA_IDS_DIR, f"metadata_ids_{metadata_type}.json")


def get_metadata_cache_path(idno: str, metadata_type: str) -> Path:
    return construct_path(METADATA_CACHE_DIR / metadata_type, f"{metadata_type}_{idno}.json")


def get_document_cache_path(idno: str, metadata_type: str) -> Path:
    return construct_path(DOCUMENT_CACHE_DIR / metadata_type, f"{metadata_type}_{idno}.pdf")


def get_contextualized_dimensions_path(idno: str, raw: bool = False) -> Path:
    root = get_discovery_data_root()
    if raw:
        return root / "contextual_dimensions" / idno / f"contextual_dimensions.{idno}.raw.json"
    return root / "contextual_dimensions" / idno / f"contextual_dimensions.{idno}.txt"
