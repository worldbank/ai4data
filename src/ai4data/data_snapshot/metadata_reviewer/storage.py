"""Discover, load, compare, validate, and save metadata review records."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4data.data_snapshot.inputs import discover_snapshot_images
from ai4data.data_snapshot.metadata_schema import DataSnapshotMetadata


@dataclass(frozen=True)
class ReviewItem:
    """Describe the files associated with one reviewable snapshot.

    Parameters
    ----------
    snapshot_id : str
        Globally unique snapshot filename stem.
    snapshot_path : pathlib.Path
        Path to the immutable snapshot image.
    generated_path : pathlib.Path
        Path to the immutable generated metadata JSON.
    reviewed_path : pathlib.Path
        Existing or intended path for the reviewed metadata JSON.
    """

    snapshot_id: str
    snapshot_path: Path
    generated_path: Path
    reviewed_path: Path

    @property
    def reviewed(self) -> bool:
        """Return whether a reviewed file exists for the snapshot.

        Returns
        -------
        bool
            ``True`` when the reviewed path is an existing file.
        """

        return self.reviewed_path.is_file()


@dataclass(frozen=True)
class DiscoveryResult:
    """Contain reviewable items and nonfatal discovery warnings.

    Parameters
    ----------
    items : tuple[ReviewItem, ...]
        Reviewable snapshot/generated pairs in deterministic ID order.
    warnings : tuple[str, ...]
        Nonfatal problems found while indexing the roots.
    """

    items: tuple[ReviewItem, ...]
    warnings: tuple[str, ...]


def discover_review_items(
    snapshot_root: str | Path,
    generated_root: str | Path,
    reviewed_root: str | Path,
) -> DiscoveryResult:
    """Discover reviewable files by globally unique snapshot stem.

    JSON filenames must exactly match the snapshot stem.
    New reviewed paths mirror the snapshot's relative layout. Files
    without a snapshot/generated counterpart are excluded with a warning.

    Parameters
    ----------
    snapshot_root : str | pathlib.Path
        Root recursively searched for PNG, JPEG, GIF, or WebP snapshots.
    generated_root : str | pathlib.Path
        Root recursively searched for generated JSON metadata.
    reviewed_root : str | pathlib.Path
        Root recursively searched for existing reviewed JSON metadata and used for
        new reviewed files.

    Returns
    -------
    DiscoveryResult
        Matched review items and nonfatal warnings.

    Raises
    ------
    FileNotFoundError
        If an input root does not exist or there are no supported images.
    NotADirectoryError
        If a supplied root exists but is not a directory.
    ValueError
        If the reviewed root overlaps an input root or a root contains duplicate IDs.
    """

    snapshots = Path(snapshot_root)
    generated = Path(generated_root)
    reviewed = Path(reviewed_root)
    _require_directory(snapshots)
    _require_directory(generated)
    if reviewed.exists() and not reviewed.is_dir():
        raise NotADirectoryError(reviewed)
    _require_separate_roots(generated, reviewed)
    _require_separate_roots(snapshots, reviewed)

    warnings: list[str] = []
    snapshot_paths: dict[str, Path] = {}
    for path in discover_snapshot_images(snapshots):
        _add_unique(snapshot_paths, path.stem, path, "snapshot")

    generated_paths = _index_json(generated, "generated")
    reviewed_paths = _index_json(reviewed, "reviewed") if reviewed.is_dir() else {}

    items: list[ReviewItem] = []
    for snapshot_id in sorted(snapshot_paths.keys() & generated_paths.keys()):
        snapshot_path = snapshot_paths[snapshot_id]
        reviewed_path = reviewed_paths.get(
            snapshot_id,
            reviewed / snapshot_path.relative_to(snapshots).with_suffix(".json"),
        )
        items.append(
            ReviewItem(
                snapshot_id=snapshot_id,
                snapshot_path=snapshot_path,
                generated_path=generated_paths[snapshot_id],
                reviewed_path=reviewed_path,
            )
        )

    for snapshot_id in sorted(snapshot_paths.keys() - generated_paths.keys()):
        warnings.append(
            f"Excluded snapshot without generated metadata: {snapshot_paths[snapshot_id]}"
        )
    for snapshot_id in sorted(generated_paths.keys() - snapshot_paths.keys()):
        warnings.append(
            f"Excluded generated metadata without snapshot: {generated_paths[snapshot_id]}"
        )
    for snapshot_id in sorted(
        reviewed_paths.keys() - (snapshot_paths.keys() & generated_paths.keys())
    ):
        warnings.append(
            f"Orphan reviewed metadata without an input pair: {reviewed_paths[snapshot_id]}"
        )

    return DiscoveryResult(items=tuple(items), warnings=tuple(warnings))


def load_review_records(
    item: ReviewItem,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load immutable generated metadata and the editable working record.

    Parameters
    ----------
    item : ReviewItem
        Review item whose JSON records should be loaded.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Generated metadata followed by an independent working copy. The working
        copy comes from reviewed when it exists, otherwise from generated metadata.

    Raises
    ------
    ValueError
        If either loaded JSON value is not an object.
    json.JSONDecodeError
        If either file contains invalid JSON.
    pydantic.ValidationError
        If a loaded record does not conform to the metadata schema.
    """

    generated = _load_json_object(item.generated_path)
    working_path = item.reviewed_path if item.reviewed else item.generated_path
    working = _load_json_object(working_path)
    validate_metadata(generated)
    validate_metadata(working)
    return generated, deepcopy(working)


def metadata_values_equal(left: Any, right: Any) -> bool:
    """Compare metadata values while treating missing and null as equivalent.

    Parameters
    ----------
    left : Any
        First JSON-compatible value.
    right : Any
        Second JSON-compatible value.

    Returns
    -------
    bool
        Whether the values are equal after omitting null-valued object fields.
    """

    return _without_null_fields(left) == _without_null_fields(right)


def validate_metadata(record: dict[str, Any]) -> DataSnapshotMetadata:
    """Validate a working record with the canonical Pydantic model.

    Parameters
    ----------
    record : dict[str, Any]
        Editable metadata record.

    Returns
    -------
    DataSnapshotMetadata
        Validated canonical model instance.

    Raises
    ------
    pydantic.ValidationError
        If the record violates the canonical metadata schema.
    """

    return DataSnapshotMetadata.model_validate(record)


def save_reviewed(item: ReviewItem, record: dict[str, Any]) -> DataSnapshotMetadata:
    """Validate and atomically save one canonical reviewed metadata record.

    Parameters
    ----------
    item : ReviewItem
        Review item defining the immutable generated and writable reviewed paths.
    record : dict[str, Any]
        Edited metadata record to validate and save.

    Returns
    -------
    DataSnapshotMetadata
        Validated model whose canonical JSON representation was saved.

    Raises
    ------
    ValueError
        If the reviewed path would overwrite an input file.
    pydantic.ValidationError
        If the record violates the canonical metadata schema.
    OSError
        If the atomic write cannot be completed.
    """

    if item.reviewed_path.resolve() in {
        item.generated_path.resolve(),
        item.snapshot_path.resolve(),
    }:
        raise ValueError("Reviewed metadata path cannot overwrite an input file.")
    metadata = validate_metadata(record)
    serialized = (
        json.dumps(
            metadata.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    _atomic_write_text(item.reviewed_path, serialized)
    return metadata


def _require_directory(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)


def _require_separate_roots(input_root: Path, reviewed: Path) -> None:
    input_resolved = input_root.resolve()
    reviewed_resolved = reviewed.resolve()
    if input_resolved.is_relative_to(
        reviewed_resolved
    ) or reviewed_resolved.is_relative_to(input_resolved):
        raise ValueError("Input and reviewed roots must not overlap.")


def _files_with_suffix(root: Path, suffix: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == suffix
    )


def _index_json(root: Path, kind: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in _files_with_suffix(root, ".json"):
        _add_unique(indexed, path.stem, path, kind)
    return indexed


def _add_unique(index: dict[str, Path], key: str, path: Path, kind: str) -> None:
    existing = index.get(key)
    if existing is not None:
        raise ValueError(f"Duplicate {kind} ID {key!r}: {existing} and {path}")
    index[key] = path


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Metadata JSON must contain an object: {path}")
    return value


def _without_null_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_null_fields(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_null_fields(item) for item in value]
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
