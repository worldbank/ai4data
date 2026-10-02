"""Materialize the exact development and held-out snapshot collections."""

from __future__ import annotations

import argparse
import csv
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import quote


DATASET_REVISION = "626f3c54ba6d8f6e0ba0795f9e544e24b2650faf"
DATASET_URL = (
    "https://huggingface.co/datasets/ai4data/data-snapshot/resolve/"
    f"{DATASET_REVISION}"
)


def materialize(
    manifest_path: Path,
    output_root: Path,
    source_root: Path | None = None,
) -> None:
    """Copy or download every file named by a snapshot manifest.

    Parameters
    ----------
    manifest_path : Path
        CSV containing source, artifact type, snapshot, and metadata filenames.
    output_root : Path
        Destination containing ``snapshots`` and ``metadata`` directories.
    source_root : Path | None, optional
        Local dataset root. When omitted, files are downloaded from Hugging Face.
    """
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        source = row["source"]
        artifact_type = row["artifact_type"]
        snapshot_name = row["snapshot_filename"]
        metadata_name = row["metadata_filename"]
        snapshot_target = (
            output_root / "snapshots" / source / artifact_type / snapshot_name
        )
        metadata_target = output_root / "metadata" / source / metadata_name
        _materialize_file(
            "snapshots", source, snapshot_name, snapshot_target, source_root
        )
        _materialize_file(
            "metadata", source, metadata_name, metadata_target, source_root
        )


def _materialize_file(
    category: str,
    source: str,
    filename: str,
    target: Path,
    source_root: Path | None,
) -> None:
    """Copy or download one dataset file if it is not already present."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_root is not None:
        shutil.copy2(source_root / category / source / filename, target)
        return
    url = "/".join(
        [DATASET_URL, quote(category), quote(source), quote(filename, safe="")]
    )
    urllib.request.urlretrieve(url, target)


def main() -> None:
    """Parse command-line arguments and materialize the selected split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("development", "heldout", "all"), default="all"
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/source"))
    args = parser.parse_args()

    manifests = Path("data/manifests")
    if args.split in {"development", "all"}:
        materialize(
            manifests / "development_snapshots.csv",
            args.output_root / "development",
            args.source_root,
        )
    if args.split in {"heldout", "all"}:
        materialize(
            manifests / "heldout_snapshots.csv",
            args.output_root / "heldout",
            args.source_root,
        )


if __name__ == "__main__":
    main()
