"""Build a sample register from a selected batch of data snapshots.

Each discovery/<sample-id>/ directory contains one image and metadata.json.
Each challenge/<sample-id>/ directory contains one image and no metadata.json.
A source CSV supplies attribution that cannot be inferred from image filenames.
Image and annotation paths in the register are relative to its output directory.
Requires Python 3.10 or later and uses only the standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SOURCE_FIELDS = {
    "sample_id",
    "collection",
    "source_document",
    "source_pdf",
    "pdf_page",
    "provenance",
}


def build_register(
    batch: Path,
    sources: Path,
    output: Path,
    snapshot_revision: str | None = None,
    gold_revision: str | None = None,
) -> dict[str, Any]:
    """Collect sample paths, source attribution, and checksums.

    Parameters
    ----------
    batch : Path
        Folder containing discovery and challenge sample directories.
    sources : Path
        CSV with one source row per sample.
    output : Path
        Register path used as the base for relative image and reference paths.
    snapshot_revision : str or None
        Source dataset revision, when applicable.
    gold_revision : str or None
        Human-reviewed annotation release revision, when applicable.

    Returns
    -------
    dict[str, Any]
        Register with deterministic sample ordering and file checksums.

    Raises
    ------
    ValueError
        If sample folders, annotations, or source rows are inconsistent.
    OSError
        If required input files or directories cannot be read.
    """
    source_rows = {}
    with sources.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not SOURCE_FIELDS.issubset(reader.fieldnames or []):
            raise ValueError("Source CSV requires: " + ", ".join(sorted(SOURCE_FIELDS)))
        for row in reader:
            if any(
                not row.get(field) or not row[field].strip() for field in SOURCE_FIELDS
            ):
                raise ValueError("Every source row must fill all required columns.")
            sample_id = row["sample_id"]
            if sample_id in source_rows:
                raise ValueError(f"Duplicate source row: {sample_id}")
            source_rows[sample_id] = row

    samples = []
    seen_ids = set()
    for split in ("discovery", "challenge"):
        folders = sorted((batch / split).iterdir())
        if not folders or any(not folder.is_dir() for folder in folders):
            raise ValueError(f"{split} must contain sample directories.")
        for folder in folders:
            sample_id = folder.name
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", sample_id):
                raise ValueError(f"Invalid sample ID: {sample_id}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            seen_ids.add(sample_id)
            if sample_id not in source_rows:
                raise ValueError(f"Missing source row: {sample_id}")
            images = [
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
            if len(images) != 1:
                raise ValueError(f"{sample_id} must contain exactly one image.")
            image = images[0]
            reference = folder / "metadata.json"
            has_reference = reference.is_file()
            if has_reference != (split == "discovery"):
                raise ValueError(
                    f"{sample_id}: metadata.json is required for Discovery "
                    "and must be absent for Challenge."
                )
            gold_sha256 = None
            gold_path = None
            if has_reference:
                annotation = json.loads(reference.read_text(encoding="utf-8"))
                if not isinstance(annotation, dict):
                    raise ValueError(
                        f"{sample_id}: metadata.json must be a JSON object."
                    )
                gold_sha256 = hashlib.sha256(reference.read_bytes()).hexdigest()
                gold_path = Path(os.path.relpath(reference, output.parent)).as_posix()
            source = source_rows[sample_id]
            pdf_page = int(source["pdf_page"])
            if pdf_page < 1:
                raise ValueError(f"{sample_id}: pdf_page must be a positive integer.")
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "image": Path(os.path.relpath(image, output.parent)).as_posix(),
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "collection": source["collection"],
                    "source_document": source["source_document"],
                    "source_pdf": source["source_pdf"],
                    "pdf_page": pdf_page,
                    "provenance": source["provenance"],
                    "gold_reference": gold_path,
                    "gold_sha256": gold_sha256,
                }
            )
    if seen_ids != set(source_rows):
        raise ValueError(
            "The source CSV contains rows without matching sample folders."
        )
    return {
        "evidence_status": "real_inputs",
        "snapshot_dataset_revision": snapshot_revision,
        "gold_dataset_revision": gold_revision,
        "samples": samples,
    }


def write_register(register: dict[str, Any], output: Path) -> None:
    """Save a complete register while preserving any existing different file.

    Parameters
    ----------
    register : dict[str, Any]
        Validated register to serialize.
    output : Path
        Destination whose parent directory already exists.

    Raises
    ------
    ValueError
        If the destination exists with different contents.
    OSError
        If the register cannot be written.
    """
    content = json.dumps(register, indent=2, ensure_ascii=False) + "\n"
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary_directory:
        temporary = Path(temporary_directory) / "samples.json"
        temporary.write_text(content, encoding="utf-8")
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_text(encoding="utf-8") != content:
                raise ValueError(
                    f"{output} already exists with different contents. "
                    "Use a new output path for a changed batch."
                ) from None


def main(argv: list[str] | None = None) -> int:
    """Register a batch from command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Arguments, or None to read the process arguments.

    Returns
    -------
    int
        Zero on success, or one for invalid input or a file error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-revision")
    parser.add_argument("--gold-revision")
    args = parser.parse_args(argv)
    try:
        register = build_register(
            args.batch.resolve(),
            args.sources,
            args.output.resolve(),
            args.snapshot_revision,
            args.gold_revision,
        )
        write_register(register, args.output)
    except (OSError, ValueError) as error:
        print(f"Registration failed: {error}", file=sys.stderr)
        return 1
    print(f"{args.output}: {len(register['samples'])} samples registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
