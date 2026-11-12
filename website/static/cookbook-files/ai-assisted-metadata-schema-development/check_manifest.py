"""Check identities and split boundaries in a small artifact register.

Usage: python check_manifest.py source_images.csv
The CSV must contain snapshot_id, split, corpus, and parent_document_id.
This standard-library script prints counts and writes nothing. Exit status is
0 for a valid register and 2 for invalid input. What this does not check:
representativeness, image contents, permissions, or unlisted study artifacts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def summarize_manifest(rows: list[dict[str, str]]) -> dict[str, int]:
    """Validate the register and count identities at both split boundaries.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Register rows with snapshot, split, corpus, and parent identifiers.

    Returns
    -------
    dict[str, int]
        Snapshot and parent counts for each split and shared parent count.

    Raises
    ------
    ValueError
        If rows are empty, required values are absent, a split is unknown,
        or a snapshot identity occurs more than once, including across splits.
    """
    if not rows:
        raise ValueError("The register is empty.")
    names: set[str] = set()
    snapshots: dict[str, set[str]] = {"development": set(), "heldout": set()}
    parents: dict[str, set[tuple[str, str]]] = {
        "development": set(),
        "heldout": set(),
    }
    for row in rows:
        for key in ["snapshot_id", "split", "corpus", "parent_document_id"]:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing register value: {key}")
        split = row["split"]
        if split not in snapshots:
            raise ValueError(f"Unknown split: {split}")
        name = row["snapshot_id"]
        if name in names:
            raise ValueError(f"Duplicate or overlapping snapshot: {name}")
        names.add(name)
        snapshots[split].add(name)
        parents[split].add((row["corpus"], row["parent_document_id"]))
    return {
        "development_snapshots": len(snapshots["development"]),
        "development_parents": len(parents["development"]),
        "heldout_snapshots": len(snapshots["heldout"]),
        "heldout_parents": len(parents["heldout"]),
        "shared_parents": len(parents["development"] & parents["heldout"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Print register counts without reading images or changing files.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments; None reads the process arguments.

    Returns
    -------
    int
        Zero on success. Invalid input exits with argparse status 2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("register", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.register.open(encoding="utf-8", newline="") as handle:
            result = summarize_manifest(list(csv.DictReader(handle)))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for split in ["development", "heldout"]:
        print(
            f"{split}: {result[split + '_snapshots']} snapshots, "
            f"{result[split + '_parents']} parent documents"
        )
    print("snapshot overlap: 0")
    print(f"shared parent documents: {result['shared_parents']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
