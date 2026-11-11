"""Review data snapshot metadata locally with an optional Streamlit editor."""

import argparse
from collections.abc import Sequence
from pathlib import Path


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the directory options shared by the launcher and Streamlit app."""
    parser = argparse.ArgumentParser(
        prog="python -m ai4data.data_snapshot.metadata_reviewer",
        description="Review data snapshot metadata locally and save reviewed copies.",
    )
    parser.add_argument(
        "--snapshots-root",
        type=Path,
        default=Path("output/doclayoutyolo/snapshots"),
        help="Snapshot directory (default: output/doclayoutyolo/snapshots).",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        help="Generated JSON directory (default: metadata beside snapshots).",
    )
    parser.add_argument(
        "--reviewed-root",
        type=Path,
        help="Reviewed JSON directory (default: reviewed beside snapshots).",
    )
    paths = parser.parse_args(args)
    paths.generated_root = (
        paths.generated_root or paths.snapshots_root.parent / "metadata"
    )
    paths.reviewed_root = (
        paths.reviewed_root or paths.snapshots_root.parent / "reviewed"
    )
    return paths
