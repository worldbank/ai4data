"""Tests for deterministic study-data materialization."""

import csv
from pathlib import Path

from scripts.fetch_data import DATASET_REVISION, DATASET_URL, materialize


def test_dataset_downloads_use_the_frozen_revision() -> None:
    """Remote materialization is pinned to the study's dataset revision."""
    assert len(DATASET_REVISION) == 40
    assert DATASET_URL.endswith(DATASET_REVISION)


def test_materialize_copies_manifest_files(tmp_path: Path) -> None:
    """A manifest selects exactly its named snapshot and metadata files."""
    source = tmp_path / "source"
    (source / "snapshots/demo").mkdir(parents=True)
    (source / "metadata/demo").mkdir(parents=True)
    (source / "snapshots/demo/report_figure_000.png").write_bytes(b"png")
    (source / "metadata/demo/report.json").write_text("{}", encoding="utf-8")

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source",
                "artifact_type",
                "snapshot_filename",
                "document_id",
                "metadata_filename",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "source": "demo",
                "artifact_type": "figure",
                "snapshot_filename": "report_figure_000.png",
                "document_id": "report",
                "metadata_filename": "report.json",
            }
        )

    output = tmp_path / "output"
    materialize(manifest, output, source)

    assert (
        output / "snapshots/demo/figure/report_figure_000.png"
    ).read_bytes() == b"png"
    assert (output / "metadata/demo/report.json").read_text(encoding="utf-8") == "{}"
