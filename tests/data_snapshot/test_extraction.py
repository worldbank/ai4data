"""Test snapshot detection and cropping orchestration."""

from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from ai4data.data_snapshot import Detection, extract_snapshots


class _FakeDetector:
    """Return fixed detections for extraction tests."""

    def detect(self, image: Image.Image) -> list[Detection]:
        """Return intentionally unordered figure and table detections."""
        return [
            Detection("figure", (0.5, 0.5, 1.0, 1.0), 0.8),
            Detection("table", (0.0, 0.0, 0.5, 0.5), 0.9),
            Detection("figure", (0.0, 0.0, 0.5, 0.5), 0.7),
        ]


def _write_pdf(path: Path) -> None:
    """Write one 72-point square PDF page for a test."""
    with pymupdf.open() as document:
        document.new_page(width=72, height=72)
        document.save(path)


def test_extract_snapshots_saves_spatially_ordered_zero_based_names(
    tmp_path: Path,
) -> None:
    """Crop from one render and assign independent label counters."""
    pdf_path = tmp_path / "report.pdf"
    _write_pdf(pdf_path)

    paths = extract_snapshots(pdf_path, tmp_path / "output", _FakeDetector(), dpi=72)

    assert [path.name for path in paths] == [
        "report_table_000.png",
        "report_figure_000.png",
        "report_figure_001.png",
    ]
    assert [Image.open(path).size for path in paths] == [
        (36, 36),
        (36, 36),
        (36, 36),
    ]


def test_extract_snapshots_skips_existing_pdf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preserve existing snapshots and report a skipped source PDF."""
    pdf_path = tmp_path / "report.pdf"
    _write_pdf(pdf_path)
    existing = tmp_path / "output" / "snapshots" / "report_table_000.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep")

    paths = extract_snapshots(pdf_path, tmp_path / "output", _FakeDetector(), dpi=72)

    assert paths == [existing]
    assert existing.read_bytes() == b"keep"
    assert f"Skipping {pdf_path}: existing snapshots found" in capsys.readouterr().out


def test_extract_snapshots_force_replaces_existing_pdf(tmp_path: Path) -> None:
    """Remove all prior crops for one PDF before extracting it again."""
    pdf_path = tmp_path / "report.pdf"
    _write_pdf(pdf_path)
    snapshots_dir = tmp_path / "output" / "snapshots"
    snapshots_dir.mkdir(parents=True)
    existing = snapshots_dir / "report_table_000.png"
    stale = snapshots_dir / "report_figure_999.png"
    existing.write_bytes(b"old")
    stale.write_bytes(b"stale")

    paths = extract_snapshots(
        pdf_path,
        tmp_path / "output",
        _FakeDetector(),
        dpi=72,
        force=True,
    )

    assert [path.name for path in paths] == [
        "report_table_000.png",
        "report_figure_000.png",
        "report_figure_001.png",
    ]
    assert existing.read_bytes() != b"old"
    assert not stale.exists()


def test_extract_snapshots_rejects_duplicate_document_stems(tmp_path: Path) -> None:
    """Reject recursive inputs that would produce colliding output names."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    _write_pdf(first_dir / "report.pdf")
    _write_pdf(second_dir / "report.pdf")

    with pytest.raises(ValueError, match="filename stems must be unique"):
        extract_snapshots(tmp_path, tmp_path / "output", _FakeDetector(), dpi=72)
