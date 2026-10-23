"""Test PDF discovery, rendering, and snapshot naming contracts."""

from pathlib import Path

import pymupdf
import pytest

from ai4data.data_snapshot import (
    build_snapshot_stem,
    discover_pdfs,
    discover_snapshot_images,
    iter_pdf_pages,
)


def test_discover_pdfs_accepts_file_or_directory(tmp_path: Path) -> None:
    """Discover one file directly or PDFs recursively in path order."""
    first_pdf = tmp_path / "a.pdf"
    second_pdf = tmp_path / "nested" / "b.PDF"
    second_pdf.parent.mkdir()
    first_pdf.touch()
    second_pdf.touch()
    (tmp_path / "notes.txt").touch()

    assert discover_pdfs(first_pdf) == [first_pdf]
    assert discover_pdfs(tmp_path) == [first_pdf, second_pdf]


def test_discover_pdfs_rejects_invalid_inputs(tmp_path: Path) -> None:
    """Reject missing, non-PDF, and empty-directory inputs."""
    text_path = tmp_path / "notes.txt"
    text_path.touch()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Input does not exist"):
        discover_pdfs(tmp_path / "missing")
    with pytest.raises(ValueError, match="not a PDF"):
        discover_pdfs(text_path)
    with pytest.raises(FileNotFoundError, match="No PDF files"):
        discover_pdfs(empty_dir)


def test_discover_snapshot_images_accepts_file_or_directory(tmp_path: Path) -> None:
    """Discover one image directly or supported images recursively in order."""
    first_image = tmp_path / "a.png"
    second_image = tmp_path / "nested" / "b.JPG"
    second_image.parent.mkdir()
    first_image.touch()
    second_image.touch()
    (tmp_path / "notes.txt").touch()

    assert discover_snapshot_images(first_image) == [first_image]
    assert discover_snapshot_images(tmp_path) == [first_image, second_image]


def test_discover_snapshot_images_rejects_invalid_inputs(tmp_path: Path) -> None:
    """Reject missing, unsupported, and empty-directory image inputs."""
    text_path = tmp_path / "notes.txt"
    text_path.touch()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Input does not exist"):
        discover_snapshot_images(tmp_path / "missing")
    with pytest.raises(ValueError, match="not a supported image"):
        discover_snapshot_images(text_path)
    with pytest.raises(FileNotFoundError, match="No snapshot images"):
        discover_snapshot_images(empty_dir)


def test_iter_pdf_pages_yields_rgb_pages(tmp_path: Path) -> None:
    """Render pages once in document order using PyMuPDF."""
    pdf_path = tmp_path / "document.pdf"
    with pymupdf.open() as document:
        document.new_page(width=72, height=36)
        document.new_page(width=144, height=72)
        document.save(pdf_path)

    pages = list(iter_pdf_pages(pdf_path, dpi=72))

    assert [(index, image.mode, image.size) for index, image in pages] == [
        (0, "RGB", (72, 36)),
        (1, "RGB", (144, 72)),
    ]


def test_iter_pdf_pages_rejects_invalid_dpi(tmp_path: Path) -> None:
    """Reject a non-positive rendering resolution before opening the PDF."""
    pdf_path = tmp_path / "document.pdf"
    pdf_path.touch()

    with pytest.raises(ValueError, match="DPI"):
        list(iter_pdf_pages(pdf_path, dpi=0))


def test_build_snapshot_stem_uses_zero_based_per_label_names() -> None:
    """Normalize supported labels and preserve zero-based indices."""
    assert build_snapshot_stem("reports/source.pdf", "Figure", 0) == (
        "source_figure_000"
    )
    assert build_snapshot_stem("reports/source.pdf", "table", 7) == ("source_table_007")


def test_build_snapshot_stem_rejects_invalid_values() -> None:
    """Reject unsupported labels and negative indices."""
    with pytest.raises(ValueError, match="Unsupported snapshot label"):
        build_snapshot_stem("source.pdf", "caption", 0)
    with pytest.raises(ValueError, match="zero or greater"):
        build_snapshot_stem("source.pdf", "figure", -1)
