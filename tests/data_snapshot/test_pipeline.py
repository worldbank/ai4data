"""Test model selection and model-scoped output orchestration."""

from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from ai4data.data_snapshot import Detection, run_extraction, run_pipeline
from ai4data.data_snapshot import pipeline
from ai4data.data_snapshot.adapters import create_adapter


class _FakeDetector:
    """Return one table detection for pipeline tests."""

    def detect(self, image: Image.Image) -> list[Detection]:
        """Return a full-page table detection."""
        return [Detection("table", (0.0, 0.0, 1.0, 1.0), 0.9)]


@pytest.mark.parametrize("model_args", [[], ["YOLO11"]])
def test_run_extraction_selects_one_model_and_scopes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_args: list[str]
) -> None:
    """Use one selected adapter and write beneath its model directory."""
    pdf_path = tmp_path / "report.pdf"
    with pymupdf.open() as document:
        document.new_page(width=72, height=72)
        document.save(pdf_path)
    selected_models: list[str] = []

    def fake_create_adapter(model_name: str) -> _FakeDetector:
        """Record and return the selected test detector."""
        selected_models.append(model_name)
        return _FakeDetector()

    monkeypatch.setattr(pipeline, "create_adapter", fake_create_adapter)

    paths = run_extraction(pdf_path, tmp_path / "output", *model_args, dpi=72)

    expected_model = model_args[0].lower() if model_args else "doclayoutyolo"
    assert selected_models == [expected_model]
    assert paths == [
        tmp_path / "output" / expected_model / "snapshots" / "report_table_000.png"
    ]


def test_create_adapter_rejects_unknown_model() -> None:
    """Reject an unknown model identifier."""
    with pytest.raises(ValueError, match="Unsupported model: unknown"):
        create_adapter("unknown")


def test_run_pipeline_skips_all_stages_for_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Skip both stages when an exact output already exists."""
    pdf_path = tmp_path / "report.pdf"
    with pymupdf.open() as document:
        document.new_page(width=72, height=72)
        document.save(pdf_path)
    snapshot_path = tmp_path / "output" / "tfid" / "snapshots" / "report_table_000.png"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"existing")
    monkeypatch.setattr(
        pipeline,
        "create_adapter",
        lambda model_name: (_ for _ in ()).throw(
            AssertionError("Skipped extraction loaded the snapshot detector.")
        ),
    )

    paths = run_pipeline(
        pdf_path,
        tmp_path / "output",
        "tfid",
        dpi=72,
    )

    assert paths == []
    assert capsys.readouterr().out == (
        "Existing output files for report.pdf exist. "
        "Skipped data snapshot and metadata extraction.\n"
    )


def test_run_extraction_force_cleans_exact_outputs_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Remove crops and metadata for only the forced PDF stem."""
    pdf_path = tmp_path / "report.pdf"
    with pymupdf.open() as document:
        document.new_page(width=72, height=72)
        document.save(pdf_path)
    model_dir = tmp_path / "output" / "tfid"
    snapshot_path = model_dir / "snapshots" / "report_table_000.png"
    metadata_path = model_dir / "metadata" / "report_table_000.json"
    prefix_neighbor = model_dir / "snapshots" / "report_figure_2024_table_000.png"
    snapshot_path.parent.mkdir(parents=True)
    metadata_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"old")
    metadata_path.write_text("{}\n", encoding="utf-8")
    prefix_neighbor.write_bytes(b"keep")
    monkeypatch.setattr(pipeline, "create_adapter", lambda model_name: _FakeDetector())

    paths = run_extraction(
        pdf_path,
        tmp_path / "output",
        "tfid",
        dpi=72,
        force=True,
    )

    assert metadata_path.exists() is False
    assert prefix_neighbor.read_bytes() == b"keep"
    assert len(paths) == 1
    assert paths[0] == snapshot_path
    assert "Cleaning up before continuing... Done!" in capsys.readouterr().out


def test_run_pipeline_extracts_metadata_for_new_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass newly extracted crops directly to metadata extraction."""
    pdf_path = tmp_path / "report.pdf"
    with pymupdf.open() as document:
        document.new_page(width=72, height=72)
        document.save(pdf_path)
    model_dir = tmp_path / "output" / "tfid"
    snapshot_path = model_dir / "snapshots" / "report_table_000.png"
    metadata_path = model_dir / "metadata" / "report_table_000.json"
    calls: list[tuple[list[Path], Path, str]] = []
    monkeypatch.setattr(pipeline, "create_adapter", lambda model_name: _FakeDetector())

    def fake_extract_metadata_files(
        snapshot_paths: list[Path],
        output_dir: Path,
        *,
        metadata_provider: str,
    ) -> list[Path]:
        """Record the crops passed to the metadata stage."""
        calls.append((snapshot_paths, output_dir, metadata_provider))
        return [metadata_path]

    monkeypatch.setattr(pipeline, "extract_metadata_files", fake_extract_metadata_files)

    paths = run_pipeline(
        pdf_path,
        tmp_path / "output",
        "tfid",
        dpi=72,
        metadata_provider="openai",
    )

    assert calls == [([snapshot_path], model_dir / "metadata", "openai")]
    assert paths == [snapshot_path, metadata_path]
