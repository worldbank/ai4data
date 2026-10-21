"""Test the pipeline command-line interfaces."""

from pathlib import Path

import pytest

from ai4data.data_snapshot import cli


@pytest.mark.parametrize(
    ("extra_args", "expected_force", "expected_provider"),
    [
        ([], False, "openrouter"),
        (["-f", "--metadata-provider=openai"], True, "openai"),
    ],
)
def test_main_runs_pipeline_with_defaults_and_force(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    expected_force: bool,
    expected_provider: str,
) -> None:
    """Forward arguments and print paths returned by the pipeline."""
    calls: list[tuple[Path, Path, str, int, bool, str]] = []

    def fake_run_pipeline(
        input_path: Path,
        output_dir: Path,
        model_name: str,
        *,
        dpi: int,
        force: bool,
        metadata_provider: str,
    ) -> list[Path]:
        """Record the CLI values without running a model."""
        calls.append(
            (input_path, output_dir, model_name, dpi, force, metadata_provider)
        )
        return [Path("output/yolo11/snapshots/example_table_000.png")]

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    result = cli.main(["--model=yolo11", "--input=data/example.pdf", *extra_args])

    assert result == 0
    assert calls == [
        (
            Path("data/example.pdf"),
            Path("output"),
            "yolo11",
            300,
            expected_force,
            expected_provider,
        )
    ]
    assert capsys.readouterr().out == (
        "output/yolo11/snapshots/example_table_000.png\n"
    )


@pytest.mark.parametrize(
    ("model_args", "expected_model"),
    [([], "doclayoutyolo"), (["--model=tfid"], "tfid")],
)
def test_snapshot_main_runs_only_snapshot_extraction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model_args: list[str],
    expected_model: str,
) -> None:
    """Use the snapshot-only operation and print its output paths."""
    calls: list[str] = []

    def fake_run_extraction(*args: object, **kwargs: object) -> list[Path]:
        """Record that snapshot extraction was selected."""
        calls.append(str(args[2]))
        return [Path(f"output/{args[2]}/snapshots/example_table_000.png")]

    monkeypatch.setattr(cli, "run_extraction", fake_run_extraction)

    result = cli.snapshot_main(["--input=data/example.pdf", *model_args])

    assert result == 0
    assert calls == [expected_model]
    assert capsys.readouterr().out.endswith("example_table_000.png\n")


def test_metadata_main_infers_output_and_forwards_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Infer the metadata sibling and run only metadata extraction."""
    snapshot_path = tmp_path / "tfid" / "snapshots" / "report_table_000.png"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"image")
    calls: list[tuple[list[Path], Path, bool, str]] = []

    def fake_extract_metadata_files(
        snapshot_paths: list[Path],
        output_dir: Path,
        *,
        force: bool,
        metadata_provider: str,
    ) -> list[Path]:
        """Record metadata command values without making an API request."""
        paths = list(snapshot_paths)
        calls.append((paths, output_dir, force, metadata_provider))
        return [output_dir / "report_table_000.json"]

    monkeypatch.setattr(cli, "extract_metadata_files", fake_extract_metadata_files)

    result = cli.metadata_main(
        [
            f"--input={snapshot_path}",
            "--force",
            "--metadata-provider=openai",
        ]
    )

    metadata_dir = tmp_path / "tfid" / "metadata"
    assert result == 0
    assert calls == [([snapshot_path], metadata_dir, True, "openai")]
    assert capsys.readouterr().out == f"{metadata_dir / 'report_table_000.json'}\n"


def test_metadata_main_requires_output_for_arbitrary_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require an explicit output directory outside a snapshots directory."""
    snapshot_path = tmp_path / "report.png"
    snapshot_path.write_bytes(b"image")

    with pytest.raises(SystemExit) as error:
        cli.metadata_main([f"--input={snapshot_path}"])

    assert error.value.code == 2
    assert "--output-dir is required" in capsys.readouterr().err


def test_main_reports_pipeline_input_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Return a command-line error for an unsupported model."""

    def reject_model(*args: object, **kwargs: object) -> list[Path]:
        """Raise the pipeline's unsupported-model error."""
        raise ValueError("Unsupported model: unknown")

    monkeypatch.setattr(cli, "run_pipeline", reject_model)

    with pytest.raises(SystemExit) as error:
        cli.main(["--model=unknown", "--input=data/example.pdf"])

    assert error.value.code == 2
    assert "Unsupported model: unknown" in capsys.readouterr().err


def test_main_reports_runtime_error_without_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report a valid command's runtime failure without argparse help."""

    def reject_request(*args: object, **kwargs: object) -> list[Path]:
        """Raise a metadata runtime error."""
        raise RuntimeError("OpenAI credits are exhausted.")

    monkeypatch.setattr(cli, "run_pipeline", reject_request)

    result = cli.main(["--model=tfid", "--input=data/example.pdf"])

    assert result == 1
    assert capsys.readouterr().err == "error: OpenAI credits are exhausted.\n"


def test_main_rejects_nonpositive_dpi() -> None:
    """Reject a rendering resolution that cannot produce an image."""
    with pytest.raises(SystemExit) as error:
        cli.main(["--model=yolo11", "--input=data/example.pdf", "--dpi=0"])

    assert error.value.code == 2
