"""Test the placeholder data snapshot submission function."""

from pathlib import Path

from pytest import CaptureFixture

from ai4data.data_snapshot import submit_snapshot


def test_submit_snapshot_prints_paths(capsys: CaptureFixture[str]) -> None:
    """Print both paths without attempting an external submission."""
    submit_snapshot(
        Path("snapshots/report_figure_000.png"), "metadata/report_figure_000.json"
    )

    assert capsys.readouterr().out == (
        "Would submit snapshots/report_figure_000.png with "
        "metadata/report_figure_000.json\n"
    )
