"""Check the installed reviewer launcher without starting a server."""

import subprocess
import sys
from pathlib import Path

import pytest

from ai4data.data_snapshot.metadata_reviewer import _parse_args
from ai4data.data_snapshot.metadata_reviewer import __main__ as launcher


def test_directory_defaults_follow_snapshot_root() -> None:
    """Infer metadata and reviewed siblings for default and custom runs."""
    for root in ("output/doclayoutyolo/snapshots", "custom run/images"):
        paths = _parse_args(["--snapshots-root", root])
        assert paths.snapshots_root == Path(root)
        assert paths.generated_root == Path(root).parent / "metadata"
        assert paths.reviewed_root == Path(root).parent / "reviewed"


def test_launch_uses_installed_app_and_localhost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve paths with spaces and return the Streamlit process status."""
    commands = []

    def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        assert not check
        commands.append(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(launcher, "find_spec", lambda name: object())
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert (
        launcher.main(
            [
                "--snapshots-root",
                str(tmp_path / "snapshot images"),
                "--generated-root",
                str(tmp_path / "generated metadata"),
                "--reviewed-root",
                str(tmp_path / "reviewed metadata"),
            ]
        )
        == 7
    )
    command = commands[0]
    assert command[:3] == [sys.executable, "-m", "streamlit"]
    assert Path(command[4]).is_file()
    assert "--server.address=127.0.0.1" in command
    assert command[command.index("--") + 1 :] == [
        "--snapshots-root",
        str(tmp_path / "snapshot images"),
        "--generated-root",
        str(tmp_path / "generated metadata"),
        "--reviewed-root",
        str(tmp_path / "reviewed metadata"),
    ]


def test_missing_streamlit_explains_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the correct extra name when the optional UI is unavailable."""
    monkeypatch.setattr(launcher, "find_spec", lambda name: None)
    with pytest.raises(SystemExit, match="data-snapshot-metadata-reviewer"):
        launcher.main([])
