"""Launch the installed data snapshot metadata reviewer on localhost."""

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

from ai4data.data_snapshot.metadata_reviewer import _parse_args


def main(argv: list[str] | None = None) -> int:
    """Launch the local Streamlit app with the requested input and output paths.

    Parameters
    ----------
    argv : list[str] | None, optional
        Directory arguments, excluding the executable name. Defaults to sys.argv.

    Returns
    -------
    int
        Streamlit's process exit code.

    Raises
    ------
    SystemExit
        If arguments are invalid or the optional Streamlit dependency is missing.
    """
    paths = _parse_args(argv)
    if find_spec("streamlit") is None:
        raise SystemExit(
            'Install the reviewer with: pip install "ai4data[data-snapshot-metadata-reviewer]"'
        )
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).with_name("app.py")),
        "--server.address=127.0.0.1",
        "--server.showEmailPrompt=false",
        "--browser.gatherUsageStats=false",
        "--",
        "--snapshots-root",
        str(paths.snapshots_root.resolve()),
        "--generated-root",
        str(paths.generated_root.resolve()),
        "--reviewed-root",
        str(paths.reviewed_root.resolve()),
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
