"""Submit extracted data snapshots and metadata."""

from pathlib import Path


def submit_snapshot(snapshot_path: str | Path, metadata_path: str | Path) -> None:
    """Print the data snapshot and metadata paths that would be submitted.

    Parameters
    ----------
    snapshot_path : str | Path
        Path to the extracted data snapshot image.
    metadata_path : str | Path
        Path to the corresponding metadata JSON file.
    """
    # Submission is not yet implemented; this mock only prints the paths.
    print(f"Would submit {snapshot_path} with {metadata_path}")
