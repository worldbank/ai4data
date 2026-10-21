"""Write validated metadata JSON for extracted snapshot images."""

from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from ai4data.data_snapshot.metadata_extraction.extraction import (
    extract_metadata,
    get_extraction_config_path,
)


def extract_metadata_files(
    snapshot_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    force: bool = False,
    metadata_provider: str = "openrouter",
) -> list[Path]:
    """Extract and save metadata for snapshot images.

    Parameters
    ----------
    snapshot_paths : Iterable[str | Path]
        Snapshot images to process.
    output_dir : str | Path
        Directory for one JSON file per snapshot.
    force : bool, optional
        Delete and replace existing metadata for each snapshot.
    metadata_provider : str, optional
        API provider whose packaged request settings are used.

    Returns
    -------
    list[Path]
        Available metadata JSON paths in snapshot order.

    Raises
    ------
    RuntimeError
        If metadata extraction or validation fails for a snapshot.
    """
    metadata_dir = Path(output_dir)
    config_path = get_extraction_config_path(metadata_provider)
    output_paths: list[Path] = []
    paths = [Path(path) for path in snapshot_paths]

    snapshot_progress = tqdm(paths, desc="Metadata", unit="snapshot")
    for snapshot_path in snapshot_progress:
        snapshot_progress.set_postfix_str(snapshot_path.name)
        metadata_path = metadata_dir / f"{snapshot_path.stem}.json"
        output_paths.append(metadata_path)
        if metadata_path.exists():
            if force:
                print(
                    f"Existing metadata output for {snapshot_path.name} exists. "
                    "Cleaning up before continuing...",
                    end=" ",
                    flush=True,
                )
                metadata_path.unlink()
                print("Done!")
            else:
                print(
                    f"Existing metadata output for {snapshot_path.name} exists. "
                    "Skipped metadata extraction."
                )
                continue

        result = extract_metadata(snapshot_path, config_path=config_path)
        if result.metadata is None:
            detail = result.error or "unknown extraction error"
            raise RuntimeError(
                f"Metadata extraction failed for {snapshot_path}: {detail}"
            )
        metadata_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            result.metadata.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)

    return output_paths
