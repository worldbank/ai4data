"""Run snapshot extraction with one explicitly selected model."""

import re
from pathlib import Path

from tqdm.auto import tqdm

from ai4data.data_snapshot.adapters import create_adapter
from ai4data.data_snapshot.extraction import extract_snapshots, validate_unique_stems
from ai4data.data_snapshot.inputs import discover_pdfs
from ai4data.data_snapshot.metadata_extraction import extract_metadata_files


def run_extraction(
    input_path: str | Path,
    output_dir: str | Path,
    model_name: str = "doclayoutyolo",
    *,
    dpi: int = 300,
    force: bool = False,
) -> list[Path]:
    """Extract snapshots beneath a directory named for the selected model.

    Parameters
    ----------
    input_path : str | Path
        One PDF file or a directory searched recursively for PDFs.
    output_dir : str | Path
        Root output directory.
    model_name : str, optional
        Identifier of the single snapshot extractor (default: doclayoutyolo).
    dpi : int, optional
        PyMuPDF rendering resolution in dots per inch.
    force : bool, optional
        Delete existing snapshots and metadata for each source PDF before
        extracting it.

    Returns
    -------
    list[Path]
        Available paths beneath ``output_dir/model_name/snapshots``.

    Raises
    ------
    ValueError
        If the model identifier is not implemented.
    """
    normalized_name = model_name.strip().lower()
    if normalized_name not in {"doclayoutyolo", "tfid", "yolo11", "yolo26"}:
        raise ValueError(f"Unsupported model: {model_name}")
    pdf_paths = discover_pdfs(input_path)
    validate_unique_stems(pdf_paths)
    model_dir = Path(output_dir) / normalized_name
    pending_pdf_paths = _prepare_pdf_paths(pdf_paths, model_dir, force=force)
    if not pending_pdf_paths:
        return []

    detector = create_adapter(normalized_name)
    output_paths: list[Path] = []
    pdf_progress = tqdm(pending_pdf_paths, desc="PDFs", unit="pdf")
    for pdf_path in pdf_progress:
        pdf_progress.set_postfix_str(pdf_path.name)
        output_paths.extend(extract_snapshots(pdf_path, model_dir, detector, dpi=dpi))
    return output_paths


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    model_name: str = "doclayoutyolo",
    *,
    dpi: int = 300,
    force: bool = False,
    metadata_provider: str = "openrouter",
) -> list[Path]:
    """Run snapshot extraction followed by semantic metadata extraction.

    Parameters
    ----------
    input_path : str | Path
        One PDF file or a directory searched recursively for PDFs.
    output_dir : str | Path
        Root output directory.
    model_name : str, optional
        Identifier of the single snapshot extractor (default: doclayoutyolo).
    dpi : int, optional
        PyMuPDF rendering resolution in dots per inch.
    force : bool, optional
        Replace outputs belonging to each source PDF.
    metadata_provider : str, optional
        API provider used for semantic metadata extraction.

    Returns
    -------
    list[Path]
        Snapshot paths followed by metadata paths.

    Raises
    ------
    RuntimeError
        If requested metadata extraction or validation fails.
    ValueError
        If the model identifier or input PDF set is invalid.
    """
    normalized_name = model_name.strip().lower()
    snapshot_paths = run_extraction(
        input_path,
        output_dir,
        normalized_name,
        dpi=dpi,
        force=force,
    )
    metadata_paths = extract_metadata_files(
        snapshot_paths,
        Path(output_dir) / normalized_name / "metadata",
        metadata_provider=metadata_provider,
    )
    return [*snapshot_paths, *metadata_paths]


def _prepare_pdf_paths(
    pdf_paths: list[Path], model_dir: Path, *, force: bool
) -> list[Path]:
    """Skip or clean existing outputs and return PDFs ready to process."""
    pending_pdf_paths: list[Path] = []
    for pdf_path in pdf_paths:
        existing_paths = _existing_output_paths(model_dir, pdf_path.stem)
        if not existing_paths:
            pending_pdf_paths.append(pdf_path)
            continue
        if not force:
            print(
                f"Existing output files for {pdf_path.name} exist. "
                "Skipped data snapshot and metadata extraction."
            )
            continue
        print(
            f"Existing output files for {pdf_path.name} exist. "
            "Cleaning up before continuing...",
            end=" ",
            flush=True,
        )
        for existing_path in existing_paths:
            existing_path.unlink()
        print("Done!")
        pending_pdf_paths.append(pdf_path)
    return pending_pdf_paths


def _existing_output_paths(model_dir: Path, pdf_stem: str) -> list[Path]:
    """Return exact crop and metadata outputs for one PDF stem."""
    pattern = re.compile(rf"{re.escape(pdf_stem)}_(?:figure|table)_\d{{3,}}")
    output_paths: list[Path] = []
    for directory, suffix in (
        (model_dir / "snapshots", ".png"),
        (model_dir / "metadata", ".json"),
    ):
        if directory.is_dir():
            output_paths.extend(
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() == suffix
                and pattern.fullmatch(path.stem)
            )
    return sorted(output_paths)
