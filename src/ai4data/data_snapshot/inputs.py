"""Discover PDF inputs for the data snapshot pipeline."""

from pathlib import Path

_IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}


def discover_pdfs(input_path: str | Path) -> list[Path]:
    """Return PDF files represented by a file or directory input.

    Directory inputs are searched recursively and returned in deterministic
    path order.

    Parameters
    ----------
    input_path : str | Path
        A PDF file or directory containing PDF files.

    Returns
    -------
    list[Path]
        Discovered PDF paths.

    Raises
    ------
    FileNotFoundError
        If the input does not exist or a directory contains no PDF files.
    ValueError
        If a file input does not have a ``.pdf`` extension.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {path}")
        return [path]

    pdf_paths = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under: {path}")
    return pdf_paths


def discover_snapshot_images(input_path: str | Path) -> list[Path]:
    """Return supported snapshot images represented by a file or directory.

    Parameters
    ----------
    input_path : str | Path
        One snapshot image or a directory searched recursively for images.

    Returns
    -------
    list[Path]
        Discovered image paths in deterministic order.

    Raises
    ------
    FileNotFoundError
        If the input does not exist or a directory contains no images.
    ValueError
        If a file input does not use a supported image extension.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"Input file is not a supported image: {path}")
        return [path]

    image_paths = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No snapshot images found under: {path}")
    return image_paths
