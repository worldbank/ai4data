"""Render PDF pages for detection and cropping with PyMuPDF."""

from collections.abc import Iterator
from pathlib import Path

import pymupdf
from PIL import Image
from tqdm.auto import tqdm


def iter_pdf_pages(
    pdf_path: str | Path, dpi: int = 300
) -> Iterator[tuple[int, Image.Image]]:
    """Yield RGB page images rendered by PyMuPDF in document order.

    Parameters
    ----------
    pdf_path : str | Path
        PDF file to render.
    dpi : int, optional
        Rendering resolution in dots per inch.

    Yields
    ------
    tuple[int, PIL.Image.Image]
        Zero-based page index and rendered RGB page image.

    Raises
    ------
    FileNotFoundError
        If the PDF path does not identify an existing file.
    ValueError
        If the path is not a PDF or ``dpi`` is not positive.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Input file is not a PDF: {path}")
    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")

    with pymupdf.open(path) as document:
        pages = tqdm(
            document,
            total=document.page_count,
            desc=f"Pages ({path.name})",
            unit="page",
            leave=False,
        )
        for page_index, page in enumerate(pages):
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            yield page_index, pixmap.pil_image().convert("RGB")
