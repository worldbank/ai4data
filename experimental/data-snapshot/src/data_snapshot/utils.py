"""
Shared utility functions for the data-snapshot package.

Provides common helpers used across adapter modules and evaluation tools.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image

from data_snapshot.constants import MIN_PREDICTION_AREA


def convert_pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 300,
    backend: Literal["pymupdf", "pdf2image"] = "pymupdf",
) -> list[Image.Image]:
    """Convert each page of a PDF file to a PIL image.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.
    dpi : int
        Resolution for rendering, in dots per inch.
    backend : {"pymupdf", "pdf2image"}
        Library to use for PDF rendering.  ``"pymupdf"`` (default) is
        faster and requires no system dependencies.  ``"pdf2image"``
        delegates to poppler via the ``pdf2image`` package.

    Returns
    -------
    list[Image.Image]
        One PIL RGB image per page, in document order.

    Raises
    ------
    ValueError
        If *backend* is not one of the supported values.
    ImportError
        If the requested backend package is not installed.
    """
    if backend == "pymupdf":
        try:
            import pymupdf  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "pymupdf is required for the 'pymupdf' backend. "
                "Install it with: pip install pymupdf"
            ) from exc

        doc = pymupdf.open(str(pdf_path))
        images: list[Image.Image] = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            images.append(pix.pil_image())
        doc.close()
        return images

    if backend == "pdf2image":
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise ImportError(
                "pdf2image is required for the 'pdf2image' backend. "
                "Install it with: pip install pdf2image"
            ) from exc

        return convert_from_path(str(pdf_path), dpi=dpi)

    raise ValueError(
        f"Unknown PDF backend: {backend!r}. "
        f"Supported values: 'pymupdf', 'pdf2image'."
    )


def load_json(path: str | Path) -> dict:
    """Load and return a JSON file as a dictionary.

    Parameters
    ----------
    path : str | Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON contents.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp01(x: float) -> float:
    """Clamp a numeric value to the ``[0, 1]`` range.

    Parameters
    ----------
    x : float
        Value to clamp.

    Returns
    -------
    float
        Clamped value.
    """
    if x < 0.0:
        return 0.0
    elif x > 1.0:
        return 1.0
    else:
        return x


def sanitize_bbox(b: list[float]) -> tuple[float, float, float, float]:
    """Validate and clamp a bounding box to ``[0, 1]``.

    Parameters
    ----------
    b : list[float]
        A four-element list ``[x1, y1, x2, y2]``.

    Returns
    -------
    tuple[float, float, float, float]
        Clamped ``(x1, y1, x2, y2)``.

    Raises
    ------
    ValueError
        If *b* is not a four-element list.
    """
    if not (isinstance(b, list) and len(b) == 4):
        raise ValueError(f"Invalid bbox: {b}")
    x1, y1, x2, y2 = map(float, b)
    return (clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2))


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with ``Z`` suffix.

    Returns
    -------
    str
        Timestamp in the form ``"2026-03-26T07:00:00Z"``.
    """
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def normalize_bboxes_xyxy(
    bboxes: Sequence[Sequence[float]], width: int, height: int
) -> list[list[float]]:
    """Convert bounding boxes to normalized ``[0, 1]`` xyxy format.

    Absolute-pixel coordinates (detected via a ``> 1.5`` heuristic) are
    divided by the image dimensions.  All coordinates are clamped to
    ``[0, 1]`` and reordered so that ``x1 < x2`` and ``y1 < y2``.
    Degenerate boxes (zero area after clamping) are silently dropped.

    Parameters
    ----------
    bboxes : Sequence[Sequence[float]]
        Raw bounding boxes, each as ``[x1, y1, x2, y2]``.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Returns
    -------
    list[list[float]]
        Normalized bounding boxes that passed validation.
    """
    out: list[list[float]] = []
    for bb in bboxes:
        if len(bb) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bb)

        # Heuristic: if any coord > 1.5, assume absolute pixels.
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
            x1, x2 = x1 / float(width), x2 / float(width)
            y1, y2 = y1 / float(height), y2 / float(height)

        nx1, nx2 = sorted([clamp01(x1), clamp01(x2)])
        ny1, ny2 = sorted([clamp01(y1), clamp01(y2)])

        if nx2 <= nx1 or ny2 <= ny1:
            continue  # degenerate

        out.append([nx1, ny1, nx2, ny2])
    return out


def filter_small_predictions(
    objects: list[dict[str, Any]],
    min_area: float = MIN_PREDICTION_AREA,
) -> list[dict[str, Any]]:
    """Remove predictions whose normalized bounding-box area is below a threshold.

    This filters out spurious, near-zero-area detections that are unlikely to
    represent meaningful page elements.

    Parameters
    ----------
    objects : list[dict[str, Any]]
        Prediction objects, each containing a ``"bbox"`` key with normalised
        ``[x1, y1, x2, y2]`` coordinates in the ``[0, 1]`` range.
    min_area : float
        Minimum normalised area (width × height) a prediction must have to be
        kept.

    Returns
    -------
    list[dict[str, Any]]
        Filtered list retaining only predictions whose area ≥ *min_area*.
    """
    kept: list[dict[str, Any]] = []
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        area = abs(x2 - x1) * abs(y2 - y1)
        if area >= min_area:
            kept.append(obj)
    return kept
