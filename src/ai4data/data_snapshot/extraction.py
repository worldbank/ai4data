"""Extract data snapshot images from PDF files."""

import re
from math import ceil, floor
from pathlib import Path

from PIL import Image

from ai4data.data_snapshot.detection import Detection, SnapshotDetector
from ai4data.data_snapshot.inputs import discover_pdfs
from ai4data.data_snapshot.naming import build_snapshot_stem
from ai4data.data_snapshot.rendering import iter_pdf_pages


def extract_snapshots(
    input_path: str | Path,
    output_dir: str | Path,
    detector: SnapshotDetector,
    *,
    dpi: int = 300,
    force: bool = False,
) -> list[Path]:
    """Detect and save figure and table snapshots from PDF inputs.

    Parameters
    ----------
    input_path : str | Path
        One PDF file or a directory searched recursively for PDFs.
    output_dir : str | Path
        Root output directory. Images are written beneath ``snapshots``.
    detector : SnapshotDetector
        Page-image detector producing canonical detections.
    dpi : int, optional
        PyMuPDF rendering resolution in dots per inch.
    force : bool, optional
        Delete existing snapshots for each source PDF before extracting it.

    Returns
    -------
    list[Path]
        Available snapshot image paths in document, page, and spatial order.

    Raises
    ------
    FileExistsError
        If a destination snapshot appears while extraction is running.
    ValueError
        If multiple source PDFs have the same filename stem.
    """
    pdf_paths = discover_pdfs(input_path)
    validate_unique_stems(pdf_paths)
    snapshots_dir = Path(output_dir) / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for pdf_path in pdf_paths:
        existing_paths = _existing_snapshot_paths(snapshots_dir, pdf_path.stem)
        if existing_paths and not force:
            print(
                f"Skipping {pdf_path}: existing snapshots found in "
                f"{snapshots_dir} (use --force to replace)."
            )
            output_paths.extend(existing_paths)
            continue
        for existing_path in existing_paths:
            existing_path.unlink()

        counters = {"figure": 0, "table": 0}
        for _, page_image in iter_pdf_pages(pdf_path, dpi=dpi):
            detections = sorted(
                detector.detect(page_image),
                key=lambda item: (
                    item.bbox[1],
                    item.bbox[0],
                    item.bbox[3],
                    item.bbox[2],
                ),
            )
            for detection in detections:
                index = counters[detection.label]
                counters[detection.label] += 1
                output_path = snapshots_dir / (
                    f"{build_snapshot_stem(pdf_path, detection.label, index)}.png"
                )
                if output_path.exists():
                    raise FileExistsError(f"Snapshot already exists: {output_path}")
                _crop_snapshot(page_image, detection).save(output_path, format="PNG")
                output_paths.append(output_path)

    return output_paths


def _existing_snapshot_paths(snapshots_dir: Path, pdf_stem: str) -> list[Path]:
    """Return extraction outputs belonging to one source PDF stem."""
    pattern = re.compile(rf"{re.escape(pdf_stem)}_(?:figure|table)_\d{{3,}}\.png")
    return sorted(
        path
        for path in snapshots_dir.iterdir()
        if path.is_file() and pattern.fullmatch(path.name)
    )


def validate_unique_stems(pdf_paths: list[Path]) -> None:
    """Reject source documents that would share output filenames."""
    stems = [path.stem for path in pdf_paths]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise ValueError(f"Source PDF filename stems must be unique: {duplicates}")


def _crop_snapshot(image: Image.Image, detection: Detection) -> Image.Image:
    """Crop a normalized detection from its rendered page image."""
    x1, y1, x2, y2 = detection.bbox
    box = (
        floor(x1 * image.width),
        floor(y1 * image.height),
        ceil(x2 * image.width),
        ceil(y2 * image.height),
    )
    return image.crop(box)
