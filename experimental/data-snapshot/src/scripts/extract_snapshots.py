"""Extract PNG snapshots from PDF pages using bounding boxes.

Reads a JSON file (ground truth or prediction) conforming to the Unified
Evaluation Schema v1.3, renders the corresponding PDF pages, and crops each
annotated bounding box into an individual PNG file.

Output filename pattern::

    {doc_name_stem}_{label}_{counter:03d}.png

where *label* is lowercased (``figure`` or ``table``) and the counter is
zero-based, per label per document.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tqdm.auto import tqdm

from dsa.constants import INPUT_PDF_DIR
from dsa.utils import convert_pdf_to_images, load_json


def _group_pages_by_doc(
    predictions: list[dict],
) -> dict[str, dict[int, dict]]:
    """Group page entries by ``doc_id`` and ``page_index``.

    Parameters
    ----------
    predictions : list[dict]
        The ``predictions`` array from the evaluation schema JSON.

    Returns
    -------
    dict[str, dict[int, dict]]
        Mapping ``{doc_id: {page_index: page_entry}}``.
    """
    out: dict[str, dict[int, dict]] = {}
    for page in predictions:
        out.setdefault(page["doc_id"], {})[page["page_index"]] = page
    return out


def _build_doc_name_map(documents: list[dict]) -> dict[str, str]:
    """Build a mapping from ``doc_id`` to ``doc_name``.

    Parameters
    ----------
    documents : list[dict]
        The ``documents`` array from the evaluation schema JSON.

    Returns
    -------
    dict[str, str]
        Mapping ``{doc_id: doc_name}``.
    """
    return {doc["doc_id"]: doc["doc_name"] for doc in documents}


def extract_snapshots(
    input_json_path: str | Path,
    input_pdf_dir: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
    pdf_backend: str = "pymupdf",
) -> None:
    """Extract cropped snapshot PNGs from PDFs based on schema bounding boxes.

    Parameters
    ----------
    input_json_path : str | Path
        Path to a JSON file following the Unified Evaluation Schema v1.3
        (ground truth or prediction).
    input_pdf_dir : str | Path
        Directory containing the source PDF files referenced by ``doc_name``.
    output_dir : str | Path
        Directory where cropped snapshot PNGs will be saved.
    dpi : int
        Resolution for PDF-to-image rendering, in dots per inch.
    pdf_backend : str
        PDF rendering backend (``"pymupdf"`` or ``"pdf2image"``).
    """
    data = load_json(input_json_path)
    documents = data["documents"]
    pages_by_doc = _group_pages_by_doc(data["predictions"])
    doc_name_map = _build_doc_name_map(documents)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    missing_pdfs: list[str] = []
    docs_with_no_objects = 0
    total_snapshots = 0

    for doc in tqdm(documents, desc="Extracting snapshots"):
        doc_id = doc["doc_id"]
        doc_name = doc["doc_name"]
        pdf_path = Path(input_pdf_dir) / doc_name

        if not pdf_path.exists():
            missing_pdfs.append(str(pdf_path))
            continue

        doc_pages = pages_by_doc.get(doc_id, {})

        # Collect all objects across pages for this document
        all_objects: list[tuple[int, dict]] = []
        for page_index, page_entry in sorted(doc_pages.items()):
            for obj in page_entry.get("objects", []):
                all_objects.append((page_index, obj))

        if not all_objects:
            docs_with_no_objects += 1
            continue

        # Render PDF pages (only once per document)
        images = convert_pdf_to_images(pdf_path, dpi=dpi, backend=pdf_backend)

        doc_stem = Path(doc_name).stem

        # Per-label counters for this document
        label_counters: dict[str, int] = {}

        for page_index, obj in all_objects:
            if page_index >= len(images):
                print(
                    f"[WARN] Page index {page_index} out of range for "
                    f"{doc_name} ({len(images)} pages). Skipping object."
                )
                continue

            page_img = images[page_index]
            w, h = page_img.size

            x1, y1, x2, y2 = obj["bbox"]
            px1 = int(x1 * w)
            py1 = int(y1 * h)
            px2 = int(x2 * w)
            py2 = int(y2 * h)

            cropped = page_img.crop((px1, py1, px2, py2))

            label = obj["label"].lower()
            counter = label_counters.get(label, 0)
            label_counters[label] = counter + 1

            filename = f"{doc_stem}_{label}_{counter:03d}.png"
            cropped.save(output_path / filename)
            total_snapshots += 1

    # --- Summary -----------------------------------------------------------
    print(f"\nExtraction complete.")
    print(f"  Total snapshots saved: {total_snapshots}")

    if docs_with_no_objects:
        print(f"  Documents with no objects (skipped): {docs_with_no_objects}")

    if missing_pdfs:
        print(f"\n  PDF files not found ({len(missing_pdfs)}):")
        for p in missing_pdfs:
            print(f"    - {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Extract PNG snapshots from PDF pages based on bounding boxes "
            "in a JSON file following the Unified Evaluation Schema v1.3."
        ),
    )
    ap.add_argument(
        "--input_json_path",
        type=str,
        help="Path to the input JSON file (ground truth or prediction).",
    )
    ap.add_argument(
        "--input_pdf_dir",
        type=str,
        default=str(INPUT_PDF_DIR),
        help="Directory containing the source PDF files.",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where cropped snapshot PNGs will be saved.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF-to-image rendering (default: 300).",
    )
    ap.add_argument(
        "--pdf_backend",
        type=str,
        choices=["pymupdf", "pdf2image"],
        default="pymupdf",
        help="PDF-to-image rendering backend (default: pymupdf).",
    )
    args = ap.parse_args()

    extract_snapshots(
        input_json_path=args.input_json_path,
        input_pdf_dir=args.input_pdf_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        pdf_backend=args.pdf_backend,
    )
