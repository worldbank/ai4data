"""Visualization tool for comparing ground-truth and predicted bounding boxes.

Renders annotated page images with color-coded bounding boxes for
visual comparison of model predictions against ground truth.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tqdm.auto import tqdm

from dsa.constants import (
    VP_GT_JSON_PATH,
    VP_PRED_JSON_PATH,
    INPUT_PDF_DIR,
    GT_COLOR_BGR,
    PRED_COLOR_BGR,
    VP_OUTPUT_DIR,
)
from dsa.utils import convert_pdf_to_images, load_json


def _group_pages_by_doc(
    predictions: list[dict],
) -> dict[str, dict[int, dict]]:
    """Generate dictionary of doc_ids, page indices, and predictions.

    This function converts the list of predictions into the following format: `{doc_id: {page_index: page_entry}}`

    Parameters
    ----------
    predictions : list of dict
        List of predictions

    Returns
    -------
    dict
    """
    out = {}
    for page in predictions:
        out.setdefault(page["doc_id"], {})[page["page_index"]] = page
    return out


def convert_pdf_to_opencv_images(
    pdf_path: str | Path,
    backend: Literal["pymupdf", "pdf2image"] = "pymupdf",
) -> list[np.ndarray]:
    """Convert PDF to a list of OpenCV BGR images.

    Parameters
    ----------
    pdf_path : str or Path
        Path to PDF file
    backend : {"pymupdf", "pdf2image"}
        Library to use for PDF-to-image rendering.

    Returns
    -------
    List of ndarray objects
    """
    pil_pages = convert_pdf_to_images(pdf_path, backend=backend)
    images = []

    for pil_img in pil_pages:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        images.append(img)

    return images


def draw_objects(img, objects, color, source):
    """Draw bounding boxes on image using normalized coords.

    Parameters
    ----------
    img : ndarray
        OpenCV BGR image
    objects : list of dict
        List of bounding boxes
    color : tuple of int
        Bounding box color in BGR
    source : str
        Source to append to the bounding box label (e.g., "GT" or "Prediction")

    Returns
    -------
    ndarray
        OpenCV BGR image
    """
    H, W = img.shape[:2]

    for obj in objects:
        # Draw rectangle
        x1, y1, x2, y2 = obj["bbox"]
        pt1 = (int(x1 * W), int(y1 * H))  # Denormalize coordinates
        pt2 = (int(x2 * W), int(y2 * H))  # Denormalize coordinates
        cv2.rectangle(img, pt1, pt2, color, 2)

        # Add label
        text = f"{obj['label']} ({source})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        text_bg_tl = (pt1[0], pt1[1] - th - 6)
        text_bg_br = (pt1[0] + tw + 4, pt1[1])
        cv2.rectangle(img, text_bg_tl, text_bg_br, color, -1)
        cv2.putText(
            img,
            text,
            (pt1[0] + 2, pt1[1] - 4),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return img


def _should_render_page(
    has_gt: bool,
    has_pred: bool,
    mode: Literal["gt", "pred", "union", "all"],
) -> bool:
    """Determine whether a page should be rendered based on the mode.

    Parameters
    ----------
    has_gt : bool
        Whether the page has ground-truth objects.
    has_pred : bool
        Whether the page has prediction objects.
    mode : {"gt", "pred", "union", "all"}
        Page filtering mode.

    Returns
    -------
    bool
        True if the page should be rendered.
    """
    if mode == "gt":
        return has_gt
    if mode == "pred":
        return has_pred
    if mode == "union":
        return has_gt and has_pred
    return True


def visualize_snapshots(
    gt_json_path: str,
    pred_json_path: str,
    input_pdf_dir: str,
    output_dir: str,
    pdf_backend: str = "pymupdf",
    mode: Literal["gt", "pred", "union", "all"] = "all",
):
    """Render visualization PNGs.

    Output filename:
        ``{filename.pdf}_page_XXX.png``

    Parameters
    ----------
    gt_json_path : str or path
        Path to ground truth json file
    pred_json_path : str or path
        Path to prediction json file
    input_pdf_dir : str or path
        Path to directory of PDF files
    output_dir : str or path
        Path to save annotated pages
    pdf_backend : {"pymupdf", "pdf2image"}
        Library to use for PDF-to-image rendering.
    mode : {"gt", "pred", "union", "all"}
        Page filtering mode. ``"gt"`` renders only pages with ground-truth
        boxes, ``"pred"`` only pages with prediction boxes, ``"union"`` only
        pages that have both, and ``"all"`` (default) renders every page.
    """
    gt = load_json(gt_json_path)
    pr = load_json(pred_json_path)

    gt_pages = _group_pages_by_doc(gt["predictions"])
    pr_pages = _group_pages_by_doc(pr["predictions"])

    os.makedirs(output_dir, exist_ok=True)

    for doc in tqdm(gt["documents"]):
        doc_id = doc["doc_id"]
        doc_name = doc["doc_name"]
        pdf_path = Path(input_pdf_dir) / doc_name

        if not pdf_path.exists():
            print(f"[WARN] Missing PDF: {pdf_path}")
            continue

        gt_doc_pages = gt_pages.get(doc_id, {})
        pr_doc_pages = pr_pages.get(doc_id, {})
        images = convert_pdf_to_opencv_images(str(pdf_path), backend=pdf_backend)
        for page_index, img in enumerate(images):
            gt_page = gt_doc_pages.get(page_index)
            pr_page = pr_doc_pages.get(page_index)

            has_gt = bool(gt_page and gt_page["objects"])
            has_pred = bool(pr_page and pr_page["objects"])

            if not _should_render_page(has_gt, has_pred, mode):
                continue

            canvas = img.copy()

            # GT boxes
            if has_gt:
                canvas = draw_objects(
                    canvas,
                    gt_page["objects"],
                    GT_COLOR_BGR,
                    "GT",
                )

            # Pred boxes
            if has_pred:
                canvas = draw_objects(
                    canvas,
                    pr_page["objects"],
                    PRED_COLOR_BGR,
                    "Prediction",
                )

            # Save image
            out_name = f"{doc_name}_page_{page_index:03d}.png"
            out_path = Path(output_dir) / out_name
            cv2.imwrite(str(out_path), canvas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gt_json_path",
        help="Path to ground truth json file",
        default=VP_GT_JSON_PATH,
    )
    ap.add_argument(
        "--pred_json_path",
        help="Path to prediction json file",
        default=VP_PRED_JSON_PATH,
    )
    ap.add_argument(
        "--input_pdf_dir",
        default=INPUT_PDF_DIR,
        help="Path to directory of PDF files",
    )
    ap.add_argument(
        "--output_dir",
        default=VP_OUTPUT_DIR,
        help="Path to save annotated pages",
    )
    ap.add_argument(
        "--pdf_backend",
        type=str,
        choices=["pymupdf", "pdf2image"],
        default="pymupdf",
        help="PDF-to-image rendering backend (default: pymupdf).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["gt", "pred", "union", "all"],
        default="all",
        help=(
            "Page filtering mode: 'gt' = only pages with GT boxes, "
            "'pred' = only pages with prediction boxes, "
            "'union' = only pages with both, "
            "'all' = all pages (default)."
        ),
    )
    args = ap.parse_args()

    visualize_snapshots(
        gt_json_path=args.gt_json_path,
        pred_json_path=args.pred_json_path,
        input_pdf_dir=args.input_pdf_dir,
        output_dir=args.output_dir,
        pdf_backend=args.pdf_backend,
        mode=args.mode,
    )

    print("Visualization complete.")
