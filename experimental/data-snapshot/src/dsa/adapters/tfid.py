"""
TF-ID Large adapter -> Unified Evaluation Schema v1.3.

- Input: directory of PDFs
- Output: single JSON file matching data-snapshot-eval-v1.3.schema.json
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Sequence


from PIL.Image import Image
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from dsa.constants import INPUT_PDF_DIR, LABEL_MAP, ROOT
from dsa.utils import (
    convert_pdf_to_images,
    filter_small_predictions,
    normalize_bboxes_xyxy,
    utc_now_iso,
)

MODEL_ID_DEFAULT = "yifeihu/TF-ID-large"
OUTPUT_JSON_PATH = ROOT / "data/evaluation_input/tfid-large.json"


# TF-ID class names that map to our canonical labels.
# All other classes are ignored.
_LABEL_NORMALIZATION: dict[str, str] = {
    "figure": "Figure",
    "fig": "Figure",
    "chart": "Figure",
    "image": "Figure",
    "diagram": "Figure",
    "table": "Table",
    "tbl": "Table",
}

_ALLOWED_LABELS = set(LABEL_MAP.values())


def _coerce_label(label: Any) -> str | None:
    """Map a raw TF-ID label to a canonical label.

    Parameters
    ----------
    label : Any
        Raw label produced by the model.  May be a string, numeric id,
        or ``None``.

    Returns
    -------
    str | None
        ``"Figure"`` or ``"Table"`` if recognized, ``None`` otherwise.
        Returning ``None`` causes the detection to be silently skipped.
    """
    if label is None:
        return None

    if isinstance(label, (int, float)):
        key = str(int(label))
        return LABEL_MAP.get(key)

    s = str(label).strip()
    if not s:
        return None

    if s in _ALLOWED_LABELS:
        return s
    s_low = s.lower()
    if s_low in _LABEL_NORMALIZATION:
        return _LABEL_NORMALIZATION[s_low]

    return None


class ExtractSnapshot:
    """Wrapper around the TF-ID model for object detection on document pages.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g. ``"yifeihu/TF-ID-large"``).
    device : str
        Torch device string (e.g. ``"cpu"``, ``"cuda:0"``).
    """

    def __init__(self, model_id: str, device: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=device,
        )

    def tf_id_detection(self, images: list[Image] | Image) -> list[dict[str, Any]]:
        """Run TF-ID object detection on one or more PIL images.

        Parameters
        ----------
        images : list[Image] | Image
            A single PIL image or a list of PIL images.

        Returns
        -------
        list[dict[str, Any]]
            One annotation dict per image, each containing ``"bboxes"``,
            ``"labels"``, and optionally ``"scores"`` keys.
        """
        if not isinstance(images, list):
            images = [images]

        prompt = ["<OD>"] * len(images)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs.to(self.model.device)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )

        annotations: list[dict[str, Any]] = []
        for i in range(len(images)):
            annotation = self.processor.post_process_generation(
                generated_text[i],
                task="<OD>",
                image_size=(images[i].width, images[i].height),
            )
            annotations.append(annotation["<OD>"])

        return annotations


def _extract_lists(
    annotation: dict[str, Any],
) -> tuple[list[list[float]], list[Any], list[float] | None]:
    """Parse bboxes, labels, and scores from a TF-ID annotation dict.

    TF-ID typically returns::

        {
            "bboxes": [[x1, y1, x2, y2], ...],
            "labels": ["Figure", "Table", ...],
            "scores": [0.93, 0.88, ...]        # may be absent
        }

    This function defensively handles missing or mistyped keys.

    Parameters
    ----------
    annotation : dict[str, Any]
        Raw annotation dictionary from ``ExtractSnapshot.tf_id_detection``.

    Returns
    -------
    tuple[list[list[float]], list[Any], list[float] | None]
        ``(bboxes, labels, scores)`` where *scores* is ``None`` if absent.
    """
    bboxes = annotation.get("bboxes") or []
    labels = annotation.get("labels") or annotation.get("classes") or []
    scores = annotation.get("scores")

    if not isinstance(bboxes, list):
        bboxes = []
    if not isinstance(labels, list):
        labels = []

    if scores is not None and not isinstance(scores, list):
        scores = None

    return bboxes, labels, scores


class TFIDConfig:
    """Configuration for the TF-ID adapter.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    device : str
        Torch device string (e.g. ``"cpu"``, ``"cuda:0"``).
    dpi : int
        Resolution for PDF-to-image rendering.
    store_doc_path_as : str
        How to record document paths: ``"relative"`` or ``"absolute"``.
    filter_small : bool
        If ``True``, discard predictions whose normalised bounding-box area
        is below ``MIN_PREDICTION_AREA``.
    pdf_backend : {"pymupdf", "pdf2image"}
        Library to use for PDF-to-image rendering.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID_DEFAULT,
        device: str = "cpu",
        dpi: int = 300,
        store_doc_path_as: str = "relative",
        filter_small: bool = True,
        pdf_backend: str = "pymupdf",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dpi = dpi
        self.store_doc_path_as = store_doc_path_as
        self.filter_small = filter_small
        self.pdf_backend = pdf_backend


def run_tfid_adapter_directory(
    input_pdf_dir: str | Path,
    output_json_path: str | Path,
    *,
    run_id: str | None = None,
    config: TFIDConfig | None = None,
) -> Path:
    """Run TF-ID on every PDF in a directory and write predictions.

    Produces a single Unified Evaluation Schema v1.3 JSON file containing
    predictions for all pages across all PDFs.

    Parameters
    ----------
    input_pdf_dir : str | Path
        Directory containing PDF files (searched recursively).
    output_json_path : str | Path
        Destination path for the prediction JSON.
    run_id : str | None
        Optional identifier for this evaluation run. Auto-generated if not
        provided.
    config : TFIDConfig | None
        Adapter configuration. Uses defaults when ``None``.

    Returns
    -------
    Path
        The path where the prediction JSON was written.

    Raises
    ------
    FileNotFoundError
        If no PDF files are found under *input_pdf_dir*.
    """
    input_pdf_dir = Path(input_pdf_dir)
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = config or TFIDConfig()
    extractor = ExtractSnapshot(model_id=cfg.model_id, device=cfg.device)

    pdf_files = sorted(input_pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found under: {input_pdf_dir}")

    run_id = run_id or f"tfid-{uuid.uuid4().hex[:10]}"

    documents: list[dict[str, str]] = []
    predictions: list[dict[str, Any]] = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        doc_id = pdf_path.name

        if cfg.store_doc_path_as == "absolute":
            doc_path = str(pdf_path.resolve())
        else:
            doc_path = str(pdf_path.resolve().relative_to(ROOT))

        documents.append(
            {
                "doc_id": doc_id,
                "doc_name": pdf_path.name,
                "doc_path": doc_path,
            }
        )

        images = convert_pdf_to_images(
            pdf_path, dpi=cfg.dpi, backend=cfg.pdf_backend
        )

        for page_index, image in enumerate(
            tqdm(images, desc=f"Pages: {pdf_path.name}", leave=False)
        ):
            ann = extractor.tf_id_detection(image)[0]
            bboxes_raw, labels_raw, scores_raw = _extract_lists(ann)

            bboxes = normalize_bboxes_xyxy(
                bboxes_raw, width=image.width, height=image.height
            )

            # Align lengths defensively.
            n = len(bboxes)
            labels_raw = list(labels_raw)[:n] + [None] * max(0, n - len(labels_raw))
            if scores_raw is not None:
                scores_raw = list(scores_raw)[:n] + [None] * max(0, n - len(scores_raw))

            page_id = f"{doc_id}::p{page_index:03d}"
            objects: list[dict[str, Any]] = []

            for i in range(n):
                label = _coerce_label(labels_raw[i])
                if label is None:
                    continue

                score_val = 1.0
                if scores_raw is not None and scores_raw[i] is not None:
                    try:
                        score_val = float(scores_raw[i])
                    except Exception:
                        score_val = 1.0

                # Clip score to [0, 1].
                if score_val < 0.0:
                    score_val = 0.0
                elif score_val > 1.0:
                    score_val = 1.0

                objects.append(
                    {
                        "id": f"{page_id}:{i}",
                        "label": label,
                        "bbox": bboxes[i],
                        "score": score_val,
                    }
                )

            if cfg.filter_small:
                objects = filter_small_predictions(objects)

            if not objects:
                continue

            predictions.append(
                {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "page_index": page_index,
                    "objects": objects,
                }
            )

    out = {
        "label_map": LABEL_MAP,
        "info": {
            "schema_version": "1.3",
            "type": "prediction",
            "created_at": utc_now_iso(),
            "run_id": run_id,
            "model": {
                "name": "yifeihu/TF-ID-large",
                "version": "unknown",
                "notes": f"adapter=tfid; device={cfg.device}; dpi={cfg.dpi}",
            },
            "coordinate_system": {
                "type": "normalized_xyxy",
                "range": [0.0, 1.0],
                "origin": "top_left",
            },
        },
        "documents": documents,
        "predictions": predictions,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)

    return output_json_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run TF-ID over a PDF directory and produce a v1.3 prediction JSON."
    )
    parser.add_argument("--input_pdf_dir", type=str, default=str(INPUT_PDF_DIR))
    parser.add_argument("--output_json_path", type=str, default=str(OUTPUT_JSON_PATH))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--store_doc_path_as",
        type=str,
        choices=["relative", "absolute"],
        default="relative",
    )
    parser.add_argument("--model_id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument(
        "--filter_small_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter out predictions with very small bounding-box area.",
    )
    parser.add_argument(
        "--pdf_backend",
        type=str,
        choices=["pymupdf", "pdf2image"],
        default="pymupdf",
        help="PDF-to-image rendering backend (default: pymupdf).",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.input_pdf_dir)
    print(f"Input PDF dir : {pdf_dir}")
    pdf_files_found = list(pdf_dir.rglob("*.pdf"))
    print(f"PDFs found    : {len(pdf_files_found)}")

    cfg = TFIDConfig(
        model_id=args.model_id,
        device=args.device,
        dpi=args.dpi,
        store_doc_path_as=args.store_doc_path_as,
        filter_small=args.filter_small_predictions,
        pdf_backend=args.pdf_backend,
    )

    out_path = run_tfid_adapter_directory(
        args.input_pdf_dir,
        args.output_json_path,
        run_id=args.run_id,
        config=cfg,
    )
    print(f"Wrote: {out_path}")
