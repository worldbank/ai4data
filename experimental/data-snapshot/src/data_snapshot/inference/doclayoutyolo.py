"""
DocLayout-YOLO adapter -> Unified Evaluation Schema v1.3.

- Input: directory of PDFs
- Output: single JSON file matching data-snapshot-eval-v1.3.schema.json

Model: DocLayout-YOLO fine-tuned on DocStructBench
  - Local  : models/doclayout_yolo_docstructbench_imgsz1024.pt
  - Remote : juliozhao/DocLayout-YOLO-DocStructBench (HuggingFace repo ID)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

from tqdm.auto import tqdm

from data_snapshot.constants import INPUT_PDF_DIR, LABEL_MAP, MODELS_DIR, ROOT
from data_snapshot.utils import (
    convert_pdf_to_images,
    filter_small_predictions,
    normalize_bboxes_xyxy,
    utc_now_iso,
)

MODEL_NAME = "juliozhao/DocLayout-YOLO-DocStructBench"
MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
OUTPUT_JSON_PATH = ROOT / "data/evaluation_input/doclayoutyolo.json"
IMGSZ = 1024

# DocLayout-YOLO / DocStructBench class names that map to our canonical labels.
# All other classes (title, text, caption, formula, …) are ignored.
_LABEL_NORMALIZATION: dict[str, str] = {
    "figure": "Figure",
    "picture": "Figure",
    "table": "Table",
}
_ALLOWED_LABELS = set(LABEL_MAP.values())


def _coerce_label(raw: Any) -> str | None:
    """Map a raw YOLO class name to a canonical label.

    Parameters
    ----------
    raw : Any
        Raw class name produced by the model (e.g. ``"figure"``, ``"table"``).

    Returns
    -------
    str | None
        ``"Figure"`` or ``"Table"`` if recognized, ``None`` otherwise.
        Returning ``None`` causes the detection to be silently skipped.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in _ALLOWED_LABELS:
        return s
    s_low = s.lower()
    if s_low in _LABEL_NORMALIZATION:
        return _LABEL_NORMALIZATION[s_low]
    return None


class DocLayoutYOLOConfig:
    """Configuration for the DocLayout-YOLO adapter.

    Parameters
    ----------
    repo_id : str
        HuggingFace model repository ID.
    filename : str
        Model weight file name on the HF repo.
    device : str
        Torch device string (e.g. ``"cpu"``, ``"cuda:0"``).
    dpi : int
        Resolution for PDF-to-image rendering.
    conf : float
        Minimum confidence threshold for detections.
    imgsz : int
        Input image size passed to the YOLO model.
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
        repo_id: str = MODEL_NAME,
        filename: str = MODEL_FILENAME,
        device: str = "cpu",
        dpi: int = 300,
        conf: float = 0.2,
        imgsz: int = 1024,
        store_doc_path_as: str = "relative",
        filter_small: bool = True,
        pdf_backend: str = "pymupdf",
    ) -> None:
        self.repo_id = repo_id
        self.filename = filename
        self.device = device
        self.dpi = dpi
        self.conf = conf
        self.imgsz = imgsz
        self.store_doc_path_as = store_doc_path_as
        self.filter_small = filter_small
        self.pdf_backend = pdf_backend


def run_doclayout_yolo_adapter_directory(
    input_pdf_dir: str | Path,
    output_json_path: str | Path,
    *,
    run_id: str | None = None,
    config: DocLayoutYOLOConfig | None = None,
) -> Path:
    """Run DocLayout-YOLO on every PDF in a directory and write predictions.

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
    config : DocLayoutYOLOConfig | None
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

    cfg = config or DocLayoutYOLOConfig()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(
        repo_id=cfg.repo_id,
        filename=cfg.filename,
        repo_type="model",
        local_dir=str(MODELS_DIR),
    )
    model = YOLOv10(model_path)

    pdf_files = sorted(input_pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found under: {input_pdf_dir}")

    run_id = run_id or f"doclayout-yolo-{uuid.uuid4().hex[:10]}"

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

        images = convert_pdf_to_images(pdf_path, dpi=cfg.dpi, backend=cfg.pdf_backend)

        for page_index, image in enumerate(
            tqdm(images, desc=f"Pages: {pdf_path.name}", leave=False)
        ):
            det_res = model.predict(
                image,
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                device=cfg.device,
                verbose=False,
            )
            result = det_res[0]

            # Extract boxes, scores, class indices from the Results object.
            boxes_tensor = result.boxes.xyxy.cpu().tolist()
            scores_list = result.boxes.conf.cpu().tolist()
            cls_list = result.boxes.cls.cpu().tolist()
            names: dict[int, str] = result.names

            bboxes_norm = normalize_bboxes_xyxy(
                boxes_tensor, width=image.width, height=image.height
            )

            page_id = f"{doc_id}::p{page_index:03d}"
            objects: list[dict[str, Any]] = []

            for i, bbox in enumerate(bboxes_norm):
                cls_idx = int(cls_list[i])
                raw_name = names.get(cls_idx, "")
                label = _coerce_label(raw_name)
                if label is None:
                    continue  # skip non-Figure/Table classes

                score = float(scores_list[i])
                score = max(0.0, min(1.0, score))

                objects.append(
                    {
                        "id": f"{page_id}:{i}",
                        "label": label,
                        "bbox": bbox,
                        "score": score,
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
                "name": cfg.repo_id,
                "version": cfg.filename,
                "notes": (
                    f"adapter=doclayout_yolo; "
                    f"device={cfg.device}; dpi={cfg.dpi}; "
                    f"conf={cfg.conf}; imgsz={cfg.imgsz}"
                ),
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
        description="Run DocLayout-YOLO over a PDF directory and produce a v1.3 prediction JSON."
    )
    parser.add_argument(
        "--input_pdf_dir",
        type=str,
        default=str(INPUT_PDF_DIR),
        help="Directory of PDF files to process.",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        default=str(OUTPUT_JSON_PATH),
        help="Destination path for the prediction JSON.",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument(
        "--store_doc_path_as",
        type=str,
        choices=["relative", "absolute"],
        default="relative",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace repo ID for the DocLayout-YOLO model.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=MODEL_FILENAME,
        help="Model weight file name on the HF repo.",
    )
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
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    print(f"PDFs found    : {len(pdf_files)}")

    cfg = DocLayoutYOLOConfig(
        repo_id=args.repo_id,
        filename=args.filename,
        device=args.device,
        dpi=args.dpi,
        conf=args.conf,
        imgsz=args.imgsz,
        store_doc_path_as=args.store_doc_path_as,
        filter_small=args.filter_small_predictions,
        pdf_backend=args.pdf_backend,
    )

    out_path = run_doclayout_yolo_adapter_directory(
        args.input_pdf_dir,
        args.output_json_path,
        run_id=args.run_id,
        config=cfg,
    )
    print(f"Wrote: {out_path}")
