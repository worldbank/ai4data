"""Detect figure and table snapshots with YOLO26 Document Layout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ai4data.data_snapshot.detection import Detection

MODEL_REPO_ID = "Armaggheddon/yolo26-document-layout"
MODEL_FILENAME = "yolo26m_doc_layout.pt"
MODEL_CACHE_DIR = Path(".cache/models")

_LABELS = {
    "figure": "figure",
    "picture": "figure",
    "table": "table",
}


@dataclass(frozen=True)
class YOLO26Config:
    """Configure YOLO26 Document Layout inference.

    Parameters
    ----------
    model_path : str | Path | None
        Optional local model weights. The configured Hugging Face model is
        downloaded when omitted.
    repo_id : str
        Hugging Face model repository identifier.
    filename : str
        Model weights filename in the repository.
    cache_dir : str | Path
        Repository-local directory used to cache downloaded model weights.
    device : str
        Ultralytics device identifier, such as ``"cpu"`` or ``"cuda:0"``.
    conf : float
        Minimum confidence passed to YOLO prediction.
    iou : float
        Intersection-over-union threshold used for non-maximum suppression.
    imgsz : int
        YOLO inference image size.
    min_area : float
        Minimum normalized bounding-box area retained as a snapshot.

    Raises
    ------
    ValueError
        If a numeric configuration value is outside its supported range.
    """

    model_path: str | Path | None = None
    repo_id: str = MODEL_REPO_ID
    filename: str = MODEL_FILENAME
    cache_dir: str | Path = MODEL_CACHE_DIR
    device: str = "cpu"
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 1280
    min_area: float = 0.008

    def __post_init__(self) -> None:
        """Validate inference settings before loading the model."""
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("Confidence threshold must be between zero and one.")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("IoU threshold must be between zero and one.")
        if self.imgsz <= 0:
            raise ValueError("Inference image size must be greater than zero.")
        if not 0.0 <= self.min_area <= 1.0:
            raise ValueError("Minimum area must be between zero and one.")


class YOLO26Adapter:
    """Convert YOLO26 model predictions into canonical detections.

    Parameters
    ----------
    config : YOLO26Config | None
        Inference configuration. Defaults are used when omitted.
    model : Any | None
        Injected Ultralytics-compatible model, primarily for offline tests.
    """

    def __init__(
        self, config: YOLO26Config | None = None, *, model: Any | None = None
    ) -> None:
        self.config = config or YOLO26Config()
        self._model = model if model is not None else _load_model(self.config)

    def detect(self, image: Image.Image) -> list[Detection]:
        """Detect figure and table snapshots in one rendered PDF page.

        Parameters
        ----------
        image : PIL.Image.Image
            RGB page image produced by the pipeline renderer.

        Returns
        -------
        list[Detection]
            Canonical detections in model output order.
        """
        results = self._model.predict(
            image,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        detections: list[Detection] = []

        for box, score, class_index in zip(boxes, scores, classes):
            label = _coerce_label(_class_name(result.names, int(class_index)))
            bbox = _normalize_bbox(box, image.width, image.height)
            if label is None or bbox is None:
                continue
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < self.config.min_area:
                continue
            detections.append(
                Detection(label=label, bbox=bbox, score=_clamp(float(score)))
            )

        return detections


def _load_model(config: YOLO26Config) -> Any:
    """Load YOLO26 from a local path or Hugging Face."""
    from ultralytics import YOLO

    if config.model_path is not None:
        model_path = Path(config.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
    else:
        model_path = Path(config.cache_dir) / config.filename
        if not model_path.is_file():
            from huggingface_hub import hf_hub_download

            model_path = Path(
                hf_hub_download(
                    repo_id=config.repo_id,
                    filename=config.filename,
                    repo_type="model",
                    local_dir=config.cache_dir,
                )
            )
    return YOLO(str(model_path))


def _class_name(names: Mapping[int, str] | Sequence[str], index: int) -> str | None:
    """Return a class name from Ultralytics mapping or sequence metadata."""
    if isinstance(names, Mapping):
        return names.get(index)
    if 0 <= index < len(names):
        return names[index]
    return None


def _coerce_label(raw_label: str | None) -> str | None:
    """Normalize one YOLO class name to a supported snapshot label."""
    if raw_label is None:
        return None
    return _LABELS.get(raw_label.strip().lower())


def _normalize_bbox(
    box: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Normalize and clamp one pixel-space YOLO bounding box."""
    if len(box) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    nx1, nx2 = sorted((_clamp(x1 / width), _clamp(x2 / width)))
    ny1, ny2 = sorted((_clamp(y1 / height), _clamp(y2 / height)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def _clamp(value: float) -> float:
    """Clamp a floating-point value to the inclusive unit interval."""
    return max(0.0, min(1.0, value))
