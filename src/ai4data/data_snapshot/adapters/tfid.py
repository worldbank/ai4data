"""Detect figure and table snapshots with TF-ID Large."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ai4data.data_snapshot.adapters.yolo26 import _clamp, _normalize_bbox
from ai4data.data_snapshot.detection import Detection

MODEL_ID_DEFAULT = "yifeihu/TF-ID-large"
MODEL_REVISION_DEFAULT = "a91ffef47039768b2deff82681b75d77252703f2"
MODEL_CACHE_DIR = Path(".cache/models/tfid")

_LABELS = {
    "1": "figure",
    "2": "table",
    "chart": "figure",
    "diagram": "figure",
    "fig": "figure",
    "figure": "figure",
    "image": "figure",
    "table": "table",
    "tbl": "table",
}


@dataclass(frozen=True)
class TFIDConfig:
    """Configure TF-ID Large inference.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier.
    revision : str
        Pinned Hugging Face model revision containing the remote model code.
    cache_dir : str | Path
        Repository-local Hugging Face model cache.
    device : str
        Torch device string, such as ``"cpu"`` or ``"cuda:0"``.
    min_area : float
        Minimum normalized bounding-box area retained as a snapshot.

    Raises
    ------
    ValueError
        If the minimum area is outside the inclusive unit interval.
    """

    model_id: str = MODEL_ID_DEFAULT
    revision: str = MODEL_REVISION_DEFAULT
    cache_dir: str | Path = MODEL_CACHE_DIR
    device: str = "cpu"
    min_area: float = 0.008

    def __post_init__(self) -> None:
        """Validate TF-ID inference settings."""
        if not 0.0 <= self.min_area <= 1.0:
            raise ValueError("Minimum area must be between zero and one.")


class TFIDAdapter:
    """Convert TF-ID Large predictions into canonical detections.

    Parameters
    ----------
    config : TFIDConfig | None
        Inference configuration. Defaults are used when omitted.
    model : Any | None
        Injected Transformers-compatible model for offline tests.
    processor : Any | None
        Injected Transformers-compatible processor for offline tests.

    Raises
    ------
    ValueError
        If only one of ``model`` and ``processor`` is supplied.
    """

    def __init__(
        self,
        config: TFIDConfig | None = None,
        *,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.config = config or TFIDConfig()
        if (model is None) != (processor is None):
            raise ValueError("Model and processor must be supplied together.")
        if model is None:
            model, processor = _load_components(self.config)
        self._model = model
        self._processor = processor

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
        inputs = self._processor(text="<OD>", images=image, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        result = self._processor.post_process_generation(
            generated_text,
            task="<OD>",
            image_size=(image.width, image.height),
        ).get("<OD>", {})
        boxes = result.get("bboxes", [])
        labels = result.get("labels", result.get("classes", []))
        scores = result.get("scores")
        if not isinstance(boxes, list) or not isinstance(labels, list):
            return []

        detections: list[Detection] = []
        for index, (box, raw_label) in enumerate(zip(boxes, labels)):
            label = _coerce_label(raw_label)
            bbox = _normalize_bbox(box, image.width, image.height)
            if label is None or bbox is None:
                continue
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < self.config.min_area:
                continue
            score = _score_at(scores, index)
            detections.append(Detection(label=label, bbox=bbox, score=score))
        return detections


def _load_components(config: TFIDConfig) -> tuple[Any, Any]:
    """Load the pinned TF-ID model and processor from Hugging Face."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    common = {
        "cache_dir": config.cache_dir,
        "revision": config.revision,
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map=config.device,
        **common,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(config.model_id, **common)
    return model, processor


def _coerce_label(raw_label: Any) -> str | None:
    """Normalize one TF-ID label to a supported snapshot label."""
    if raw_label is None:
        return None
    if isinstance(raw_label, (int, float)):
        raw_label = int(raw_label)
    return _LABELS.get(str(raw_label).strip().lower())


def _score_at(scores: Any, index: int) -> float:
    """Return one optional TF-ID score, defaulting to full confidence."""
    if not isinstance(scores, list) or index >= len(scores):
        return 1.0
    try:
        return _clamp(float(scores[index]))
    except (TypeError, ValueError):
        return 1.0
