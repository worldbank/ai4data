"""Detect figure and table snapshots with YOLO11 Document Layout."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4data.data_snapshot.adapters.yolo26 import YOLO26Adapter, YOLO26Config

MODEL_REPO_ID = "Armaggheddon/yolo11-document-layout"
MODEL_FILENAME = "yolo11m_doc_layout.pt"
MODEL_CACHE_DIR = Path(".cache/models")


@dataclass(frozen=True)
class YOLO11Config(YOLO26Config):
    """Configure medium YOLO11 Document Layout inference."""

    repo_id: str = MODEL_REPO_ID
    filename: str = MODEL_FILENAME
    cache_dir: str | Path = MODEL_CACHE_DIR


class YOLO11Adapter(YOLO26Adapter):
    """Convert YOLO11 model predictions into canonical detections.

    Parameters
    ----------
    config : YOLO11Config | None
        Inference configuration. Defaults are used when omitted.
    model : Any | None
        Injected Ultralytics-compatible model, primarily for offline tests.
    """

    def __init__(
        self, config: YOLO11Config | None = None, *, model: Any | None = None
    ) -> None:
        super().__init__(config or YOLO11Config(), model=model)
