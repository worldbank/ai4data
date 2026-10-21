"""Detect figure and table snapshots with DocLayout-YOLO."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4data.data_snapshot.adapters.yolo26 import YOLO26Adapter, YOLO26Config

MODEL_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
MODEL_CACHE_DIR = Path(".cache/models")


@dataclass(frozen=True)
class DocLayoutYOLOConfig(YOLO26Config):
    """Configure DocLayout-YOLO inference with official defaults."""

    repo_id: str = MODEL_REPO_ID
    filename: str = MODEL_FILENAME
    cache_dir: str | Path = MODEL_CACHE_DIR
    conf: float = 0.2
    imgsz: int = 1024


class DocLayoutYOLOAdapter(YOLO26Adapter):
    """Convert DocLayout-YOLO predictions into canonical detections.

    Parameters
    ----------
    config : DocLayoutYOLOConfig | None
        Inference configuration. Defaults are used when omitted.
    model : Any | None
        Injected DocLayout-YOLO-compatible model, primarily for offline tests.
    """

    def __init__(
        self,
        config: DocLayoutYOLOConfig | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        resolved_config = config or DocLayoutYOLOConfig()
        super().__init__(
            resolved_config,
            model=model if model is not None else _load_model(resolved_config),
        )


def _load_model(config: DocLayoutYOLOConfig) -> Any:
    """Load DocLayout-YOLO from the local cache or Hugging Face."""
    from doclayout_yolo import YOLOv10

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
    return YOLOv10(str(model_path))
