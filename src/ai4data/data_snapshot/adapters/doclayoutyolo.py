"""Detect figure and table snapshots with DocLayout-YOLO."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai4data.data_snapshot.adapters.yolo26 import YOLO26Adapter, YOLO26Config
from ai4data.data_snapshot.model_config import get_model_setting


@dataclass(frozen=True)
class DocLayoutYOLOConfig(YOLO26Config):
    """Configure DocLayout-YOLO inference with official defaults.

    Notes
    -----
    Inherits the YOLO26 configuration parameters. Omitted model location fields
    use ``AI4DATA_SNAPSHOT_DOCLAYOUTYOLO_*`` environment variables at construction
    time. Explicit arguments take precedence.
    """

    repo_id: str = field(
        default_factory=lambda: get_model_setting("doclayoutyolo", "repo_id")
    )
    filename: str = field(
        default_factory=lambda: get_model_setting("doclayoutyolo", "filename")
    )
    cache_dir: str | Path = field(
        default_factory=lambda: Path(get_model_setting("doclayoutyolo", "cache_dir"))
    )
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
