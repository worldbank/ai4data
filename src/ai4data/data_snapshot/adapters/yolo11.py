"""Detect figure and table snapshots with YOLO11 Document Layout."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai4data.data_snapshot.adapters.yolo26 import YOLO26Adapter, YOLO26Config
from ai4data.data_snapshot.model_config import get_model_setting


@dataclass(frozen=True)
class YOLO11Config(YOLO26Config):
    """Configure medium YOLO11 Document Layout inference.

    Notes
    -----
    Inherits the YOLO26 configuration parameters. Omitted model location fields
    use ``AI4DATA_SNAPSHOT_YOLO11_*`` environment variables at construction time.
    Explicit arguments take precedence.
    """

    repo_id: str = field(default_factory=lambda: get_model_setting("yolo11", "repo_id"))
    filename: str = field(
        default_factory=lambda: get_model_setting("yolo11", "filename")
    )
    cache_dir: str | Path = field(
        default_factory=lambda: Path(get_model_setting("yolo11", "cache_dir"))
    )


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
