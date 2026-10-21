"""Create snapshot detector adapters for data snapshot extraction."""

from ai4data.data_snapshot.detection import SnapshotDetector


def create_adapter(model_name: str) -> SnapshotDetector:
    """Create the one snapshot detector selected for a pipeline run.

    Parameters
    ----------
    model_name : str
        Extractor identifier, currently ``"doclayoutyolo"``, ``"tfid"``,
        ``"yolo11"``, or ``"yolo26"``.

    Returns
    -------
    SnapshotDetector
        Selected snapshot detector adapter.

    Raises
    ------
    ValueError
        If the model identifier is not implemented.
    """
    if model_name == "doclayoutyolo":
        from ai4data.data_snapshot.adapters.doclayoutyolo import (
            DocLayoutYOLOAdapter,
        )

        return DocLayoutYOLOAdapter()
    if model_name == "tfid":
        from ai4data.data_snapshot.adapters.tfid import TFIDAdapter

        return TFIDAdapter()
    if model_name == "yolo11":
        from ai4data.data_snapshot.adapters.yolo11 import YOLO11Adapter

        return YOLO11Adapter()
    if model_name == "yolo26":
        from ai4data.data_snapshot.adapters.yolo26 import YOLO26Adapter

        return YOLO26Adapter()
    raise ValueError(f"Unsupported model: {model_name}")
