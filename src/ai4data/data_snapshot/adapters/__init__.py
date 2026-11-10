"""Create snapshot detector adapters for data snapshot extraction."""

from ai4data.data_snapshot.detection import SnapshotDetector


def create_adapter(model_name: str, *, device: str = "cpu") -> SnapshotDetector:
    """Create the one snapshot detector selected for a pipeline run.

    Parameters
    ----------
    model_name : str
        Extractor identifier, currently ``"doclayoutyolo"``, ``"tfid"``,
        ``"yolo11"``, or ``"yolo26"``.
    device : str, optional
        Inference device, such as ``"cpu"`` or ``"cuda:0"``.

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
            DocLayoutYOLOConfig,
        )

        return DocLayoutYOLOAdapter(DocLayoutYOLOConfig(device=device))
    if model_name == "tfid":
        from ai4data.data_snapshot.adapters.tfid import TFIDAdapter, TFIDConfig

        return TFIDAdapter(TFIDConfig(device=device))
    if model_name == "yolo11":
        from ai4data.data_snapshot.adapters.yolo11 import YOLO11Adapter, YOLO11Config

        return YOLO11Adapter(YOLO11Config(device=device))
    if model_name == "yolo26":
        from ai4data.data_snapshot.adapters.yolo26 import YOLO26Adapter, YOLO26Config

        return YOLO26Adapter(YOLO26Config(device=device))
    raise ValueError(f"Unsupported model: {model_name}")
