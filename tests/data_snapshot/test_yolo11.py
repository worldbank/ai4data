"""Test the YOLO11 adapter without loading model weights."""

from pathlib import Path

from ai4data.data_snapshot.adapters.yolo11 import YOLO11Adapter, YOLO11Config


def test_yolo11_adapter_uses_medium_weights_in_local_cache() -> None:
    """Keep medium YOLO11 weights in the repository-local ignored cache."""
    model = object()
    adapter = YOLO11Adapter(model=model)

    assert adapter._model is model
    assert adapter.config == YOLO11Config()
    assert adapter.config.filename == "yolo11m_doc_layout.pt"
    assert Path(adapter.config.cache_dir) == Path(".cache/models")
