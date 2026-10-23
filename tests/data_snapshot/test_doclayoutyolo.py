"""Test the DocLayout-YOLO adapter without loading model weights."""

from pathlib import Path

from ai4data.data_snapshot.adapters.doclayoutyolo import (
    DocLayoutYOLOAdapter,
    DocLayoutYOLOConfig,
)


def test_doclayoutyolo_adapter_uses_official_defaults_in_local_cache() -> None:
    """Keep the official checkpoint and inference defaults in local cache."""
    model = object()
    adapter = DocLayoutYOLOAdapter(model=model)

    assert adapter._model is model
    assert adapter.config == DocLayoutYOLOConfig()
    assert adapter.config.filename == "doclayout_yolo_docstructbench_imgsz1024.pt"
    assert Path(adapter.config.cache_dir) == Path(".cache/models")
    assert adapter.config.conf == 0.2
    assert adapter.config.imgsz == 1024
