"""Test the YOLO26 adapter without loading model weights."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from ai4data.data_snapshot.adapters.yolo26 import YOLO26Adapter, YOLO26Config


class _FakeTensor:
    """Provide the tensor calls used by the adapter."""

    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def cpu(self) -> "_FakeTensor":
        """Return this CPU-backed fake tensor."""
        return self

    def tolist(self) -> list[Any]:
        """Return the stored JSON-compatible values."""
        return self._values


class _FakeBoxes:
    """Provide fake YOLO box tensors."""

    xyxy = _FakeTensor(
        [
            [10.0, 20.0, 90.0, 80.0],
            [0.0, 0.0, 20.0, 10.0],
            [0.0, 0.0, 1.0, 1.0],
            [10.0, 10.0, 30.0, 30.0],
        ]
    )
    conf = _FakeTensor([0.9, 1.2, 0.5, 0.8])
    cls = _FakeTensor([0.0, 1.0, 0.0, 2.0])


class _FakeResult:
    """Provide one fake Ultralytics result."""

    boxes = _FakeBoxes()
    names = {0: "Picture", 1: "Table", 2: "Text"}


class _FakeModel:
    """Record prediction arguments and return a fake result."""

    def __init__(self) -> None:
        self.arguments: dict[str, Any] | None = None

    def predict(self, image: Image.Image, **arguments: Any) -> list[_FakeResult]:
        """Return fixed detections for the supplied image."""
        self.arguments = arguments
        return [_FakeResult()]


def test_yolo26_adapter_normalizes_and_filters_detections() -> None:
    """Map model output to canonical labels, boxes, and scores."""
    model = _FakeModel()
    config = YOLO26Config(device="cpu", imgsz=640, min_area=0.008)
    adapter = YOLO26Adapter(config, model=model)

    detections = adapter.detect(Image.new("RGB", (100, 100)))

    assert [(item.label, item.bbox, item.score) for item in detections] == [
        ("figure", (0.1, 0.2, 0.9, 0.8), 0.9),
        ("table", (0.0, 0.0, 0.2, 0.1), 1.0),
    ]
    assert model.arguments == {
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "device": "cpu",
        "verbose": False,
    }


def test_yolo26_config_rejects_invalid_threshold() -> None:
    """Reject invalid model configuration before inference."""
    with pytest.raises(ValueError, match="Confidence"):
        YOLO26Config(conf=1.1)


def test_yolo26_config_uses_medium_weights_in_local_cache() -> None:
    """Keep medium weights in the repository-local ignored cache."""
    config = YOLO26Config()

    assert config.filename == "yolo26m_doc_layout.pt"
    assert Path(config.cache_dir) == Path(".cache/models")


def test_yolo26_adapter_loads_cached_weights_without_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load existing medium weights without contacting Hugging Face."""
    model_path = tmp_path / "yolo26m_doc_layout.pt"
    model_path.touch()
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=lambda path: path),
    )

    adapter = YOLO26Adapter(YOLO26Config(cache_dir=tmp_path))

    assert adapter._model == str(model_path)
