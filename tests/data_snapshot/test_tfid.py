"""Test the TF-ID adapter without loading model weights."""

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ai4data.data_snapshot.adapters.tfid import TFIDAdapter, TFIDConfig


class _FakeInputs(dict[str, object]):
    """Provide the device transfer used by Transformers batch inputs."""

    def to(self, device: str) -> "_FakeInputs":
        """Record no device state for the CPU-only fake inputs."""
        return self


class _FakeModel:
    """Return a fixed generated token sequence."""

    device = "cpu"

    def generate(self, **arguments: Any) -> list[list[int]]:
        """Return one fake generated sequence."""
        assert arguments == {
            "input_ids": "input_ids",
            "pixel_values": "pixel_values",
            "max_new_tokens": 1024,
            "do_sample": False,
            "num_beams": 3,
        }
        return [[1, 2, 3]]


class _FakeProcessor:
    """Return fixed TF-ID inputs and parsed object detections."""

    def __call__(self, **arguments: Any) -> _FakeInputs:
        """Return the two tensors consumed by generation."""
        assert arguments["text"] == "<OD>"
        assert arguments["return_tensors"] == "pt"
        return _FakeInputs(input_ids="input_ids", pixel_values="pixel_values")

    def batch_decode(self, *args: object, **kwargs: object) -> list[str]:
        """Return one generated TF-ID response."""
        return ["generated"]

    def post_process_generation(
        self, *args: object, **kwargs: object
    ) -> dict[str, Any]:
        """Return figure, table, ignored, and undersized predictions."""
        return {
            "<OD>": {
                "bboxes": [
                    [10.0, 20.0, 90.0, 80.0],
                    [0.0, 0.0, 20.0, 10.0],
                    [10.0, 10.0, 30.0, 30.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                "labels": ["Figure", "Table", "Text", "Figure"],
                "scores": [0.9, 1.2, 0.8, 0.7],
            }
        }


def test_tfid_adapter_normalizes_and_filters_detections() -> None:
    """Map TF-ID output to canonical labels, boxes, and scores."""
    adapter = TFIDAdapter(
        TFIDConfig(min_area=0.008),
        model=_FakeModel(),
        processor=_FakeProcessor(),
    )

    detections = adapter.detect(Image.new("RGB", (100, 100)))

    assert [(item.label, item.bbox, item.score) for item in detections] == [
        ("figure", (0.1, 0.2, 0.9, 0.8), 0.9),
        ("table", (0.0, 0.0, 0.2, 0.1), 1.0),
    ]


def test_tfid_config_uses_pinned_large_model_in_local_cache() -> None:
    """Keep TF-ID Large and its remote code pinned in the local cache."""
    config = TFIDConfig()

    assert config.model_id == "yifeihu/TF-ID-large"
    assert config.revision == "a91ffef47039768b2deff82681b75d77252703f2"
    assert Path(config.cache_dir) == Path(".cache/models/tfid")


def test_tfid_adapter_requires_model_and_processor_together() -> None:
    """Reject incomplete dependency injection before inference."""
    with pytest.raises(ValueError, match="supplied together"):
        TFIDAdapter(model=_FakeModel())
