"""Test the canonical layout detection contract."""

import pytest

from ai4data.data_snapshot import Detection


def test_detection_accepts_canonical_values() -> None:
    """Retain a valid normalized figure detection."""
    detection = Detection("figure", (0.1, 0.2, 0.8, 0.9), 0.75)

    assert detection.label == "figure"


@pytest.mark.parametrize(
    ("label", "bbox", "score"),
    [
        ("caption", (0.1, 0.2, 0.8, 0.9), 0.75),
        ("figure", (-0.1, 0.2, 0.8, 0.9), 0.75),
        ("table", (0.8, 0.2, 0.1, 0.9), 0.75),
        ("table", (0.1, 0.2, 0.8, 0.9), 1.1),
    ],
)
def test_detection_rejects_invalid_values(
    label: str, bbox: tuple[float, float, float, float], score: float
) -> None:
    """Reject noncanonical labels, boxes, and scores."""
    with pytest.raises(ValueError):
        Detection(label, bbox, score)  # type: ignore[arg-type]
