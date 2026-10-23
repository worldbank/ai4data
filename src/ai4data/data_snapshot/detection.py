"""Define canonical data snapshot detections and detector protocol."""

from dataclasses import dataclass
from typing import Literal, Protocol

from PIL import Image


@dataclass(frozen=True)
class Detection:
    """Represent one normalized figure or table detection.

    Parameters
    ----------
    label : {"figure", "table"}
        Canonical snapshot label.
    bbox : tuple[float, float, float, float]
        Normalized ``(x1, y1, x2, y2)`` coordinates in the range ``[0, 1]``.
    score : float
        Model confidence in the range ``[0, 1]``.

    Raises
    ------
    ValueError
        If the label, bounding box, or score is invalid.
    """

    label: Literal["figure", "table"]
    bbox: tuple[float, float, float, float]
    score: float

    def __post_init__(self) -> None:
        """Validate the canonical detection values."""
        if self.label not in {"figure", "table"}:
            raise ValueError(f"Unsupported detection label: {self.label}")
        x1, y1, x2, y2 = self.bbox
        if not all(0.0 <= coordinate <= 1.0 for coordinate in self.bbox):
            raise ValueError("Detection coordinates must be between zero and one.")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Detection bounding box must have positive area.")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Detection score must be between zero and one.")


class SnapshotDetector(Protocol):
    """Define the page-image interface implemented by snapshot detectors."""

    def detect(self, image: Image.Image) -> list[Detection]:
        """Return canonical detections for one rendered PDF page.

        Parameters
        ----------
        image : PIL.Image.Image
            Rendered RGB PDF page.

        Returns
        -------
        list[Detection]
            Figure and table detections for the page.
        """
        ...
