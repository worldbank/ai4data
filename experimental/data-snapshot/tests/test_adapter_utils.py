"""Tests for shared adapter utilities in ``dsa.utils``."""

import re

import pytest

from dsa.utils import clamp01, normalize_bboxes_xyxy, sanitize_bbox, utc_now_iso


class TestUtcNowIso:
    """Tests for ``utc_now_iso``."""

    def test_returns_string(self) -> None:
        assert isinstance(utc_now_iso(), str)

    def test_ends_with_z(self) -> None:
        assert utc_now_iso().endswith("Z")

    def test_iso8601_format(self) -> None:
        ts = utc_now_iso()
        # Should match YYYY-MM-DDTHH:MM:SSZ
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts)


class TestClamp01:
    """Tests for ``clamp01``."""

    def test_value_in_range(self) -> None:
        assert clamp01(0.5) == 0.5

    def test_negative_clamped(self) -> None:
        assert clamp01(-0.1) == 0.0

    def test_above_one_clamped(self) -> None:
        assert clamp01(1.5) == 1.0

    def test_boundary_zero(self) -> None:
        assert clamp01(0.0) == 0.0

    def test_boundary_one(self) -> None:
        assert clamp01(1.0) == 1.0


class TestSanitizeBbox:
    """Tests for ``sanitize_bbox``."""

    def test_valid_bbox(self) -> None:
        assert sanitize_bbox([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)

    def test_clamping(self) -> None:
        assert sanitize_bbox([-0.1, 0.0, 1.5, 1.0]) == (0.0, 0.0, 1.0, 1.0)

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid bbox"):
            sanitize_bbox([0.1, 0.2])

    def test_not_a_list_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid bbox"):
            sanitize_bbox("bad")  # type: ignore[arg-type]


class TestNormalizeBboxesXyxy:
    """Tests for ``normalize_bboxes_xyxy``."""

    def test_absolute_to_normalized(self) -> None:
        bboxes = [[100.0, 200.0, 300.0, 400.0]]
        result = normalize_bboxes_xyxy(bboxes, width=1000, height=1000)
        assert len(result) == 1
        assert result[0] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_already_normalized_passthrough(self) -> None:
        bboxes = [[0.1, 0.2, 0.9, 0.8]]
        result = normalize_bboxes_xyxy(bboxes, width=1000, height=1000)
        assert len(result) == 1
        assert result[0] == pytest.approx([0.1, 0.2, 0.9, 0.8])

    def test_clips_to_01(self) -> None:
        bboxes = [[-50.0, -50.0, 1100.0, 1100.0]]
        result = normalize_bboxes_xyxy(bboxes, width=1000, height=1000)
        assert len(result) == 1
        assert result[0] == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_degenerate_box_dropped(self) -> None:
        bboxes = [[0.5, 0.5, 0.5, 0.5]]  # zero area
        result = normalize_bboxes_xyxy(bboxes, width=1000, height=1000)
        assert len(result) == 0

    def test_wrong_length_dropped(self) -> None:
        bboxes = [[0.1, 0.2, 0.3]]  # only 3 elements
        result = normalize_bboxes_xyxy(bboxes, width=1000, height=1000)
        assert len(result) == 0

    def test_swapped_coords_reordered(self) -> None:
        bboxes = [[0.9, 0.8, 0.1, 0.2]]  # x1>x2 and y1>y2
        result = normalize_bboxes_xyxy(bboxes, width=1000, height=1000)
        assert len(result) == 1
        assert result[0] == pytest.approx([0.1, 0.2, 0.9, 0.8])

    def test_multiple_boxes(self) -> None:
        bboxes = [
            [100.0, 100.0, 500.0, 500.0],
            [0.1, 0.2, 0.3, 0.4],
        ]
        result = normalize_bboxes_xyxy(bboxes, width=1000, height=1000)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        result = normalize_bboxes_xyxy([], width=1000, height=1000)
        assert result == []
