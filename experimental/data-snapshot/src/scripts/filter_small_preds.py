"""Filter small bounding-box predictions from a prediction JSON file.

Reads a prediction file conforming to the Unified Evaluation Schema v1.3,
removes predictions whose normalised bounding-box area falls below a
configurable threshold, and writes the filtered result to a new JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsa.constants import MIN_PREDICTION_AREA
from dsa.utils import filter_small_predictions, load_json


def filter_prediction_file(
    input_json: Path,
    output_json: Path,
    min_area: float = MIN_PREDICTION_AREA,
) -> None:
    """Load a prediction file, filter small detections, and save the result.

    Iterates over every page in the ``predictions`` list and applies
    :func:`~dsa.utils.filter_small_predictions` to each page's ``objects``
    list.  All other fields (``info``, ``label_map``, ``documents``, etc.)
    are preserved unchanged.

    Parameters
    ----------
    input_json : Path
        Path to the input prediction JSON file.
    output_json : Path
        Destination path for the filtered prediction JSON file.
    min_area : float
        Minimum normalised bounding-box area (width × height) a prediction
        must have to be kept.  Defaults to ``MIN_PREDICTION_AREA``.

    Raises
    ------
    FileNotFoundError
        If *input_json* does not exist.
    """
    input_json = Path(input_json)
    if not input_json.is_file():
        raise FileNotFoundError(f"Input file not found: {input_json}")

    data = load_json(input_json)

    total_before = 0
    total_after = 0

    for page in data.get("predictions", []):
        objects = page.get("objects", [])
        total_before += len(objects)
        filtered = filter_small_predictions(objects, min_area=min_area)
        total_after += len(filtered)
        page["objects"] = filtered

    # Write output
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    removed = total_before - total_after
    print(
        f"Filtered predictions saved to: {output_json}\n"
        f"  Min area threshold: {min_area}\n"
        f"  Objects before:     {total_before}\n"
        f"  Objects after:      {total_after}\n"
        f"  Removed:            {removed}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Filter out small bounding-box predictions from a prediction "
            "JSON file.  Predictions whose normalised area (width x height) "
            "is below the threshold are removed."
        )
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="Path to the input prediction JSON file.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Destination path for the filtered prediction JSON file.",
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=MIN_PREDICTION_AREA,
        help=(
            f"Minimum normalised bounding-box area to keep "
            f"(default: {MIN_PREDICTION_AREA})."
        ),
    )
    args = parser.parse_args()
    filter_prediction_file(
        input_json=args.input_json,
        output_json=args.output_json,
        min_area=args.min_area,
    )
