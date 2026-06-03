"""Combine multiple batch prediction JSON files into a single prediction file.

Reads all ``*.json`` files from a given directory, merges their documents and
page-level predictions, and writes a single output file conforming to the
Unified Evaluation Schema v1.3.

All input files must share the same ``label_map``, model name, and coordinate
system.  Documents and pages are deduplicated by ``doc_id`` and ``page_id``
respectively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dsa.utils import load_json


def _generate_run_id(model_name: str) -> str:
    """Generate a unique run ID based on model name and current timestamp.

    Parameters
    ----------
    model_name : str
        Name of the model used for predictions.

    Returns
    -------
    str
        A run ID in the format ``"<model_prefix>-<hash>"``.
    """
    ts = datetime.now(tz=timezone.utc).isoformat()
    digest = hashlib.sha256(f"{model_name}:{ts}".encode()).hexdigest()[:10]
    prefix = model_name.split("/")[-1].lower().replace(" ", "-")[:30]
    return f"{prefix}-combined-{digest}"


def _validate_batch_consistency(batch_data: list[dict], paths: list[Path]) -> None:
    """Validate that all batch files share consistent metadata.

    Checks that ``label_map``, model name, and coordinate system are
    identical across all batch files.

    Parameters
    ----------
    batch_data : list[dict]
        Parsed JSON dicts from each batch file.
    paths : list[Path]
        Corresponding file paths (used in error messages).

    Raises
    ------
    ValueError
        If any metadata field is inconsistent across batch files.
    """
    reference = batch_data[0]
    ref_label_map = reference.get("label_map")
    ref_model_name = reference.get("info", {}).get("model", {}).get("name")
    ref_coord_system = reference.get("info", {}).get("coordinate_system")

    for i, data in enumerate(batch_data[1:], start=1):
        cur_label_map = data.get("label_map")
        cur_model_name = data.get("info", {}).get("model", {}).get("name")
        cur_coord_system = data.get("info", {}).get("coordinate_system")

        if cur_label_map != ref_label_map:
            raise ValueError(
                f"label_map mismatch between '{paths[0].name}' and "
                f"'{paths[i].name}': {ref_label_map} vs {cur_label_map}"
            )
        if cur_model_name != ref_model_name:
            raise ValueError(
                f"Model name mismatch between '{paths[0].name}' and "
                f"'{paths[i].name}': {ref_model_name} vs {cur_model_name}"
            )
        if cur_coord_system != ref_coord_system:
            raise ValueError(
                f"Coordinate system mismatch between '{paths[0].name}' and "
                f"'{paths[i].name}'"
            )


def _merge_documents(batch_data: list[dict]) -> list[dict]:
    """Merge document metadata from all batches, deduplicating by ``doc_id``.

    Parameters
    ----------
    batch_data : list[dict]
        Parsed JSON dicts from each batch file.

    Returns
    -------
    list[dict]
        Combined list of unique document metadata dicts.
    """
    seen: set[str] = set()
    merged: list[dict] = []
    for data in batch_data:
        for doc in data.get("documents", []):
            doc_id = doc.get("doc_id", "")
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc)
    return merged


def _merge_predictions(batch_data: list[dict]) -> list[dict]:
    """Merge page-level predictions from all batches, deduplicating by ``page_id``.

    If the same ``page_id`` appears in multiple batches, only the first
    occurrence is kept (with a warning printed).

    Parameters
    ----------
    batch_data : list[dict]
        Parsed JSON dicts from each batch file.

    Returns
    -------
    list[dict]
        Combined list of unique page prediction dicts.
    """
    seen: set[str] = set()
    merged: list[dict] = []
    for data in batch_data:
        for page in data.get("predictions", []):
            page_id = page.get("page_id", "")
            if page_id in seen:
                print(f"  [WARN] Duplicate page_id '{page_id}' — skipping.")
                continue
            seen.add(page_id)
            merged.append(page)
    return merged


def combine_batch_predictions(
    input_dir: Path,
    output_json_path: Path,
) -> None:
    """Combine all prediction JSON files in a directory into one file.

    Reads every ``*.json`` file from *input_dir*, validates metadata
    consistency, merges documents and predictions (deduplicating by ID),
    and writes a single schema-compliant output file.

    Parameters
    ----------
    input_dir : Path
        Directory containing batch prediction JSON files.
    output_json_path : Path
        Destination path for the combined prediction JSON file.

    Raises
    ------
    FileNotFoundError
        If *input_dir* does not exist or contains no JSON files.
    ValueError
        If batch files have inconsistent metadata.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    print(f"Found {len(json_files)} batch file(s) in '{input_dir}':")
    for f in json_files:
        print(f"  - {f.name}")

    # Load all batch files
    batch_data = [load_json(f) for f in json_files]

    # Validate consistency
    _validate_batch_consistency(batch_data, json_files)
    print("[OK] All batch files have consistent metadata.")

    # Merge documents and predictions
    merged_documents = _merge_documents(batch_data)
    merged_predictions = _merge_predictions(batch_data)

    # Build combined info block from the first file
    reference_info = batch_data[0].get("info", {})
    model_name = reference_info.get("model", {}).get("name", "unknown")

    combined_info = {
        "schema_version": reference_info.get("schema_version", "1.3"),
        "type": reference_info.get("type", "prediction"),
        "created_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": _generate_run_id(model_name),
        "model": reference_info.get("model", {}),
        "coordinate_system": reference_info.get("coordinate_system", {}),
    }

    output = {
        "label_map": batch_data[0].get("label_map", {}),
        "info": combined_info,
        "documents": merged_documents,
        "predictions": merged_predictions,
    }

    # Write output
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    n_docs = len(merged_documents)
    n_pages = len(merged_predictions)
    n_objects = sum(len(p.get("objects", [])) for p in merged_predictions)
    print(
        f"\nCombined file saved to: {output_json_path}\n"
        f"  Documents: {n_docs}\n"
        f"  Pages:     {n_pages}\n"
        f"  Objects:   {n_objects}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple batch prediction JSON files from a directory "
            "into a single prediction file conforming to the Unified "
            "Evaluation Schema v1.3."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing batch prediction JSON files.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Destination path for the combined prediction JSON file.",
    )
    args = parser.parse_args()
    combine_batch_predictions(
        input_dir=args.input_dir,
        output_json_path=args.output_json,
    )
