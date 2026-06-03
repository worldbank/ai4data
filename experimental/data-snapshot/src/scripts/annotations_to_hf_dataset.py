"""Split an evaluation JSON file into per-document annotation files.

Reads a JSON file conforming to the Unified Evaluation Schema v1.3 and
produces one JSON file per document, suitable for upload to a HuggingFace
dataset. Each output file retains the full schema structure with a
single-element ``documents`` array and only the predictions belonging to
that document.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jsonschema

from dsa.utils import load_json
from dsa.constants import ROOT

# Resolve the schema path relative to the project root.
_SCHEMA_PATH = ROOT / "docs/data-snapshot-eval-v1.3.schema.json"


def split_annotations(
    input_json_path: Path,
    output_dir: Path,
) -> None:
    """Split an evaluation JSON into per-document files.

    For each document in the input file, a new JSON file is written to
    *output_dir* containing the shared ``label_map`` and ``info``, the
    single document entry (wrapped in an array), and only the prediction
    pages that belong to that document.

    Parameters
    ----------
    input_json_path : Path
        Path to the evaluation JSON file.
    output_dir : Path
        Directory where per-document JSON files will be written.
        Created automatically if it does not exist.

    Raises
    ------
    FileNotFoundError
        If *input_json_path* does not exist.
    FileNotFoundError
        If the evaluation schema file cannot be found.
    KeyError
        If the input JSON is missing required top-level keys.
    """
    input_json_path = Path(input_json_path)
    output_dir = Path(output_dir)

    if not input_json_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_json_path}")

    if not _SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Schema file not found: {_SCHEMA_PATH}")

    schema = load_json(_SCHEMA_PATH)
    data = load_json(input_json_path)

    label_map = data["label_map"]
    info = data["info"]
    documents = data["documents"]
    predictions = data["predictions"]

    # Strip timezone offset from created_at for HuggingFace compatibility.
    if "created_at" in info and isinstance(info["created_at"], str):
        info["created_at"] = info["created_at"].replace("Z", "").replace("+00:00", "")

    # Build a lookup: doc_id -> list of prediction page entries
    preds_by_doc: dict[str, list[dict]] = defaultdict(list)
    for page_entry in predictions:
        preds_by_doc[page_entry["doc_id"]].append(page_entry)

    output_dir.mkdir(parents=True, exist_ok=True)

    docs_with_annotations = 0
    docs_without_annotations = 0
    validation_failures: list[str] = []

    for doc in documents:
        doc_id = doc["doc_id"]
        doc_preds = preds_by_doc.get(doc_id, [])

        doc_json = {
            "label_map": label_map,
            "info": info,
            "documents": [doc],
            "predictions": doc_preds,
        }

        # Validate against the schema before writing
        try:
            jsonschema.validate(instance=doc_json, schema=schema)
        except jsonschema.ValidationError as exc:
            validation_failures.append(doc_id)
            print(f"  [SCHEMA ERROR] {doc_id}: {exc.message}")
            continue

        fname = f"{Path(doc_id).stem}_annotations.json"
        out_path = output_dir / fname

        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(doc_json, fh, indent=2)

        if doc_preds:
            docs_with_annotations += 1
        else:
            docs_without_annotations += 1
            print(f"  [no annotations] {doc_id}")

    total = len(documents)
    files_written = total - len(validation_failures)
    print(
        f"\nSplit complete.\n"
        f"  Input file:              {input_json_path}\n"
        f"  Output directory:        {output_dir}\n"
        f"  Total documents:         {total}\n"
        f"  With annotations:        {docs_with_annotations}\n"
        f"  Without annotations:     {docs_without_annotations}\n"
        f"  Validation failures:     {len(validation_failures)}\n"
        f"  Files written:           {files_written}"
    )

    if validation_failures:
        print("\nDocuments that failed schema validation:")
        for doc_id in validation_failures:
            print(f"  - {doc_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split an evaluation JSON file into per-document files for HuggingFace dataset upload."
    )
    parser.add_argument(
        "--input_json_path",
        type=Path,
        required=True,
        help="Path to the annotation JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write per-document annotation files.",
    )
    args = parser.parse_args()
    split_annotations(
        input_json_path=args.input_json_path,
        output_dir=args.output_dir,
    )
