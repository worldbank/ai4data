"""Combine two prediction files by selecting classes from each source.

Takes Figures from one prediction file and Tables from another, merging them
into a single unified prediction file conforming to the Unified Evaluation
Schema v1.3.
"""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from dsa.utils import load_json


def filter_objects_by_label(predictions: list[dict], keep_label: str) -> list[dict]:
    """Filter the objects in each page's prediction to a single label class.

    Parameters
    ----------
    predictions : list[dict]
        List of page-level prediction dicts (each containing an ``objects``
        key).
    keep_label : str
        The label string to retain (e.g. ``"Figure"`` or ``"Table"``).

    Returns
    -------
    list[dict]
        A new list of page prediction dicts with only the desired label kept.
        Pages with no matching objects are still included (with an empty
        ``objects`` list) so that the document structure is preserved.
    """
    filtered = []
    for page in predictions:
        page_copy = copy.deepcopy(page)
        page_copy["objects"] = [
            obj
            for obj in page_copy.get("objects", [])
            if obj.get("label") == keep_label
        ]
        filtered.append(page_copy)
    return filtered


def merge_page_predictions(
    figure_pages: list[dict], table_pages: list[dict]
) -> list[dict]:
    """Merge two lists of page predictions into one.

    Pages are matched by ``page_id``. If a ``page_id`` exists in only one
    source, it is included with whatever objects it contains. The merged
    object list for each page is the union of both sources' objects.

    Parameters
    ----------
    figure_pages : list[dict]
        Page predictions containing only Figure objects.
    table_pages : list[dict]
        Page predictions containing only Table objects.

    Returns
    -------
    list[dict]
        Merged list of page prediction dicts, with objects from both sources
        combined and ordered by their original position (figures first, then
        tables, per page).
    """
    figure_map: dict[str, dict] = {p["page_id"]: p for p in figure_pages}
    table_map: dict[str, dict] = {p["page_id"]: p for p in table_pages}

    all_page_ids = list(figure_map.keys()) + [
        pid for pid in table_map if pid not in figure_map
    ]

    merged: list[dict] = []
    for page_id in all_page_ids:
        fig_page = figure_map.get(page_id)
        tbl_page = table_map.get(page_id)

        if fig_page and tbl_page:
            combined = copy.deepcopy(fig_page)
            combined["objects"] = fig_page["objects"] + tbl_page["objects"]
        elif fig_page:
            combined = copy.deepcopy(fig_page)
        else:
            combined = copy.deepcopy(tbl_page)  # type: ignore[arg-type]

        merged.append(combined)

    return merged


def merge_document_lists(fig_docs: list[dict], tbl_docs: list[dict]) -> list[dict]:
    """Merge two document metadata lists, deduplicating by ``doc_id``.

    Parameters
    ----------
    fig_docs : list[dict]
        Document metadata from the figures prediction file.
    tbl_docs : list[dict]
        Document metadata from the tables prediction file.

    Returns
    -------
    list[dict]
        Combined list of unique document metadata dicts.
    """
    seen: set[str] = set()
    merged: list[dict] = []
    for doc in fig_docs + tbl_docs:
        doc_id = doc.get("doc_id", "")
        if doc_id not in seen:
            seen.add(doc_id)
            merged.append(doc)
    return merged


def combine_predictions(
    figure_preds_path: Path,
    table_preds_path: Path,
    output_json_path: Path,
) -> None:
    """Combine two prediction files by selecting classes from each source.

    Figures are taken exclusively from *figure_preds_path* and Tables are
    taken exclusively from *table_preds_path*. The resulting file is saved to
    *output_json_path* and conforms to the Unified Evaluation Schema v1.3.

    Parameters
    ----------
    figure_preds_path : Path
        Path to the prediction JSON file from which ``Figure`` objects are
        extracted.
    table_preds_path : Path
        Path to the prediction JSON file from which ``Table`` objects are
        extracted.
    output_json_path : Path
        Destination path for the merged prediction JSON file.
    """
    figure_data = load_json(figure_preds_path)
    table_data = load_json(table_preds_path)

    # Extract and filter page-level predictions
    figure_pages = filter_objects_by_label(
        figure_data.get("predictions", []), keep_label="Figure"
    )
    table_pages = filter_objects_by_label(
        table_data.get("predictions", []), keep_label="Table"
    )

    merged_predictions = merge_page_predictions(figure_pages, table_pages)
    merged_documents = merge_document_lists(
        figure_data.get("documents", []),
        table_data.get("documents", []),
    )

    # Build combined output — use figure file's info block as the base and
    # update the timestamp and relevant notes to reflect the merge.
    combined_info = copy.deepcopy(figure_data.get("info", {}))
    combined_info["run_id"] = "combined"
    combined_info["created_at"] = datetime.now(tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    combined_info["type"] = "prediction"

    fig_model = (
        figure_data.get("info", {})
        .get("model", {})
        .get("name", str(figure_preds_path.name))
    )
    tbl_model = (
        table_data.get("info", {})
        .get("model", {})
        .get("name", str(table_preds_path.name))
    )
    combined_info["model"] = {
        "notes": f"Combined prediction: Figures from '{fig_model}, Tables from {tbl_model}'"
    }

    combined_label_map = {
        k: v
        for k, v in {
            **figure_data.get("label_map", {}),
            **table_data.get("label_map", {}),
        }.items()
        if v in ("Figure", "Table")
    }

    output = {
        "label_map": combined_label_map,
        "info": combined_info,
        "documents": merged_documents,
        "predictions": merged_predictions,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    n_objects = sum(len(p["objects"]) for p in merged_predictions)
    n_pages = len(merged_predictions)
    print(
        f"Combined file saved to: {output_json_path} ({n_objects} pages, {n_pages} total objects)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Combine two prediction JSON files by selecting Figure objects "
            "from one file and Table objects from another."
        )
    )
    parser.add_argument(
        "--figure_preds",
        type=Path,
        required=True,
        help="Path to the prediction file from which Figure objects are taken.",
    )
    parser.add_argument(
        "--table_preds",
        type=Path,
        required=True,
        help="Path to the prediction file from which Table objects are taken.",
    )
    parser.add_argument(
        "--output_json_path",
        type=Path,
        required=True,
        help="Destination path for the merged prediction JSON file.",
    )
    args = parser.parse_args()
    combine_predictions(
        figure_preds_path=args.figure_preds,
        table_preds_path=args.table_preds,
        output_json_path=args.output_json_path,
    )
