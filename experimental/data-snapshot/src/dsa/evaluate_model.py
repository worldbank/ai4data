"""
Evaluation framework for layout detection models.

Computes detection accuracy (precision, recall) and spatial accuracy
(mean IoU, coverage, purity) by comparing prediction and ground-truth
JSON files conforming to the Unified Evaluation Schema v1.3.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dsa.constants import (
    ROOT,
    GT_JSON_PATH,
    PRED_JSON_PATH,
    OUTPUT_REPORT_PATH,
    IOU_THRESHOLDS,
    LABELS_TO_CONSIDER,
)
from dsa.utils import load_json, sanitize_bbox


def bbox_area(b: tuple[float, float, float, float]) -> float:
    """Compute the area of a bounding box.

    Parameters
    ----------
    b : tuple[float, float, float, float]
        Bounding box as ``(x1, y1, x2, y2)``.

    Returns
    -------
    float
        Area of the box (zero if degenerate).
    """
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Compute the intersection area of two bounding boxes.

    Parameters
    ----------
    a : tuple[float, float, float, float]
        First bounding box as ``(x1, y1, x2, y2)``.
    b : tuple[float, float, float, float]
        Second bounding box as ``(x1, y1, x2, y2)``.

    Returns
    -------
    float
        Intersection area (zero if no overlap).
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    a : tuple[float, float, float, float]
        First bounding box as ``(x1, y1, x2, y2)``.
    b : tuple[float, float, float, float]
        Second bounding box as ``(x1, y1, x2, y2)``.

    Returns
    -------
    float
        IoU value in ``[0, 1]``.
    """
    inter = bbox_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def area_recall(
    pred: tuple[float, float, float, float],
    gt: tuple[float, float, float, float],
) -> float:
    """Compute coverage: intersection divided by ground-truth area.

    Parameters
    ----------
    pred : tuple[float, float, float, float]
        Predicted bounding box.
    gt : tuple[float, float, float, float]
        Ground-truth bounding box.

    Returns
    -------
    float
        Coverage value in ``[0, 1]``.
    """
    inter = bbox_intersection(pred, gt)
    g = bbox_area(gt)
    return inter / g if g > 0.0 else 0.0


def area_precision(
    pred: tuple[float, float, float, float],
    gt: tuple[float, float, float, float],
) -> float:
    """Compute purity: intersection divided by prediction area.

    Parameters
    ----------
    pred : tuple[float, float, float, float]
        Predicted bounding box.
    gt : tuple[float, float, float, float]
        Ground-truth bounding box.

    Returns
    -------
    float
        Purity value in ``[0, 1]``.
    """
    inter = bbox_intersection(pred, gt)
    p = bbox_area(pred)
    return inter / p if p > 0.0 else 0.0


@dataclass(frozen=True)
class DetObj:
    """A single detected object (prediction or ground truth).

    Parameters
    ----------
    page_id : str
        Page identifier in ``"doc_id::pNNN"`` format.
    obj_id : str
        Unique object identifier.
    label : str
        Canonical label (``"Figure"`` or ``"Table"``).
    bbox : tuple[float, float, float, float]
        Normalized bounding box ``(x1, y1, x2, y2)``.
    score : float | None
        Confidence score (predictions only).
    """

    page_id: str
    obj_id: str
    label: str
    bbox: tuple[float, float, float, float]
    score: float | None = None


def prepare_prediction_objects(
    pred_dict: dict, filter_list: list[str] | None = None
) -> list[DetObj]:
    """Extract detection objects from a schema-compliant JSON dict.

    Parameters
    ----------
    pred_dict : dict
        Parsed JSON conforming to the Unified Evaluation Schema v1.3.
    filter_list : list[str] | None
        Document IDs to exclude.  ``None`` means no filtering.

    Returns
    -------
    list[DetObj]
        Flat list of detection objects across all pages.
    """
    if filter_list is None:
        filter_list = []

    preds = pred_dict.get("predictions", [])
    objs: list[DetObj] = []

    for p in preds:
        if p.get("doc_id") in filter_list:
            continue
        page_id = p.get("page_id")
        for o in p.get("objects", []):
            label = o.get("label")
            obj_id = str(o.get("id", ""))
            bb = sanitize_bbox(o.get("bbox"))
            score = o.get("score", None)

            objs.append(
                DetObj(
                    page_id=page_id,
                    obj_id=obj_id,
                    label=label,
                    bbox=bb,
                    score=score,
                )
            )

    return objs


@dataclass
class Stats:
    """Accumulator for detection and spatial accuracy metrics.

    Tracks true positives, false positives, false negatives, and
    running sums of spatial metrics (IoU, coverage, purity) for
    matched prediction–ground-truth pairs.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched: int = 0
    iou_sum: float = 0.0
    area_recall_sum: float = 0.0
    area_precision_sum: float = 0.0

    def add_match(
        self,
        pred_box: tuple[float, float, float, float],
        gt_box: tuple[float, float, float, float],
    ) -> None:
        """Record a matched prediction–ground-truth pair.

        Parameters
        ----------
        pred_box : tuple[float, float, float, float]
            Predicted bounding box.
        gt_box : tuple[float, float, float, float]
            Ground-truth bounding box.
        """
        self.tp += 1
        self.matched += 1
        self.iou_sum += iou(pred_box, gt_box)
        self.area_recall_sum += area_recall(pred_box, gt_box)
        self.area_precision_sum += area_precision(pred_box, gt_box)

    def add_fp(self, n: int) -> None:
        """Add *n* false positives.

        Parameters
        ----------
        n : int
            Number of false positives to add.
        """
        self.fp += n

    def add_fn(self, n: int) -> None:
        """Add *n* false negatives.

        Parameters
        ----------
        n : int
            Number of false negatives to add.
        """
        self.fn += n

    def precision(self) -> float:
        """Compute precision: ``TP / (TP + FP)``."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else math.nan

    def recall(self) -> float:
        """Compute recall: ``TP / (TP + FN)``."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else math.nan

    def mean_iou(self) -> float:
        """Compute mean IoU across matched pairs."""
        return self.iou_sum / self.matched if self.matched > 0 else math.nan

    def mean_area_recall(self) -> float:
        """Compute mean coverage across matched pairs."""
        return self.area_recall_sum / self.matched if self.matched > 0 else math.nan

    def mean_area_precision(self) -> float:
        """Compute mean purity across matched pairs."""
        return self.area_precision_sum / self.matched if self.matched > 0 else math.nan


def greedy_match(
    gt: list[DetObj], pred: list[DetObj], thr: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy one-to-one matching by IoU (descending).

    Only pairs with ``IoU >= thr`` are considered.  Each ground-truth
    and prediction object may be matched at most once.

    Parameters
    ----------
    gt : list[DetObj]
        Ground-truth objects for a single page and label.
    pred : list[DetObj]
        Predicted objects for a single page and label.
    thr : float
        Minimum IoU threshold for a valid match.

    Returns
    -------
    tuple[list[tuple[int, int, float]], list[int], list[int]]
        ``(matches, unmatched_pred_indices, unmatched_gt_indices)``
        where each match is ``(pred_idx, gt_idx, iou_value)``.
    """
    cand: list[tuple[int, int, float]] = []
    for p_idx, p in enumerate(pred):
        for g_idx, g in enumerate(gt):
            v = iou(p.bbox, g.bbox)
            if v >= thr:
                cand.append((p_idx, g_idx, v))

    cand.sort(reverse=True, key=lambda t: t[2])

    used_p, used_g = set(), set()
    matches: list[tuple[int, int, float]] = []
    for p_idx, g_idx, v in cand:
        if p_idx in used_p or g_idx in used_g:
            continue
        used_p.add(p_idx)
        used_g.add(g_idx)
        matches.append((p_idx, g_idx, v))

    unmatched_p = [i for i in range(len(pred)) if i not in used_p]
    unmatched_g = [i for i in range(len(gt)) if i not in used_g]
    return matches, unmatched_p, unmatched_g


def get_doc_ids(pred_dict: dict) -> set[str]:
    """Extract document IDs from a schema-compliant JSON dict.

    Parameters
    ----------
    pred_dict : dict
        Parsed JSON conforming to the Unified Evaluation Schema v1.3.

    Returns
    -------
    set[str]
        Set of ``doc_id`` values.
    """
    docs = pred_dict.get("documents", [])

    ids = set()
    for d in docs:
        if isinstance(d, dict):
            ids.add(d.get("doc_id"))

    return ids


def get_document_mismatch(gt: dict, pred: dict) -> tuple[list[str], list[str]]:
    """Compare document IDs between ground truth and predictions.

    Prints warnings for any mismatches and returns the differences.

    Parameters
    ----------
    gt : dict
        Ground-truth JSON dict.
    pred : dict
        Prediction JSON dict.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(only_in_gt, only_in_pred)`` — document IDs present in one
        file but not the other.
    """
    gt_docs = get_doc_ids(gt)
    pred_docs = get_doc_ids(pred)

    only_gt = sorted(gt_docs - pred_docs)
    only_pred = sorted(pred_docs - gt_docs)

    if not only_gt and not only_pred:
        print("[OK] documents are consistent: same doc_id set in GT and predictions.")
    if only_gt:
        print(
            f"[WARN] doc_ids present in GT but missing in predictions ({len(only_gt)}):"
        )
        for x in only_gt:
            print(f"  - {x}")
    if only_pred:
        print(
            f"[WARN] doc_ids present in predictions but missing in GT ({len(only_pred)}):"
        )
        for x in only_pred:
            print(f"  - {x}")

    return only_gt, only_pred


def print_summary_metrics(output_report_path: str | Path, iou_threshold: float) -> None:
    """Print summary metrics in a markdown table format.

    Parameters
    ----------
    output_report_path : str | Path
        Path to the evaluation report JSON.
    iou_threshold : float
        IoU threshold to extract metrics for.
    """
    report = load_json(output_report_path)
    metrics = report.get("metrics", {}).get(str(iou_threshold))

    if not metrics:
        print(f"[WARN] No metrics found for IoU threshold {iou_threshold}")
        return

    print("\nSummary metrics")
    per_class = metrics.get("per_class", {})

    for lab in ["Figure", "Table"]:
        class_metrics = per_class.get(lab)
        if not class_metrics:
            continue

        print(f"### {lab} (IoU={iou_threshold})")
        print("|     Category     |     Metric     | Score |")
        print("|:----------------:|:--------------:|:-----:|")
        print(
            f"| Detection        | Precision      | {class_metrics['precision']:.3f} |"
        )
        print(f"|                  | Recall         | {class_metrics['recall']:.3f} |")
        print(
            f"| Spatial accuracy | IoU            | {class_metrics['mean_iou']:.3f} |"
        )
        print(
            f"|                  | Area precision | {class_metrics['mean_area_precision']:.3f} |"
        )
        print(
            f"|                  | Area recall    | {class_metrics['mean_area_recall']:.3f} |"
        )

    # Object counts
    for lab in ["Figure", "Table"]:
        class_metrics = per_class.get(lab)
        if not class_metrics:
            continue

        object_count = class_metrics["tp"] + class_metrics["fn"]
        print(f"{lab} count: {object_count}")


def evaluate(
    gt_json_path: str | Path,
    pred_json_path: str | Path,
    *,
    iou_thresholds: tuple[float, ...] = (0.5, 0.75),
    labels: tuple[str, ...] = ("Figure", "Table"),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run evaluation comparing ground truth and predictions.

    Computes detection metrics (precision, recall) and spatial accuracy
    metrics (mean IoU, coverage, purity) at each IoU threshold, per
    class and micro-averaged.

    Parameters
    ----------
    gt_json_path : str | Path
        Path to the ground-truth JSON file.
    pred_json_path : str | Path
        Path to the prediction JSON file.
    iou_thresholds : tuple[float, ...]
        IoU thresholds at which to evaluate.
    labels : tuple[str, ...]
        Object classes to evaluate.
    output_path : str | Path | None
        If provided, the evaluation report is written to this path.

    Returns
    -------
    dict[str, Any]
        Complete evaluation report dictionary.
    """
    # Load files
    gt_json_path = Path(gt_json_path)
    pred_json_path = Path(pred_json_path)
    gt = load_json(gt_json_path)
    pred = load_json(pred_json_path)

    # Prepare files
    only_gt, only_pred = get_document_mismatch(gt, pred)
    gt_objects = prepare_prediction_objects(gt)
    pred_objects = prepare_prediction_objects(pred)
    page_ids = sorted(
        set([x.page_id for x in gt_objects]) | set([x.page_id for x in pred_objects])
    )

    # Prepare report dict
    schema_version = gt.get("info").get("schema_version")
    label_map = gt.get("label_map")
    label_map = {k: v for k, v in label_map.items() if v in labels}
    doc_mismatch = {"only_gt": only_gt, "only_pred": only_pred}
    report: dict[str, Any] = {
        "info": {
            "schema_version": schema_version,
            "gt_path": str(gt_json_path.resolve().relative_to(ROOT)),
            "pred_path": str(pred_json_path.resolve().relative_to(ROOT)),
        },
        "label_map": label_map,
        "thresholds": list(iou_thresholds),
        "documents_mismatch": doc_mismatch,
        "metrics": {},
    }

    # Calculate metrics
    for thr in iou_thresholds:
        per_class: dict[str, Stats] = {lab: Stats() for lab in labels}
        micro = Stats()

        for p in page_ids:
            for lab in labels:
                gt_lab = [o for o in gt_objects if o.label == lab and o.page_id == p]
                pred_lab = [
                    o for o in pred_objects if o.label == lab and o.page_id == p
                ]

                matches, unmatched_p, unmatched_g = greedy_match(gt_lab, pred_lab, thr)

                st = per_class[lab]
                for p_idx, g_idx, _ in matches:
                    pred_box = pred_lab[p_idx].bbox
                    gt_box = gt_lab[g_idx].bbox
                    st.add_match(pred_box, gt_box)
                    micro.add_match(pred_box, gt_box)

                st.add_fp(len(unmatched_p))
                st.add_fn(len(unmatched_g))
                micro.add_fp(len(unmatched_p))
                micro.add_fn(len(unmatched_g))

        report["metrics"][str(thr)] = {
            "micro": {
                "tp": micro.tp,
                "fp": micro.fp,
                "fn": micro.fn,
                "precision": micro.precision(),
                "recall": micro.recall(),
                "matched": micro.matched,
                "mean_iou": micro.mean_iou(),
                "mean_area_precision": micro.mean_area_precision(),
                "mean_area_recall": micro.mean_area_recall(),
            },
            "per_class": {
                lab: {
                    "tp": st.tp,
                    "fp": st.fp,
                    "fn": st.fn,
                    "precision": st.precision(),
                    "recall": st.recall(),
                    "matched": st.matched,
                    "mean_iou": st.mean_iou(),
                    "mean_area_precision": st.mean_area_precision(),
                    "mean_area_recall": st.mean_area_recall(),
                }
                for lab, st in per_class.items()
            },
        }

    # Save report
    if output_path is not None:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate a layout detection model against ground truth."
    )
    ap.add_argument(
        "--gt_json_path",
        help="Path to ground truth json file",
        default=GT_JSON_PATH,
    )
    ap.add_argument(
        "--pred_json_path",
        help="Path to prediction json file",
        default=PRED_JSON_PATH,
    )
    ap.add_argument(
        "--output_report_path",
        default=OUTPUT_REPORT_PATH,
        help="Path to save output report json",
    )
    ap.add_argument(
        "--thr",
        nargs="*",
        type=float,
        default=IOU_THRESHOLDS,
        help="IoU thresholds (space separated)",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        type=str,
        default=LABELS_TO_CONSIDER,
        help="Labels to consider (space separated)",
    )
    args = ap.parse_args()

    rep = evaluate(
        gt_json_path=args.gt_json_path,
        pred_json_path=args.pred_json_path,
        iou_thresholds=tuple(args.thr),
        labels=tuple(args.labels),
        output_path=args.output_report_path,
    )
    print(f"Done! Evaluations report saved at {args.output_report_path}")
    print_summary_metrics(output_report_path=args.output_report_path, iou_threshold=0.5)
