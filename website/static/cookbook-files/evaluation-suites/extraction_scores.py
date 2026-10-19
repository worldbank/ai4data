"""Score an extraction run against gold spans, with an error taxonomy.

Reads aligned gold and predicted spans (CSV: doc_id, gold_span,
gold_type, pred_span, pred_type; a row with an empty gold is a spurious
prediction, a row with an empty prediction is a miss) and reports
precision, recall, and F1 under two rules (exact span and type;
overlapping span and type), and the counts of each error kind:

    missed      a gold span with no prediction
    spurious    a prediction with no gold span
    boundary    overlapping spans, same type (a partial match)
    wrong type  same or overlapping span, different type

The taxonomy says what to fix: boundary errors point to the model's span
rules, wrong types to the label definitions, spurious predictions to the
threshold, misses to coverage. Uses pandas and rapidfuzz (partial-ratio overlap).

Usage:
    python extraction_scores.py extraction_gold_pred.csv

What this does not do: the alignment between gold and predictions is
given in the file; producing it from two separate lists is the matching
step of an evaluation harness. Twelve rows demonstrate the taxonomy; a
measure needs a labelled sample of a few hundred spans across document
types and languages.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from rapidfuzz import fuzz

OVERLAP = (
    80  # partial-ratio score (0 to 100) above which two spans count as overlapping
)

# What each kind of row contributes under the exact rule and the lenient rule.
OUTCOMES = {
    "correct": {"exact": "tp", "lenient": "tp"},
    "boundary": {"exact": "fp+fn", "lenient": "tp"},
    "wrong type": {"exact": "fp+fn", "lenient": "fp+fn"},
    "wrong span": {"exact": "fp+fn", "lenient": "fp+fn"},
    "missed": {"exact": "fn", "lenient": "fn"},
    "spurious": {"exact": "fp", "lenient": "fp"},
}


def classify(r: pd.Series) -> str:
    """Name the kind of row: correct, boundary, wrong type, wrong span, missed, or spurious."""
    g, p = r["gold_span"].strip(), r["pred_span"].strip()
    if g and not p:
        return "missed"
    if p and not g:
        return "spurious"
    if r["gold_type"] != r["pred_type"]:
        return "wrong type"
    if g.lower() == p.lower():
        return "correct"
    if fuzz.partial_ratio(g.lower(), p.lower()) >= OVERLAP:
        return "boundary"
    return "wrong span"


def prf(counts: pd.Series) -> tuple[float, float, float]:
    tp, fp, fn = counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("aligned")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.aligned, dtype=str).fillna("")
    df["kind"] = df.apply(classify, axis=1)
    kinds = df["kind"].value_counts()

    print(f"{len(df)} aligned rows\n")
    for rule, name in (
        ("exact", "exact span and type"),
        ("lenient", "overlapping span, same type"),
    ):
        outcomes = (
            df["kind"]
            .map(lambda k, rule=rule: OUTCOMES[k][rule])
            .str.split("+")
            .explode()
            .value_counts()
        )
        p, r, f = prf(outcomes)
        print(
            f"{name:<30} precision {p:.2f}  recall {r:.2f}  F1 {f:.2f}  (tp {outcomes.get('tp', 0)}, fp {outcomes.get('fp', 0)}, fn {outcomes.get('fn', 0)})"
        )
    print("\nerror taxonomy: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    print(
        "boundary -> span rules; wrong type -> label definitions; spurious -> threshold; missed -> coverage"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
