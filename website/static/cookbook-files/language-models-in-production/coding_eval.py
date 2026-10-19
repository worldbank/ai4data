"""Evaluate model coding of open text to a hierarchical classification.

Reads coded responses (CSV: response_id, response_text, gold_code,
model_code, confidence) for a classification with a digit hierarchy
such as ISCO-08 (major group 1 digit, sub-major 2, minor 3, unit group
4) and reports the accuracy at each level, then the trade-off that
decides the production rule: for several confidence thresholds, the
share of responses the model would code automatically and the accuracy
on those, with the rest routed to human coders. Uses pandas and scikit-learn (accuracy).

Usage:
    python coding_eval.py occupation_coding.csv --levels 1 2 3 4

What this does not do: twenty responses demonstrate the curve; the
threshold for production is set on a few thousand coded responses per
language and survey, re-measured when the model or the classification
version changes. Accuracy against gold codes inherits the gold coders'
own consistency, which is measured as the labelling chapters of the
evaluation cookbook describe.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.metrics import accuracy_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("coded")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.0, 0.5, 0.6, 0.8, 0.9]
    )
    args = parser.parse_args(argv)
    df = pd.read_csv(args.coded, dtype={"gold_code": str, "model_code": str})
    n = len(df)

    print(f"{n} responses\n")
    print("accuracy by level")
    for level in args.levels:
        acc = accuracy_score(df["gold_code"].str[:level], df["model_code"].str[:level])
        print(f"  {level}-digit {acc:.2f} ({round(acc * n)}/{n})")

    print(
        "\nauto-coding rule: code automatically at or above the threshold, route the rest to coders"
    )
    print(
        f"{'threshold':>9} {'auto-coded':>10} {'accuracy (4-digit)':>18} {'to coders':>9}"
    )
    correct = df["gold_code"] == df["model_code"]
    for t in args.thresholds:
        auto = df["confidence"] >= t
        acc = correct[auto].mean() if auto.any() else 0.0
        print(f"{t:>9.2f} {auto.mean():>10.2f} {acc:>18.2f} {1 - auto.mean():>9.2f}")

    wrong = df[~correct]
    if not wrong.empty:
        print("\nerrors (gold -> model, confidence):")
        for r in wrong.itertuples():
            print(
                f"  {r.response_id} {r.response_text!r}: {r.gold_code} -> {r.model_code} ({r.confidence})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
