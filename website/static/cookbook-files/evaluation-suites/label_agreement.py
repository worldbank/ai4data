"""Measure agreement between two labellers and list their disagreements.

Reads a CSV with one row per item and two label columns (label_a,
label_b) and reports the percent agreement, Cohen's kappa (agreement
corrected for chance), and the items the labellers disagree on, which
are the ones to adjudicate and the ones that refine the labelling
guide. Uses pandas and scikit-learn (Cohen's kappa).

Usage:
    python label_agreement.py labels_two_annotators.csv

What this does not do: two labellers on twenty items give a first
reading; a labelling guide is calibrated on a hundred or more, and a
third labeller adjudicates the disagreements. Kappa depends on the
label distribution, so it is reported with the percent agreement and
the number of items, never alone.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("labels")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.labels, dtype=str)

    agree = int((df["label_a"] == df["label_b"]).sum())
    kappa = cohen_kappa_score(df["label_a"], df["label_b"])
    print(
        f"{len(df)} items: agreement {agree / len(df):.2f} ({agree}/{len(df)}), Cohen's kappa {kappa:.2f}"
    )

    disagreements = df[df["label_a"] != df["label_b"]]
    if not disagreements.empty:
        print("\ndisagreements to adjudicate:")
        for r in disagreements.itertuples():
            print(f"  {r.item_id}: {r.text!r}: A {r.label_a} / B {r.label_b}")
    none_split = int(disagreements[["label_a", "label_b"]].eq("NONE").any(axis=1).sum())
    if none_split:
        print(
            f"\n{none_split} of {len(disagreements)} disagreements involve NONE: the guide's rule on near concepts needs an example"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
