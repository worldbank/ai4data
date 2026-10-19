"""Validate an automatic grader against human scores and look for bias.

Reads a CSV with one row per graded item (item_id, human_score 0|1,
judge_score 0|1, answer_length in words, position A|B for the order in
which the answer was shown in a pairwise or listed layout) and reports
agreement and Cohen's kappa between the grader and the humans, the
grader's false-pass and false-fail rates, and two bias checks: the pass
rate of the grader by answer length (short versus long) and by position,
compared with the humans' pass rates in the same groups. Uses pandas and scikit-learn (Cohen's kappa).

Usage:
    python judge_agreement.py judge_vs_human.csv

What this does not do: twenty items show the shape of the checks; a
grader is validated on a hundred or more human-scored items per task
and re-validated when the model or the rubric changes. Agreement with
humans bounds the grader at the humans' own consistency, measured with
the labelling chapter's script.
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
    parser.add_argument("scores")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.scores)

    agree = (df["human_score"] == df["judge_score"]).mean()
    kappa = cohen_kappa_score(df["human_score"], df["judge_score"])
    false_pass = int(((df["human_score"] == 0) & (df["judge_score"] == 1)).sum())
    false_fail = int(((df["human_score"] == 1) & (df["judge_score"] == 0)).sum())
    print(
        f"{len(df)} items: agreement {agree:.2f}, kappa {kappa:.2f}; grader passes what humans fail {false_pass}, fails what humans pass {false_fail}"
    )

    df["length"] = (
        df["answer_length"]
        .where(df["answer_length"] < df["answer_length"].median(), "long")
        .mask(lambda s: s != "long", "short")
    )
    groups = {
        "short answers": df[df["length"] == "short"],
        "long answers": df[df["length"] == "long"],
        "position A": df[df["position"] == "A"],
        "position B": df[df["position"] == "B"],
    }
    print(f"\n{'group':<24} {'n':>3} {'human pass':>11} {'grader pass':>12}")
    for name, g in groups.items():
        print(
            f"{name:<24} {len(g):>3} {g['human_score'].mean():>11.2f} {g['judge_score'].mean():>12.2f}"
        )

    def gap(g: pd.DataFrame) -> float:
        return g["judge_score"].mean() - g["human_score"].mean()

    gap_len = gap(groups["long answers"]) - gap(groups["short answers"])
    if gap_len > 0.15:
        print(
            f"\nlength bias: the grader passes long answers {gap_len:.2f} more often than humans do, relative to short ones; add a length rule to the rubric"
        )
    print(
        "\nThe grader is bounded by the humans' own agreement; validate on a hundred items per task and after every rubric or model change."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
