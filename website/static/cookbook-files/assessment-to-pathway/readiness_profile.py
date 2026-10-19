"""Compute a readiness profile from question-level scores.

Reads the question scores of an AI-readiness assessment (one row per
question with a score of 0 to 3 for the answers A to D, or NA) and
prints the dimension averages with a maturity letter, the pillar
averages, and the overall score on a 0 to 100 scale, following the
framework's scoring: a dimension is the unweighted average of its
applicable questions, a pillar the average of its dimensions, and the
overall score the average of the two pillars scaled to 100. The letter
is the level nearest the average (A below 0.5, B to 1.5, C to 2.5, D
above). Exit status 1 when a score is outside 0 to 3.

Usage:
    python readiness_profile.py question_scores.csv

Requires pandas.

What this does not do: the framework's own scoring scripts are the
reference for the index and for how not-applicable questions are
treated; this script reproduces the published rule for planning and
reports the letters as a convention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

LETTERS = ["A", "B", "C", "D"]


def level(avg: float) -> str:
    """The maturity letter nearest to a dimension average."""
    if avg < 0.5:
        return "A"
    if avg < 1.5:
        return "B"
    if avg < 2.5:
        return "C"
    return "D"


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per dimension: pillar, name, questions, applicable, average, level."""
    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    g = df.groupby(["dimension", "pillar", "dimension_name"], sort=True)
    out = g.agg(
        questions=("question_id", "count"),
        applicable=("score", "count"),
        average=("score", "mean"),
    ).reset_index()
    out["level"] = out["average"].map(level)
    out["lowest"] = g["score"].apply(lambda s: int((s == 0).sum())).values
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scores", type=Path)
    args = parser.parse_args(argv)
    df = pd.read_csv(args.scores, dtype=str).fillna("")
    numeric = pd.to_numeric(df["score"], errors="coerce")
    bad = df[
        (df["score"] != "")
        & (df["score"].str.upper() != "NA")
        & (numeric.isna() | (numeric < 0) | (numeric > 3))
    ]
    for _, row in bad.iterrows():
        print(
            f"problem: {row['question_id']}: score {row['score']!r} is not 0 to 3 or NA"
        )
    if not bad.empty:
        print("result: FAIL")
        return 1

    prof = profile(df)
    print(f"{'dim':<5} {'dimension':<48} {'q':>3} {'avg':>5}  level  at A")
    for _, r in prof.iterrows():
        print(
            f"{r['dimension']:<5} {r['dimension_name'][:48]:<48} {r['applicable']:>3} {r['average']:>5.2f}  {r['level']:<5}  {r['lowest']}"
        )
    pillars = prof.groupby("pillar")["average"].mean()
    print()
    for p, v in pillars.items():
        print(f"pillar {p}: average {v:.2f} ({level(v)})")
    overall = pillars.mean() / 3 * 100
    print(f"overall: {overall:.1f} of 100")
    print(
        f"questions scored A: {int(prof['lowest'].sum())} of {int(prof['applicable'].sum())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
