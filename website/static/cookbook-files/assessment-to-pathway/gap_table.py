"""Size the gap between the current level and the target level of each dimension.

Reads the question scores and the targets (one target letter per
dimension with a reason) and prints, per dimension, the current level,
the target, the gap in levels, the number of questions below the
target's numeric value, and the reason. Dimensions are ordered by gap
and then by the number of questions below target. Exit status 1 when a
dimension has no target or a target is below the current level.

Usage:
    python gap_table.py question_scores.csv targets.csv

Requires pandas and readiness_profile.py in the same folder.

What this does not check: whether the targets are the right ones. The
reason column is where the organization says why, and the sequence of
the next chapter says when.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

LETTERS = ["A", "B", "C", "D"]


def load_profile_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "readiness_profile", here / "readiness_profile.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gaps(scores: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rp = load_profile_module()
    prof = rp.profile(scores)
    t = dict(zip(targets["dimension"], targets["target_level"]))
    reasons = dict(zip(targets["dimension"], targets["reason"]))
    scores = scores.copy()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    rows = []
    for _, r in prof.iterrows():
        target = t.get(r["dimension"], "")
        gap = LETTERS.index(target) - LETTERS.index(r["level"]) if target else None
        below = (
            int(
                (
                    scores.loc[scores["dimension"] == r["dimension"], "score"]
                    < LETTERS.index(target)
                ).sum()
            )
            if target
            else None
        )
        rows.append(
            {
                "dimension": r["dimension"],
                "name": r["dimension_name"],
                "current": r["level"],
                "average": r["average"],
                "target": target,
                "gap": gap,
                "below_target": below,
                "reason": reasons.get(r["dimension"], ""),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["gap", "below_target", "dimension"], ascending=[False, False, True]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scores", type=Path)
    parser.add_argument("targets", type=Path)
    args = parser.parse_args(argv)
    scores = pd.read_csv(args.scores, dtype=str).fillna("")
    targets = pd.read_csv(args.targets, dtype=str).fillna("")
    table = gaps(scores, targets)
    problems = []
    print(f"{'dim':<5} {'current':>10} {'target':>6} {'gap':>4} {'below':>6}  name")
    for _, r in table.iterrows():
        if r["target"] == "":
            problems.append(f"{r['dimension']}: no target")
            continue
        if r["gap"] < 0:
            problems.append(
                f"{r['dimension']}: target {r['target']} is below the current level {r['current']}"
            )
        current = f"{r['current']} ({r['average']:.2f})"
        print(
            f"{r['dimension']:<5} {current:>10} {r['target']:>6} {r['gap']:>4} {r['below_target']:>6}  {r['name'][:44]}"
        )
    with_gap = table[table["gap"].fillna(0) > 0]
    print(
        f"\ndimensions with a gap: {len(with_gap)} of {len(table)}; questions below target: {int(table['below_target'].fillna(0).sum())}"
    )
    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
