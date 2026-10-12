"""Compare two readiness profiles, before and after the pathway.

Reads two question score files (the first assessment and the
re-assessment) and the targets, and prints, per dimension, the level
and average before and after, whether the target was reached, and the
number of questions that moved up or down. It ends with the pillar and
overall scores for both. Exit status 1 when the two files do not have
the same questions.

Usage:
    python compare_profiles.py question_scores.csv scores_after.csv targets.csv

Requires pandas and readiness_profile.py in the same folder.

What this does not do: the framework warns that scores across versions
of the framework are not directly comparable, and that readiness scores
are baselines for action; this comparison assumes the same version and
reads the change as a planning result.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("targets", type=Path)
    args = parser.parse_args(argv)
    rp = load_profile_module()
    b = pd.read_csv(args.before, dtype=str).fillna("")
    a = pd.read_csv(args.after, dtype=str).fillna("")
    if set(b["question_id"]) != set(a["question_id"]):
        print("problem: the two files do not have the same questions")
        print("result: FAIL")
        return 1
    targets = dict(
        zip(
            *[
                pd.read_csv(args.targets, dtype=str)[c]
                for c in ("dimension", "target_level")
            ]
        )
    )
    pb, pa = rp.profile(b), rp.profile(a)
    merged = pb.merge(
        pa, on=["dimension", "pillar", "dimension_name"], suffixes=("_before", "_after")
    )
    b["num"] = pd.to_numeric(b["score"], errors="coerce")
    a["num"] = pd.to_numeric(a["score"], errors="coerce")
    q = b[["question_id", "dimension", "num"]].merge(
        a[["question_id", "num"]], on="question_id", suffixes=("_b", "_a")
    )
    q["delta"] = q["num_a"] - q["num_b"]
    reached = 0
    print(
        f"{'dim':<5} {'before':>12} {'after':>12} {'target':>6} {'reached':>7} {'up':>3} {'down':>4}  name"
    )
    for _, r in merged.iterrows():
        t = targets.get(r["dimension"], "")
        ok = t and LETTERS.index(r["level_after"]) >= LETTERS.index(t)
        reached += bool(ok)
        qq = q[q["dimension"] == r["dimension"]]
        before = f"{r['level_before']} ({r['average_before']:.2f})"
        after = f"{r['level_after']} ({r['average_after']:.2f})"
        print(
            f"{r['dimension']:<5} {before:>12} {after:>12} "
            f"{t:>6} {'yes' if ok else 'no':>7} {int((qq['delta'] > 0).sum()):>3} {int((qq['delta'] < 0).sum()):>4}  {r['dimension_name'][:40]}"
        )
    print(
        f"\ntargets reached: {reached} of {len(merged)}; questions moved up: {int((q['delta'] > 0).sum())}, down: {int((q['delta'] < 0).sum())}"
    )
    for label, prof in (("before", pb), ("after", pa)):
        pillars = prof.groupby("pillar")["average"].mean()
        print(
            f"{label}: pillar 1 {pillars.get('1', float('nan')):.2f}, pillar 2 {pillars.get('2', float('nan')):.2f}, overall {pillars.mean() / 3 * 100:.1f} of 100"
        )
    print("result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
