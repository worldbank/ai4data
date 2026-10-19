"""Estimate the effort, the roles, and the kinds of cost of the pathway.

Reads the question scores, the targets, and the recipe effort table
(one row per recipe with the dimension it serves, the level it reaches,
a range of person-days, the roles, and the kind of cost) and sums, per
dimension, the recipes that reach a level above the current one and at
or below the target. It prints the range of person-days per dimension
and in total, the roles the pathway needs, and the kinds of cost
beyond staff time. Exit status 1 when a dimension with a gap has no
recipe that serves it.

Usage:
    python effort_estimate.py question_scores.csv targets.csv recipe_effort.csv

Requires pandas, and readiness_profile.py and gap_table.py in the same folder.

What this does not do: the days are written for a first implementation
on the running examples of the cookbooks; a large catalog multiplies
the per-item work, and money is given as kinds of cost because prices
differ too much between countries and years for a figure to hold.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

LETTERS = ["A", "B", "C", "D"]


def load(name: str):
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def estimate(table: pd.DataFrame, effort: pd.DataFrame) -> pd.DataFrame:
    """Recipes selected per dimension: level reached above current and at or below target."""
    effort = effort.copy()
    effort["days_low"] = pd.to_numeric(effort["days_low"])
    effort["days_high"] = pd.to_numeric(effort["days_high"])
    rows = []
    for _, r in table.iterrows():
        if not r["gap"] or r["gap"] <= 0:
            continue
        cur, tgt = LETTERS.index(r["current"]), LETTERS.index(r["target"])
        mine = effort[
            (effort["dimension"] == r["dimension"])
            & (effort["reaches"].map(LETTERS.index) > cur)
            & (effort["reaches"].map(LETTERS.index) <= tgt)
        ]
        for _, e in mine.iterrows():
            rows.append({**e.to_dict(), "current": r["current"], "target": r["target"]})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scores", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("effort", type=Path)
    args = parser.parse_args(argv)
    gt = load("gap_table")
    scores = pd.read_csv(args.scores, dtype=str).fillna("")
    targets = pd.read_csv(args.targets, dtype=str).fillna("")
    effort = pd.read_csv(args.effort, dtype=str).fillna("")
    table = gt.gaps(scores, targets)
    sel = estimate(table, effort)
    problems = []
    print(f"{'dim':<5} {'levels':<8} {'recipes':>7} {'days':>10}  roles")
    for _, r in table.sort_values("dimension").iterrows():
        if not r["gap"] or r["gap"] <= 0:
            continue
        mine = sel[sel["dimension"] == r["dimension"]] if not sel.empty else sel
        if mine.empty:
            problems.append(
                f"{r['dimension']}: no recipe serves the gap {r['current']} -> {r['target']}"
            )
            continue
        roles = sorted({x.strip() for s in mine["roles"] for x in s.split(";")})
        levels = f"{r['current']} -> {r['target']}"
        days = f"{int(mine['days_low'].sum())} to {int(mine['days_high'].sum())}"
        print(
            f"{r['dimension']:<5} {levels:<8} {len(mine):>7} {days:>10}  {'; '.join(roles)[:60]}"
        )
    if not sel.empty:
        print(
            f"\ntotal: {len(sel)} recipes, {int(sel['days_low'].sum())} to {int(sel['days_high'].sum())} person-days"
        )
        outside = sel[sel["cookbook"] == "outside the series"]
        print(
            f"of which outside the cookbook series: {len(outside)} items, {int(outside['days_low'].sum())} to {int(outside['days_high'].sum())} person-days"
        )
        roles = sorted({x.strip() for s in sel["roles"] for x in s.split(";")})
        print(f"roles: {'; '.join(roles)}")
        kinds = sorted(
            {x.strip() for s in sel["cost_kind"] for x in s.split(";")} - {"Staff time"}
        )
        print(f"kinds of cost beyond staff time: {'; '.join(kinds)}")
    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
