"""Check the AI component register for completeness and consistency.

Reads the register (CSV, one row per component) and reports a row
without an owner, an exit plan, an approval date, or a next review
date; a row whose data are confidential and leave the organization;
a risk tier outside Low, Medium, High; and, with --as-of, the rows
whose review is due on or before that date. Exit status 1 when any
row has a problem (a due review is a notice, not a problem).

Usage:
    python check_register.py component_register.csv --as-of 2026-01-15

Requires pandas.

What this does not check: whether the owner is the right person, or
whether the exit plan would work. The replacement drill of the
providers chapter tests the plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED = [
    "component_id",
    "component",
    "purpose",
    "owner",
    "model_or_service",
    "provider",
    "data_used",
    "data_sensitivity",
    "human_review",
    "external_transfer",
    "risk_tier",
    "approved_on",
    "last_reviewed",
    "next_review",
    "exit_plan",
]
TIERS = {"Low", "Medium", "High"}
SENSITIVITY = {"Public", "Internal", "Confidential"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("register", type=Path)
    parser.add_argument("--as-of", help="date (YYYY-MM-DD) for the review check")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.register, dtype=str).fillna("")
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        print(f"missing columns: {', '.join(missing_cols)}")
        print("result: FAIL")
        return 1

    problems: list[str] = []
    for _, row in df.iterrows():
        cid = row["component_id"]
        for col in ("owner", "exit_plan", "approved_on", "next_review"):
            if not row[col].strip():
                problems.append(f"{cid}: {col} is empty")
        if row["risk_tier"] not in TIERS:
            problems.append(
                f"{cid}: risk_tier {row['risk_tier']!r} is not Low, Medium, or High"
            )
        if row["data_sensitivity"] not in SENSITIVITY:
            problems.append(
                f"{cid}: data_sensitivity {row['data_sensitivity']!r} is not Public, Internal, or Confidential"
            )
        if row["data_sensitivity"] == "Confidential" and row[
            "external_transfer"
        ].lower().startswith("y"):
            problems.append(f"{cid}: confidential data leave the organization")
        if "no" in row["human_review"].lower()[:3] and row["risk_tier"] != "Low":
            problems.append(
                f"{cid}: no human review on a {row['risk_tier']} tier component"
            )

    by_provider = df.groupby("provider")["component_id"].count().to_dict()
    print(
        f"{len(df)} components; "
        + ", ".join(f"{n} {p}" for p, n in sorted(by_provider.items()))
        + "; "
        + ", ".join(
            f"{(df['risk_tier'] == t).sum()} {t}" for t in ("Low", "Medium", "High")
        )
    )
    if args.as_of:
        due = df[df["next_review"] <= args.as_of].sort_values("next_review")
        print(f"reviews due on or before {args.as_of}: {len(due)}")
        for _, row in due.iterrows():
            print(
                f"  {row['component_id']} {row['component']}: due {row['next_review']}, owner {row['owner']}"
            )
    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
