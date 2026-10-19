"""Check the decision log against the register and list open follow-ups.

Reads the decision log (one row per decision) and the component
register, and reports a component without an approval decision; a
High-tier component whose approval was not decided by the person who
accepts residual risk; an incident without a closing decision; and,
with --as-of, follow-ups that are open past their due date. Exit status
1 when a component lacks approval, a High-tier approval lacks the
accepting role, or an incident is not closed.

Usage:
    python check_decision_log.py decision_log.csv component_register.csv --as-of 2026-01-15

Requires pandas.

What this does not check: whether the decisions were good ones. It
checks that every component and every incident has one, by the role the
terms of reference name.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ACCEPTS_RESIDUAL_RISK = "Director General"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("register", type=Path)
    parser.add_argument("--as-of", help="date (YYYY-MM-DD) for the follow-up check")
    args = parser.parse_args(argv)

    log = pd.read_csv(args.log, dtype=str).fillna("")
    reg = pd.read_csv(args.register, dtype=str).fillna("")
    problems: list[str] = []

    approvals = log[log["type"] == "approval"]
    for _, comp in reg.iterrows():
        mine = approvals[approvals["component_id"] == comp["component_id"]]
        if mine.empty:
            problems.append(f"{comp['component_id']}: no approval decision")
            continue
        if (
            comp["risk_tier"] == "High"
            and not (mine["decided_by"] == ACCEPTS_RESIDUAL_RISK).any()
        ):
            problems.append(
                f"{comp['component_id']}: High tier, approval not decided by the {ACCEPTS_RESIDUAL_RISK}"
            )

    incidents = log[log["type"] == "incident"]
    opened = {}
    closed = set()
    for _, row in incidents.iterrows():
        m = re.search(r"(I-\d{4}-\d+)", row["decision"])
        if not m:
            continue
        if row["decision"].lower().startswith("close"):
            closed.add(m.group(1))
        else:
            opened[m.group(1)] = row["component_id"]
    for inc, cid in sorted(opened.items()):
        if inc not in closed:
            problems.append(f"{inc} ({cid}): incident not closed")

    counts = log["type"].value_counts().to_dict()
    print(
        f"{len(log)} decisions: "
        + ", ".join(
            f"{counts.get(t, 0)} {t}"
            for t in ("policy", "approval", "incident", "review")
        )
    )
    print(
        f"components approved: {len(reg) - sum('no approval' in p for p in problems)}/{len(reg)}"
    )
    print(f"incidents: {len(opened)} opened, {len(closed & set(opened))} closed")

    open_fu = log[(log["closed"].str.lower() != "yes") & (log["follow_up"] != "")]
    print(f"open follow-ups: {len(open_fu)}")
    for _, row in open_fu.iterrows():
        overdue = (
            args.as_of and row["follow_up_due"] and row["follow_up_due"] < args.as_of
        )
        print(
            f"  {row['decision_id']} ({row['component_id']}): {row['follow_up']}, due {row['follow_up_due'] or 'n/a'}"
            + ("  OVERDUE" if overdue else "")
        )

    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
