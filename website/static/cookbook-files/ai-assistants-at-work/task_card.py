"""Print the one-page task card for a role from the task list, and check the list.

Reads the task list (one row per role and task, with the role of the
model, whether the task is allowed, the highest data tier the task may
use, and the review the person makes) and prints the card for one role
or for every role: the tasks allowed with their tier and review, then
the tasks that stay with the person. Exit status 1 when a row has an
unknown model role, an allowed value outside yes and no, a tier outside
public, internal, and confidential, or an allowed task with no review.

Usage:
    python task_card.py task_card.csv --role "Data curator"
    python task_card.py task_card.csv --all

Requires pandas.

What this does not do: it prints the list the organization agreed. Whether
the list is right for a unit is the unit's session of chapter 6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MODEL_ROLES = {
    "draft",
    "summarize",
    "translate",
    "explain",
    "classify",
    "code",
    "decide",
}
TIERS = {"public", "internal", "confidential"}


def check(df: pd.DataFrame) -> list[str]:
    problems = []
    for _, r in df.iterrows():
        tid = r["task_id"]
        if r["model_role"] not in MODEL_ROLES:
            problems.append(
                f"{tid}: model role {r['model_role']!r} is not one of the seven"
            )
        if r["allowed"] not in {"yes", "no"}:
            problems.append(f"{tid}: allowed must be yes or no")
        if r["allowed"] == "yes" and r["data_tier"] not in TIERS:
            problems.append(
                f"{tid}: data tier {r['data_tier']!r} is not public, internal, or confidential"
            )
        if r["allowed"] == "yes" and not r["review"].strip():
            problems.append(f"{tid}: allowed task without a review")
    return problems


def card(df: pd.DataFrame, role: str) -> str:
    rows = df[df["role"].isin([role, "All staff"])]
    out = [f"Task card: {role}", ""]
    out.append("An assistant may help with:")
    for _, r in rows[rows["allowed"] == "yes"].iterrows():
        tier = {
            "public": "public text only",
            "internal": "internal text, no confidential records",
            "confidential": "local assistant only",
        }[r["data_tier"]]
        out.append(f"  {r['task_id']}  {r['task']} ({r['model_role']}; {tier})")
        out.append(f"       review: {r['review']}")
    out.append("")
    out.append("Stays with the person:")
    for _, r in rows[rows["allowed"] == "no"].iterrows():
        out.append(f"  {r['task_id']}  {r['task']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tasks", type=Path)
    parser.add_argument(
        "--role", help="print the card for this role (plus the tasks for all staff)"
    )
    parser.add_argument("--all", action="store_true", help="print every role's card")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.tasks, dtype=str).fillna("")
    problems = check(df)
    roles = [r for r in df["role"].unique() if r != "All staff"]
    if args.all:
        for role in roles:
            print(card(df, role))
            print()
    elif args.role:
        if args.role not in roles:
            print(f"problem: no role named {args.role!r}; roles are {', '.join(roles)}")
            return 1
        print(card(df, args.role))
    print(
        f"\n{len(df)} tasks for {len(roles)} roles and all staff; {int((df['allowed'] == 'yes').sum())} allowed, {int((df['allowed'] == 'no').sum())} stay with the person"
    )
    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
