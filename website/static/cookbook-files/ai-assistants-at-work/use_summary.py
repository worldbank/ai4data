"""Summarize the use log of everyday assistants for a unit and the board.

Reads the use log (one row per use, with the unit, the role, the task,
the tool, the data tier of the text entered, who reviewed the output,
and what happened to it) and prints the uses by task, by tool, and by
tier, the share of uses with a named reviewer, and the outcomes. It
reports a use whose tier was confidential and whose tool was not the
local assistant, and a use whose tier was internal or confidential at a
hosted assistant. Exit status 1 when any such use exists.

Usage:
    python use_summary.py use_log.csv task_card.csv

Requires pandas.

What this does not check: uses that were not logged. The share of uses
logged is the unit's own estimate, and the session of chapter 6 is where
staff say what they used and did not log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("tasks", type=Path)
    args = parser.parse_args(argv)
    log = pd.read_csv(args.log, dtype=str).fillna("")
    tasks = pd.read_csv(args.tasks, dtype=str).fillna("")
    names = dict(zip(tasks["task_id"], tasks["task"]))

    print(
        f"{len(log)} uses, {log['unit'].nunique()} units, {log['date'].min()} to {log['date'].max()}"
    )
    print("by task:")
    for tid, n in log["task_id"].value_counts().sort_index().items():
        print(f"  {tid}  {n:>3}  {names.get(tid, '(not on the task card)')[:50]}")
    print(
        "by tool: "
        + ", ".join(
            f"{n} {t}" for t, n in log["tool"].value_counts().sort_index().items()
        )
    )
    print(
        "by tier: "
        + ", ".join(
            f"{n} {t}" for t, n in log["data_tier"].value_counts().sort_index().items()
        )
    )
    reviewed = (log["reviewed_by"] != "").sum()
    print(f"reviewed by a named person: {reviewed} of {len(log)}")
    print(
        "outcomes: "
        + ", ".join(
            f"{n} {o}" for o, n in log["outcome"].value_counts().sort_index().items()
        )
    )

    problems = []
    for _, r in log.iterrows():
        if r["data_tier"] == "confidential" and r["tool"] != "local assistant":
            problems.append(
                f"{r['date']} {r['unit']} {r['task_id']}: confidential text at a {r['tool']}"
            )
        elif r["data_tier"] == "internal" and r["tool"] == "hosted assistant":
            problems.append(
                f"{r['date']} {r['unit']} {r['task_id']}: internal text at a hosted assistant"
            )
        if r["task_id"] not in names:
            problems.append(
                f"{r['date']} {r['unit']} {r['task_id']}: task not on the task card"
            )
    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
