"""Check the evidence register against the questions below target.

Reads the question scores, the targets, and the evidence register
(one row per question with the evidence that will show the question
moved, its owner, its due date, and its status) and reports every
question below its dimension's target that has no evidence row, the
rows without an owner or a due date, and, with --as-of, the rows that
are not done past their due date. Exit status 1 when a question below
target has no evidence row or a row lacks an owner or a date.

Usage:
    python check_evidence.py question_scores.csv targets.csv evidence_register.csv --as-of 2026-01-15

Requires pandas.

What this does not check: whether the evidence would satisfy an
assessor. The framework's response options say what each level looks
like; the register names the document that will show it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

LETTERS = ["A", "B", "C", "D"]
STATUSES = {"planned", "in progress", "done"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scores", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--as-of", help="date (YYYY-MM-DD) for the overdue check")
    args = parser.parse_args(argv)
    scores = pd.read_csv(args.scores, dtype=str).fillna("")
    targets = dict(
        zip(
            *[
                pd.read_csv(args.targets, dtype=str)[c]
                for c in ("dimension", "target_level")
            ]
        )
    )
    ev = pd.read_csv(args.evidence, dtype=str).fillna("")
    scores["num"] = pd.to_numeric(scores["score"], errors="coerce")
    below = scores[
        scores.apply(
            lambda r: r["num"] < LETTERS.index(targets.get(r["dimension"], "A")), axis=1
        )
    ]
    have = set(ev["question_id"])
    problems = []
    missing = below[~below["question_id"].isin(have)]
    for _, r in missing.iterrows():
        problems.append(
            f"{r['question_id']} ({r['topic']}): below target, no evidence row"
        )
    for _, r in ev.iterrows():
        if not r["owner"]:
            problems.append(f"{r['question_id']}: no owner")
        if not r["due"]:
            problems.append(f"{r['question_id']}: no due date")
        if r["status"] not in STATUSES:
            problems.append(
                f"{r['question_id']}: status {r['status']!r} is not planned, in progress, or done"
            )

    print(
        f"questions below target: {len(below)}; with evidence rows: {len(below) - len(missing)}"
    )
    by_dim = ev.groupby("dimension")["status"].value_counts().unstack(fill_value=0)
    for col in ("done", "in progress", "planned"):
        if col not in by_dim.columns:
            by_dim[col] = 0
    print(f"{'dim':<5} {'done':>5} {'in progress':>12} {'planned':>8}")
    for dim, r in by_dim.sort_index().iterrows():
        print(
            f"{dim:<5} {int(r['done']):>5} {int(r['in progress']):>12} {int(r['planned']):>8}"
        )
    if args.as_of:
        late = ev[
            (ev["status"] != "done") & (ev["due"] != "") & (ev["due"] < args.as_of)
        ]
        print(f"\noverdue on {args.as_of}: {len(late)}")
        for _, r in late.iterrows():
            print(
                f"  {r['question_id']}: {r['evidence']} ({r['owner']}, due {r['due']}, {r['status']})"
            )
    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
