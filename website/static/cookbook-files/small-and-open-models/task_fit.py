"""Say, task by task, whether a small or open model is the first choice.

Reads a task list (CSV: task, output code|short text|paragraph|long
text|vector|spans|structured, reasoning none|low|medium|high, languages
separated by ";", confidential yes|no, volume_per_month, latency
batch|interactive) and applies the decision rule of this guide:

    confidential text   a model the organization runs (open weights),
                        whatever else is true
    output code, spans, vector, short text, with low or no reasoning
                        a small model, local, tested on the suite
    paragraph with medium reasoning
                        a mid-sized open model first; a large hosted
                        model as the comparison
    long text or high reasoning, interactive
                        a large model first; a mid-sized open model as
                        the comparison, where cost or language favours it
    high volume         local favoured on cost at any size that passes

Prints the first choice and the comparison per task with the reasons,
so that the evaluation of the comparison chapter has its candidates.
Uses pandas.

Usage:
    python task_fit.py tasks.csv

What this does not do: the rule names where to start; the suite decides.
A small model that passes the suite is the choice regardless of the
rule, and a large model that fails on the national language is not.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

SMALL_OUTPUTS = {"code", "spans", "vector", "short text", "structured"}


def decide(t: pd.Series) -> tuple[str, str, str]:
    """First choice, the model to compare it with, and the reasons, from the task's properties."""
    reasons = []
    confidential = t["confidential"].strip().lower() == "yes"
    volume = int(t["volume_per_month"])
    if confidential:
        reasons.append("confidential text stays on local open weights")
    if t["output"] in SMALL_OUTPUTS and t["reasoning"] in {"none", "low", "medium"}:
        first, compare = "small open model, local", "mid-sized open model"
        reasons.append(
            f"{t['output']} output with {t['reasoning']} reasoning suits a small model"
        )
    elif t["output"] == "paragraph" and t["reasoning"] in {"low", "medium"}:
        first, compare = "mid-sized open model, local", "large hosted model"
        reasons.append("paragraph output with medium reasoning")
    else:
        first, compare = (
            ("large open model, local" if confidential else "large hosted model"),
            "mid-sized open model",
        )
        reasons.append(
            f"{t['output']} output with {t['reasoning']} reasoning starts large"
        )
    if volume >= 100000:
        reasons.append(f"{volume:,} per month favours local on cost")
    if "national" in t["languages"]:
        reasons.append("national language: test coverage on the suite")
    return first, compare, "; ".join(reasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tasks")
    args = parser.parse_args(argv)
    tasks = pd.read_csv(args.tasks, dtype=str).fillna("")
    decisions = tasks.apply(decide, axis=1, result_type="expand").set_axis(
        ["first", "compare", "why"], axis=1
    )

    print(f"{len(tasks)} tasks\n")
    print(f"{'task':<32} {'first choice':<30} {'compare with':<24} reasons")
    for task, d in zip(tasks["task"], decisions.itertuples()):
        print(f"{task:<32} {d.first:<30} {d.compare:<24} {d.why}")
    small = int(decisions["first"].str.startswith("small").sum())
    print(
        f"\n{small} of {len(tasks)} tasks start with a small open model; every first choice is tested against its comparison on the suite"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
