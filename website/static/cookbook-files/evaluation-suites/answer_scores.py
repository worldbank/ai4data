"""Score a run of generated answers on four measures.

Reads answers (JSON lines: question_id, language, expected_behaviour
answer|decline, expected_values, retrieved_ids, retrieved_values, answer,
cited_ids, declined) and reports:

    numeric accuracy    share of answered questions whose expected value
                        appears in the answer (French decimals and
                        thousands separators are normalized)
    citation validity   share of cited identifiers that were among the
                        retrieved records
    refusal accuracy    share of questions that should be declined and
                        were, and share that should be answered and were
    unsupported numbers share of numbers in answers that are neither a
                        retrieved value nor a year, a date, a "per 1,000"
                        denominator, or an identifier: the working
                        definition of a hallucinated figure

Each measure is also reported per language. Uses pandas.

Usage:
    python answer_scores.py answers_run.jsonl

What this does not do: eight answers demonstrate the measures. A number
that matches a retrieved value can still be the wrong value for the
question (the right series, the wrong period), which the grader of the
judges chapter and the human review catch. Units and caveats are not
scored here.
"""

from __future__ import annotations

import argparse
import math
import re
import sys

import pandas as pd

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:[ ,]\d{3})+|\d+)(?:[.,](\d+))?(?![\w])")
IGNORE = re.compile(
    r"\d{4}-\d{2}-\d{2}|per\s+[\d,. ]+\b|\b[A-Z][A-Z0-9_]{3,}\b"
)  # dates, denominators, identifiers
MEASURES = [
    ("numeric", "numeric accuracy"),
    ("citation", "citation validity"),
    ("declined_when_expected", "refusal accuracy"),
    ("answered_when_expected", "answer rate when expected"),
    ("unsupported", "unsupported numbers"),
]


def numbers(text: str) -> list[float]:
    """Numbers in a text, in reading order, with thousands separators and decimal commas handled."""
    out = []
    for whole, frac in NUMBER.findall(IGNORE.sub(" ", text)):
        whole = re.sub(r"[ ,]", "", whole)
        out.append(float(f"{whole}.{frac}") if frac else float(whole))
    return out


def is_year(v: float) -> bool:
    return 1900 <= v <= 2100 and v.is_integer()


def score(r: dict) -> tuple[list[tuple[str, float]], list[str]]:
    """Return (measure, value) pairs and findings for one answer."""
    checks: list[tuple[str, float]] = []
    findings: list[str] = []
    found = numbers(r["answer"])
    if r["expected_behaviour"] == "answer":
        hit = (not r["declined"]) and any(
            math.isclose(v, e) for e in r["expected_values"] for v in found
        )
        checks += [
            ("numeric", float(hit)),
            ("answered_when_expected", float(not r["declined"])),
        ]
        if not hit:
            findings.append(
                f"{r['question_id']}: expected {r['expected_values']}, answer gave {found}"
            )
    else:
        checks.append(("declined_when_expected", float(r["declined"])))
        if not r["declined"]:
            findings.append(
                f"{r['question_id']}: should have declined, answered {r['answer'][:60]!r}"
            )
    for cid in r["cited_ids"]:
        valid = cid in r["retrieved_ids"]
        checks.append(("citation", float(valid)))
        if not valid:
            findings.append(f"{r['question_id']}: cited {cid}, which was not retrieved")
    supported = set(r["retrieved_values"]) | set(r["expected_values"])
    for v in found:
        if is_year(v):
            continue
        ok = any(math.isclose(v, s) for s in supported)
        checks.append(("unsupported", float(not ok)))
        if not ok:
            findings.append(
                f"{r['question_id']}: number {v:g} is not a retrieved value"
            )
    return checks, findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("answers")
    args = parser.parse_args(argv)
    answers = pd.read_json(args.answers, lines=True).to_dict("records")

    records, findings = [], []
    for r in answers:
        checks, found = score(r)
        findings += found
        records += [
            {"language": r["language"], "measure": m, "value": v} for m, v in checks
        ]
    df = pd.DataFrame(records)
    by_slice = pd.concat(
        [df.assign(slice="all"), df.assign(slice="language=" + df["language"])]
    )
    table = by_slice.groupby(["slice", "measure"], sort=False)["value"].agg(
        ["mean", "size"]
    )

    print(f"{len(answers)} answers\n")
    print(f"{'slice':<14}" + "".join(f"{label:>26}" for _, label in MEASURES))
    for s in by_slice["slice"].unique():
        cells = []
        for key, _ in MEASURES:
            cells.append(
                f"{table.loc[(s, key), 'mean']:.2f} (n={int(table.loc[(s, key), 'size'])})"
                if (s, key) in table.index
                else "-"
            )
        print(f"{s:<14}" + "".join(f"{c:>26}" for c in cells))
    if findings:
        print("\nfindings:")
        for f in findings:
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
