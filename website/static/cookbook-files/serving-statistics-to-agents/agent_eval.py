"""Score an agent's logged runs against a question set.

Reads a question set (CSV: question_id, language, question, expected_tools
separated by ";", expected_series, expected_value, expected_behaviour
answer|decline) and the traces an agent harness logged (JSON lines:
question_id, tools_called, series, answer, declined, latency_s), and
reports per question and in total:

    tools     every expected tool was called
    series    the series the agent used is the expected one
    value     the expected value appears in the answer (within tolerance)
    cite      the answer contains a source URL
    behave    the agent answered when it should and declined when it should

Exit status is 0 when every check passes, 1 otherwise. Uses pandas.

What this does not do: it scores what the trace records. Whether the
answer's wording is correct beyond the value and the citation, whether
caveats were repeated, and whether the tool calls were efficient are
judged by a reviewer on a sample. Eight questions give a demonstration; a
suite has 30 to 50 per language.

Usage:
    python agent_eval.py agent_questions.csv agent_traces.jsonl
"""

from __future__ import annotations

import argparse
import math
import re
import sys

import pandas as pd

NUMBER = re.compile(r"-?\d[\d,]*(?:[.,]\d+)?")
URL = re.compile(r"https?://\S+")
CHECKS = ("tools", "series", "value", "cite", "behave")


def numbers_in(text: str) -> list[float]:
    """Numbers in a text; '6,3' (a French decimal) is 6.3 and '12,500' is 12500."""
    out = []
    for raw in NUMBER.findall(text):
        cleaned = raw.replace(" ", "")
        cleaned = (
            cleaned.replace(",", ".")
            if re.fullmatch(r"-?\d+,\d{1,2}", cleaned)
            else cleaned.replace(",", "")
        )
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def judge(q: pd.Series, t: dict, tolerance: float) -> dict[str, bool | None]:
    """Apply the five checks to one question and its trace; None means the check does not apply."""
    expected_tools = [x for x in q["expected_tools"].split(";") if x]
    should_decline = q["expected_behaviour"] == "decline"
    r: dict[str, bool | None] = {
        "tools": all(tool in t.get("tools_called", []) for tool in expected_tools),
        "behave": bool(t.get("declined")) == should_decline,
        "series": None,
        "value": None,
        "cite": None,
    }
    if not should_decline:
        r["series"] = (
            (t.get("series") == q["expected_series"]) if q["expected_series"] else None
        )
        if q["expected_value"]:
            target = float(q["expected_value"])
            r["value"] = any(
                math.isclose(n, target, abs_tol=tolerance)
                for n in numbers_in(t.get("answer", ""))
            )
        r["cite"] = bool(URL.search(t.get("answer", "")))
    return r


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("questions")
    parser.add_argument("traces")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    questions = pd.read_csv(args.questions, dtype=str).fillna("")
    traces = {
        t["question_id"]: t
        for t in pd.read_json(args.traces, lines=True).to_dict("records")
    }

    results, failures, latencies = [], [], []
    cell = lambda v: "-" if v is None else ("ok" if v else "FAIL")
    print(
        f"{'id':<4} {'lang':<5} {'tools':<6} {'series':<7} {'value':<6} {'cite':<5} {'behave':<7} question"
    )
    for _, q in questions.iterrows():
        t = traces.get(q["question_id"])
        if t is None:
            failures.append(f"{q['question_id']}: no trace")
            continue
        latencies.append(float(t.get("latency_s", 0)))
        r = judge(q, t, args.tolerance)
        results.append(r)
        failures += [
            f"{q['question_id']}: {c} failed ({q['note'] or q['question']})"
            for c in CHECKS
            if r[c] is False
        ]
        print(
            f"{q['question_id']:<4} {q['language']:<5} {cell(r['tools']):<6} {cell(r['series']):<7} {cell(r['value']):<6} {cell(r['cite']):<5} {cell(r['behave']):<7} {q['question']}"
        )

    print()
    scores = pd.DataFrame(results)
    for c in CHECKS:
        applicable = scores[c].notna().sum()
        if applicable:
            print(
                f"{c:<7} {int(scores[c].sum())}/{applicable}  {scores[c].sum() / applicable:.2f}"
            )
    if latencies:
        s = pd.Series(latencies)
        print(f"latency median {s.median():.1f}s, max {s.max():.1f}s")
    if failures:
        print("\nfailures:")
        for f in failures:
            print(f"  {f}")
    print(
        "\nNot checked here: wording, caveats, and efficiency of the tool calls; a reviewer samples those."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
