"""Check that chapters and fixtures agree with the running example's facts.

The running example (the example organization, its series, surveys, and
documents) is shared by every cookbook. Its facts live in
website/static/cookbook-files/_example/facts.json. This check reads every
chapter and every fixture, collects the identifiers they use, and reports:

- an identifier of a series, survey, or document that the facts do not list
  (a typo, or an object one cookbook invented that the others do not know);
- a fixture row that gives a different value for a series and period than
  the facts (columns SERIES/TIME_PERIOD/OBS_VALUE, or series/period/value
  with previous_value for the period before);
- a chapter sentence that quotes a value for a listed series and period
  that differs from the facts.

Exit status 1 on any finding.

Usage:
    python scripts/docs/check_example_facts.py

What this does not do: it knows the identifiers and values the facts file
lists. A new object has to be added there first, which is the point.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FACTS = REPO / "website" / "static" / "cookbook-files" / "_example" / "facts.json"
FILES = REPO / "website" / "static" / "cookbook-files"
ID = re.compile(
    r"\b(?:[A-Z]{2,}_[A-Z0-9_]{3,}|EX-[A-Z0-9]+(?:-[A-Z0-9.]+)*(?:_\w+)?|WB-PRWP-\d+)\b"
)
# column names and codes that look like identifiers and are not objects of the example
NOT_IDS = {
    "OBS_VALUE",
    "OBS_STATUS",
    "REF_AREA",
    "TIME_PERIOD",
    "UNIT_MEASURE",
    "SOURCE_URL",
    "LOW_MIDDLE",
    "STATS_API_URL",
    "CC_BY",
    "ISCO_08",
    "ISIC_REV",
}


def ids_in(text: str) -> set[str]:
    return {
        re.sub(r"_p\d+.*$", "", t.rstrip("."))
        for t in ID.findall(text)
        if t not in NOT_IDS
        and not t.startswith(("OBS_", "REF_", "TIME_", "UNIT_", "SOURCE_"))
    }


def main() -> int:
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    known = (
        set(facts["series"])
        | set(facts["datasets"])
        | set(facts["documents"])
        | set(facts.get("distractors", {}))
    )
    problems: list[str] = []

    sources = [
        p
        for p in (REPO / "cookbook").glob("*/*.mdx")
        if p.parent.name not in ("_template", "_shared")
    ]
    sources += [
        p
        for p in FILES.glob("*/*")
        if p.suffix in (".csv", ".json", ".jsonl", ".md", ".txt")
        and p.parent.name != "_example"
    ]
    for p in sources:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for t in sorted(ids_in(text) - known):
            if re.match(r"EX-|WB-|[A-Z]{2}_", t):
                problems.append(f"{p.relative_to(REPO)}: unknown identifier {t}")

    # values in fixtures
    for p in FILES.glob("*/*.csv"):
        with p.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            sid = r.get("SERIES") or r.get("series")
            if not sid or sid not in facts["series"]:
                continue
            values = facts["series"][sid]["values"]
            period = r.get("TIME_PERIOD") or r.get("period")
            value = r.get("OBS_VALUE") or r.get("value")
            if period in values and value and float(value) != float(values[period]):
                problems.append(
                    f"{p.relative_to(REPO)}: {sid} {period} is {value}; the facts say {values[period]}"
                )
            if r.get("previous_value") and period:
                periods = list(values)
                if period in periods and periods.index(period) > 0:
                    prev = periods[periods.index(period) - 1]
                    if float(r["previous_value"]) != float(values[prev]):
                        problems.append(
                            f"{p.relative_to(REPO)}: {sid} previous value {r['previous_value']}; the facts say {values[prev]} for {prev}"
                        )

    # answer records (JSONL): every expected value is a value of one of the retrieved series
    for p in FILES.glob("*/*.jsonl"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            series = [s for s in r.get("retrieved_ids", []) if s in facts["series"]]
            allowed = {
                float(v) for s in series for v in facts["series"][s]["values"].values()
            }
            if not series:
                continue
            for key in ("expected_values", "retrieved_values"):
                for v in r.get(key, []):
                    if float(v) not in allowed:
                        problems.append(
                            f"{p.relative_to(REPO)}:{i}: {key} has {v}; the facts for {', '.join(series)} do not"
                        )

    # values quoted in chapters: "<series> ... <period> ... <number>" within one sentence is too loose; check "value in <period>" patterns per series name
    for p in (REPO / "cookbook").glob("*/*.mdx"):
        text = re.sub(r"```.*?```", "", p.read_text(encoding="utf-8"), flags=re.DOTALL)
        for sid, info in facts["series"].items():
            for period, value in info["values"].items():
                for m in re.finditer(
                    rf"(\d+(?:\.\d+)?)\s*(?:%|percent)[^.]{{0,60}}\b{re.escape(period)}\b",
                    text,
                ):
                    if (
                        sid in text
                        and info["name"].lower() in text.lower()
                        and float(m.group(1)) != float(value)
                        and float(m.group(1)) < 100
                    ):
                        # only when the sentence names the series
                        sentence = text[max(0, m.start() - 160) : m.end() + 40]
                        if sid in sentence or info["name"].lower() in sentence.lower():
                            problems.append(
                                f"{p.relative_to(REPO)}: quotes {m.group(1)} for {sid} {period}; the facts say {value}"
                            )
    for s in problems:
        print(f"error    {s}")
    print(f"{len(problems)} finding(s) against {FACTS.relative_to(REPO)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
