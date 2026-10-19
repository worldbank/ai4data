"""Write the statement of model use for a release from the register and the evaluation summary.

Reads the component register and the evaluation summary (one row per
component, release, and measure) and prints the statement of model use
for one release in Markdown, with the six sections a quality report
needs: where a model was used, models and versions, measured
performance, human oversight, effect on the statistics, contact. With
--check it prints the statement and then verifies that every section is
present, that every use names a GSBPM sub-process, and that the
performance section has at least one measured number. Exit status 1
when --check finds a gap or the release has no evaluation rows.

Usage:
    python statement_of_model_use.py component_register.csv evaluation_summary.csv --release EX-REP-2025-02 --check

Requires pandas.

What this does not do: it cannot judge whether the measured performance
is good enough. The required score of the evaluation suite does that,
and the head of methodology signs the statement.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SECTIONS = [
    "Where a model was used",
    "Models and versions",
    "Measured performance",
    "Human oversight",
    "Effect on the statistics",
    "Contact",
]


def lower_first(text: str) -> str:
    """Lower-case the first character only, so that acronyms keep their case."""
    return text[:1].lower() + text[1:]


def build(reg: pd.DataFrame, ev: pd.DataFrame, release: str, contact: str) -> str:
    rows = ev[ev["release"] == release]
    ids = list(dict.fromkeys(rows["component_id"]))
    comps = reg.set_index("component_id").loc[ids]
    out = [f"# Statement of model use: {release}", ""]
    out += ["## Where a model was used", ""]
    for cid, c in comps.iterrows():
        g = rows[rows["component_id"] == cid]["gsbpm"].iloc[0]
        out.append(f"- {c['component']} (GSBPM {g}): {lower_first(c['purpose'])}.")
    out += ["", "## Models and versions", ""]
    for cid, c in comps.iterrows():
        where = (
            "run on the organization's servers; no record left the organization"
            if c["external_transfer"].lower().startswith("n")
            else f"hosted by {c['provider']}; only {c['data_used'].lower()} were sent"
        )
        out.append(
            f"- {c['component']}: {c['model_or_service']}, version recorded in the AI component register, {where}."
        )
    out += ["", "## Measured performance", ""]
    for _, r in rows.iterrows():
        out.append(
            f"- {comps.loc[r['component_id'], 'component']}: {lower_first(r['measure'])} {r['value']} ({r['sample']})."
        )
    out += ["", "## Human oversight", ""]
    for cid, c in comps.iterrows():
        out.append(f"- {c['component']}: {c['human_review'].rstrip('.')}.")
    out += ["", "## Effect on the statistics", ""]
    effects = rows[rows["effect_on_statistics"] != ""]
    for _, r in effects.iterrows():
        out.append(
            f"- {comps.loc[r['component_id'], 'component']}: {r['effect_on_statistics'].rstrip('.')}."
        )
    out.append(
        "- Automatically accepted outputs entered the statistics under rules the owners approved, "
        "with the measured checks reported above; every other model output entered only after a person's decision."
    )
    out += ["", "## Contact", "", contact, ""]
    return "\n".join(out)


def check(text: str) -> list[str]:
    problems = []
    for s in SECTIONS:
        if not re.search(rf"^## {re.escape(s)}\s*$", text, flags=re.MULTILINE):
            problems.append(f"section missing: {s}")
    uses = (
        text.split("## Where a model was used")[1].split("## Models and versions")[0]
        if "## Models and versions" in text
        else ""
    )
    bullets = re.findall(r"^- (.+)$", uses, flags=re.MULTILINE)
    for b in bullets:
        if not re.search(r"\(GSBPM [\d.]+\)", b):
            problems.append(f"use without a GSBPM sub-process: {b[:50]}")
    perf = (
        text.split("## Measured performance")[1].split("## Human oversight")[0]
        if "## Human oversight" in text
        else ""
    )
    if not re.search(r"\d", perf):
        problems.append("no measured number in the performance section")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("register", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument(
        "--release",
        required=True,
        help="release identifier, for example EX-REP-2025-02",
    )
    parser.add_argument("--contact", default="Methodology unit, methods@stats.example.")
    parser.add_argument(
        "--check", action="store_true", help="verify the statement after writing it"
    )
    args = parser.parse_args(argv)

    reg = pd.read_csv(args.register, dtype=str).fillna("")
    ev = pd.read_csv(args.evaluation, dtype=str).fillna("")
    if ev[ev["release"] == args.release].empty:
        print(f"no evaluation rows for release {args.release}")
        return 1
    text = build(reg, ev, args.release, args.contact)
    print(text)
    if args.check:
        problems = check(text)
        for p in problems:
            print(f"problem: {p}")
        print("check: " + ("FAIL" if problems else "PASS"))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
