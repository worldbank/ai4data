"""Check a statement of model use for the sections a quality report needs.

Reads a statement of model use (Markdown) and reports whether it has
each required section (where a model was used, models and versions,
measured performance, human oversight, effect on the statistics,
contact), whether every use names its GSBPM sub-process, and whether
the performance section contains at least one measured number. Exit
status 1 when a section is missing. Standard library only.

Usage:
    python check_statement.py model_use_statement.md

What this does not do: it checks presence. Whether the measured
performance is good enough is the required score of the evaluation
suite, and whether the statement is true is the methodology unit's
signature.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED = [
    "Where a model was used",
    "Models and versions",
    "Measured performance",
    "Human oversight",
    "Effect on the statistics",
    "Contact",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("statement", type=Path)
    args = parser.parse_args(argv)
    text = args.statement.read_text(encoding="utf-8")
    sections = {}
    for m in re.finditer(r"^##\s+(.+)$", text, flags=re.MULTILINE):
        sections[m.group(1).strip().lower()] = m.start()
    missing = [r for r in REQUIRED if r.lower() not in sections]
    for r in REQUIRED:
        print(f"  {'ok     ' if r not in missing else 'MISSING'} {r}")
    uses = re.findall(r"^- (.+?)\(GSBPM ([\d.]+)\)", text, flags=re.MULTILINE)
    bullets_in_use = len(
        re.findall(
            r"^- ",
            text[
                sections.get("where a model was used", 0) : sections.get(
                    "models and versions", len(text)
                )
            ],
            flags=re.MULTILINE,
        )
    )
    print(f"uses with a GSBPM sub-process: {len(uses)}/{bullets_in_use}")
    perf = (
        text[
            sections.get("measured performance", 0) : sections.get(
                "human oversight", len(text)
            )
        ]
        if "measured performance" in sections
        else ""
    )
    numbers = re.findall(r"\d+\.\d+|\d+", perf)
    print(f"measured numbers in the performance section: {len(numbers)}")
    failed = bool(missing) or len(uses) < bullets_in_use or not numbers
    print("result: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
