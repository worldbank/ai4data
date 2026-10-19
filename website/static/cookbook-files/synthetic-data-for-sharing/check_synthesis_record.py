"""Check a synthesis record for the sections a synthetic release needs.

Reads a synthesis record (Markdown) and reports whether each required
section is present (source data, method and parameters, utility results,
disclosure risk results, intended uses, prohibited uses and limits,
labelling and licence, contact), whether the method section names a seed,
whether the risk section names a reviewer or a review date, and whether
the labelling section says the file is marked SYNTHETIC. Exit status 1
when anything is missing. Standard library only.

Usage:
    python check_synthesis_record.py synthesis_record.md

What this does not do: it checks presence. Whether the utility is enough
for the intended uses and the risk low enough for the release tier are
the decisions of the methodologist and the disclosure control unit,
recorded in the sections the check looks for.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED = [
    "Source data",
    "Method and parameters",
    "Utility results",
    "Disclosure risk results",
    "Intended uses",
    "Prohibited uses and limits",
    "Labelling and licence",
    "Contact",
]


def section(text: str, name: str) -> str:
    m = re.search(
        r"^##\s+" + re.escape(name) + r"\s*$(.*?)(?=^##\s|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return m.group(1) if m else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("record", type=Path)
    args = parser.parse_args(argv)
    text = args.record.read_text(encoding="utf-8")
    missing = [r for r in REQUIRED if not section(text, r).strip()]
    for r in REQUIRED:
        print(f"  {'ok     ' if r not in missing else 'MISSING'} {r}")
    checks = {
        "seed stated in the method": bool(
            re.search(
                r"\bseed\b", section(text, "Method and parameters"), re.IGNORECASE
            )
        ),
        "risk reviewed by a named unit or on a date": bool(
            re.search(
                r"reviewed|\d{4}-\d{2}-\d{2}",
                section(text, "Disclosure risk results"),
                re.IGNORECASE,
            )
        ),
        "file labelled SYNTHETIC": "SYNTHETIC"
        in section(text, "Labelling and licence"),
    }
    for name, ok in checks.items():
        print(f"  {'ok     ' if ok else 'MISSING'} {name}")
    failed = bool(missing) or not all(checks.values())
    print("result: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
