"""Check every number in an assistant's answer against the source table.

Reads an answer (Markdown or plain text) and a source table (CSV with
value and previous_value columns) and reports each number in the answer
as verified, when it equals a value or a previous value in the table,
or unverified. Reference periods such as 2025-Q2, years, and the ages
in the unit text are left out. Thousands separators are removed before
the comparison. Exit status 1 when any number is unverified.

Usage:
    python check_numbers.py assistant_answer.md release_table.csv

Requires pandas.

What this does not check: whether a verified number is attached to the
right series or period, or whether a claim in words is true. The person
reads the answer against the table for that; the script makes sure
that no figure in it comes from anywhere else.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

NUMBER = re.compile(r"(?<![\w-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w-])")
SKIP = re.compile(
    r"\b(19|20)\d{2}(-Q[1-4])?\b|\b\d+ and over\b|\baged \d+(-\d+)?\b|\bQ[1-4]\b"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("answer", type=Path)
    parser.add_argument("table", type=Path)
    args = parser.parse_args(argv)
    text = SKIP.sub(" ", args.answer.read_text(encoding="utf-8"))
    table = pd.read_csv(args.table, dtype=str)
    known = set()
    for col in ("value", "previous_value"):
        for v in table[col].dropna():
            known.add(float(str(v).replace(",", "")))
    numbers = [m.group(1) for m in NUMBER.finditer(text)]
    unverified = []
    for n in numbers:
        value = float(n.replace(",", ""))
        status = "verified" if value in known else "UNVERIFIED"
        if status != "verified":
            unverified.append(n)
        print(f"  {n:>8}  {status}")
    print(
        f"{len(numbers)} numbers; {len(numbers) - len(unverified)} verified, {len(unverified)} unverified"
    )
    for n in unverified:
        print(f"problem: {n} is not a value or a previous value in the table")
    print("result: " + ("FAIL" if unverified else "PASS"))
    return 1 if unverified else 0


if __name__ == "__main__":
    sys.exit(main())
