"""Convert an extracted chart or table into a tidy CSV with provenance.

One row per value, with the columns that let any later user trace the
value back to the page: document_id, page, bbox, class, title, row,
column, value, unit, status. The status column records the verification
result when a verification report from verify_extraction.py is given;
otherwise every value is "extracted".

Uses pandas.

Usage:
    python snapshot_to_tidy.py example_table.json -o T3.1.csv
    python verify_extraction.py example_table.json > report.txt
    python snapshot_to_tidy.py example_table.json --verification report.txt -o T3.1.csv

What this does not do: it does not change values. A flagged value stays in
the file with status "flagged" so that the correction is a documented
edit, with the page open.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPORT_LINE = re.compile(r"^\s+(verified|flagged|estimated)\s+(\S.*?)\s{2,}")
FIELDS = [
    "document_id",
    "page",
    "bbox",
    "class",
    "title",
    "row",
    "column",
    "value",
    "unit",
    "status",
]


def read_statuses(report: Path | None) -> dict[str, str]:
    """Map 'row/column' to its status from the output of verify_extraction.py."""
    if report is None:
        return {}
    matches = (
        REPORT_LINE.match(line)
        for line in report.read_text(encoding="utf-8").splitlines()
    )
    return {m.group(2).strip(): m.group(1) for m in matches if m}


def rows_for(x: dict, statuses: dict[str, str]) -> list[dict]:
    """One row per value of a chart or a table record, with its provenance."""
    base = {
        "document_id": x["document_id"],
        "page": x["page"],
        "bbox": " ".join(map(str, x["bbox"])),
        "class": x["class"],
        "title": x.get("title", ""),
    }
    if x["class"].lower() == "figure":
        cells = [
            (cat, s["name"], v, x.get("unit", ""))
            for s in x["series"]
            for cat, v in zip(x["categories"], s["values"])
        ]
    else:
        units = x.get("units", {})
        cells = [
            (r[0], col, r[j], units.get(col, ""))
            for r in x["rows"]
            for j, col in enumerate(x["columns"][1:], start=1)
        ]
    return [
        {
            **base,
            "row": row,
            "column": col,
            "value": v,
            "unit": unit,
            "status": statuses.get(
                f"{row}/{col}" if x["class"].lower() != "figure" else f"{col}/{row}",
                "extracted",
            ),
        }
        for row, col, v, unit in cells
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("extraction", type=Path)
    parser.add_argument(
        "--verification", type=Path, help="output of verify_extraction.py"
    )
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)

    x = json.loads(args.extraction.read_text(encoding="utf-8"))
    tidy = pd.DataFrame(rows_for(x, read_statuses(args.verification)), columns=FIELDS)
    tidy.to_csv(args.output or sys.stdout, index=False)
    if args.output:
        print(f"wrote {args.output}: {len(tidy)} values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
