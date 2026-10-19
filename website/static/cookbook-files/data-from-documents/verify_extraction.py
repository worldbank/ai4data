"""Verify data extracted from a document against the document itself.

Reads an extraction JSON (a chart snapshot or a table, see the example
files) and runs the checks that the document makes possible:

  chart   each extracted value is matched to a value label printed on the
          chart (the `printed_values` list, read from the snapshot image);
          charts without printed labels are reported as estimated, which
          this check cannot verify
  table   each numeric column sums to the total row within a tolerance;
          each derived column (a rate computed from two other columns)
          matches the computed value within a tolerance

Every value gets a status: verified, flagged, or estimated. Exit status is
0 when nothing is flagged, 1 otherwise. Uses pandas.

What this does not check: whether the extraction picked the right table or
chart, whether the row and column labels are correct, or whether a value
without a printed label, a total, or a derivation is right. Those need a
person with the page open; recipe 4.2 samples them.

Usage:
    python verify_extraction.py example_snapshot.json
    python verify_extraction.py example_table.json --abs-tolerance 0.5 --rel-tolerance 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

Result = tuple[str, str, str]  # (location, status, detail)


def close(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def verify_chart(x: dict) -> list[Result]:
    """Check each chart value against the labels printed on the chart, where there are any."""
    cells = [
        (f"{s['name']}/{cat}", v)
        for s in x["series"]
        for cat, v in zip(x["categories"], s["values"])
    ]
    if not x.get("value_labels_printed"):
        return [
            (where, "estimated", f"{v}: no printed label; read from the axis")
            for where, v in cells
        ]
    remaining = list(x.get("printed_values", []))
    results: list[Result] = []
    for where, v in cells:
        if v in remaining:
            remaining.remove(v)
            results.append((where, "verified", f"{v} matches a printed label"))
        else:
            results.append((where, "flagged", f"{v} is not among the printed labels"))
    results += [
        ("printed label", "flagged", f"{left} printed on the chart but not extracted")
        for left in remaining
    ]
    return results


def verify_table(x: dict, abs_tol: float, rel_tol: float) -> list[Result]:
    """Check derived columns by recomputing them and other columns against the total row."""
    table = pd.DataFrame(x["rows"], columns=x["columns"]).set_index(x["columns"][0])
    total_label = x.get("total_row")
    body = table.drop(index=total_label) if total_label in table.index else table
    results: list[Result] = []
    for col in table.columns:
        if not pd.api.types.is_numeric_dtype(table[col]):
            continue
        if col in x.get("derived", {}):
            spec = x["derived"][col]
            computed = (
                table[spec["numerator"]]
                / table[spec["denominator"]]
                * spec.get("scale", 1)
            )
            for label, v, c in zip(table.index, table[col], computed):
                results.append(
                    (
                        f"{label}/{col}",
                        "verified" if close(v, c, abs_tol, rel_tol) else "flagged",
                        f"{v} vs computed {c:.2f}",
                    )
                )
        elif total_label in table.index:
            total, s = table.at[total_label, col], body[col].sum()
            ok = close(total, s, abs_tol, rel_tol)
            results.append(
                (
                    f"{total_label}/{col}",
                    "verified" if ok else "flagged",
                    f"total {total} vs sum of rows {s:g}",
                )
            )
            results += [
                (
                    f"{label}/{col}",
                    "verified" if ok else "estimated",
                    f"{v} (covered by the total check)",
                )
                for label, v in body[col].items()
            ]
        else:
            results += [
                (
                    f"{label}/{col}",
                    "estimated",
                    f"{v}: no total or derivation to check against",
                )
                for label, v in table[col].items()
            ]
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("extraction", type=Path, help="chart snapshot or table JSON")
    parser.add_argument("--abs-tolerance", type=float, default=0.5)
    parser.add_argument("--rel-tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)

    x = json.loads(args.extraction.read_text(encoding="utf-8"))
    kind = x.get("class", "").lower()
    if kind == "figure":
        results = verify_chart(x)
    elif kind == "table":
        results = verify_table(x, args.abs_tolerance, args.rel_tolerance)
    else:
        sys.exit("the extraction must have class Figure or Table")

    print(f"{x['document_id']} page {x['page']}: {x.get('title', '')} ({kind})")
    for where, status, detail in results:
        print(f"  {status:<9} {where:<32} {detail}")
    counts = pd.Series([s for _, s, _ in results]).value_counts()
    print(
        f"\n{counts.get('verified', 0)} verified, {counts.get('flagged', 0)} flagged, {counts.get('estimated', 0)} estimated"
    )
    print(
        "Not checked here: whether the right table or chart was extracted, or whether labels are correct."
    )
    return 1 if counts.get("flagged", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
