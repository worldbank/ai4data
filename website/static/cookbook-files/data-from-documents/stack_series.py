"""Combine tidy extractions from several documents into one series.

Reads tidy CSV files (the output of snapshot_to_tidy.py: one row per value
with document_id, page, title, row, column, value, unit, status) and a
mapping file that says which table in which document feeds which series,
which of the table's dimensions is the area and which the period, and
which unit labels mean the same unit. Writes one series file with one row
per area and period, the value, the source document and page, and a
status:

    single      the value appears in one document
    confirmed   two documents report the same value for the area and period
    revised     a later document reports a different value; the later
                value is kept, the earlier one is recorded beside it

Documents are ordered by their year in the mapping's document identifier
order of appearance, earliest first. Prints the counts and the revisions.
Uses pandas.

Usage:
    python stack_series.py series_mapping.csv tidy_fs2024.csv tidy_fs2025.csv -o series_fs.csv

What this does not do: it matches areas by exact label and periods by
exact string; a renamed region or a fiscal-year period needs a mapping
row. It keeps the latest value of a revision, which is right for a
revised statistic and wrong for a definitional change; the documents'
notes say which, and the series record carries the break.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

OUT = [
    "series_id",
    "area",
    "period",
    "value",
    "unit",
    "source_document",
    "source_page",
    "status",
    "previous_value",
    "previous_document",
]


def map_rows(tidy: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Attach series, area, and period to each tidy row through the mapping; count rows no mapping covers."""
    mapped = []
    unmatched = 0
    for r in tidy.itertuples():
        m = mapping[
            (mapping["document_id"] == r.document_id)
            & mapping["title_prefix"].map(r.title.startswith)
        ]
        if m.empty:
            unmatched += 1
            continue
        m = m.iloc[0]
        aliases = {a.strip().lower() for a in m["unit_aliases"].split(";")}
        if r.unit.strip().lower() not in aliases:
            unmatched += 1
            continue
        area, period = (
            (r.row, r.column) if m["row_role"] == "area" else (r.column, r.row)
        )
        mapped.append(
            {
                "series_id": m["series_id"],
                "area": area,
                "period": period,
                "value": r.value,
                "unit": m["unit_label"],
                "source_document": r.document_id,
                "source_page": r.page,
            }
        )
    return pd.DataFrame(mapped), unmatched


def resolve(mapped: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest value per series, area, and period; mark it confirmed or revised when an earlier document reported it."""
    out = []
    for _, g in mapped.groupby(["series_id", "area", "period"], sort=True):
        first, last = g.iloc[0], g.iloc[-1]
        row = last.to_dict() | {
            "status": "single",
            "previous_value": "",
            "previous_document": "",
        }
        if len(g) > 1:
            if first["value"] == last["value"]:
                row["status"] = "confirmed"
            else:
                row |= {
                    "status": "revised",
                    "previous_value": first["value"],
                    "previous_document": first["source_document"],
                }
        out.append(row)
    return pd.DataFrame(out, columns=OUT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mapping", type=Path)
    parser.add_argument("tidy", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    mapping = pd.read_csv(args.mapping, dtype=str).fillna("")
    order = {
        d: i for i, d in enumerate(mapping["document_id"].unique())
    }  # documents in the mapping's order, earliest first
    tidy = pd.concat(
        [pd.read_csv(p, dtype=str).fillna("") for p in args.tidy], ignore_index=True
    )
    tidy = tidy.sort_values("document_id", key=lambda s: s.map(order), kind="stable")

    mapped, unmatched = map_rows(tidy, mapping)
    series = resolve(mapped)
    series.to_csv(args.output or sys.stdout, index=False)

    counts = series["status"].value_counts()
    print(
        f"{len(tidy)} tidy rows from {len(args.tidy)} documents -> {len(series)} series values; {unmatched} rows unmatched",
        file=sys.stderr,
    )
    print(
        ", ".join(
            f"{k} {counts.get(k, 0)}" for k in ("single", "confirmed", "revised")
        ),
        file=sys.stderr,
    )
    for r in series[series["status"] == "revised"].itertuples():
        print(
            f"revised: {r.area} {r.period}: {r.previous_value} ({r.previous_document}) -> {r.value} ({r.source_document})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
