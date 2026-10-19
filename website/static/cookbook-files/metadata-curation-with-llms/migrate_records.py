"""Migrate legacy catalog rows into schema-shaped records through a field mapping.

Reads a legacy catalog (CSV with the organization's own column names) and
a field mapping (CSV: legacy_column, schema_field, transform, confidence,
status, note) that says which schema field each legacy column feeds and
how its values are transformed:

    codes:A=annual;Q=quarterly   replace legacy codes by schema values
    split:;                      split a delimited string into a list
    date:%d/%m/%Y                parse a date and write it as YYYY-MM-DD

A mapping row can be ``confirmed`` by a curator or ``proposed`` by a
model; proposed rows are applied and reported, so that the curator sees
what the migration would do before confirming. Columns mapped to no
field are reported as unplaced, with their values, and values the
transform cannot handle (an unknown code, an unparseable date) are left
as they are and reported. Writes one JSON record per row in the shape of
the World Bank indicator schema and prints a report. Uses pandas.

Usage:
    python migrate_records.py legacy_catalog.csv field_mapping.csv -o migrated_records.json

What this does not do: it moves and reshapes values. Whether a legacy
"Description" is the schema's definition or its methodology is the
curator's confirmation on the mapping row, and the migrated records go
through the completeness check of the dissemination cookbook and the
review of this cookbook before publication.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def apply(transform: str, value: str, problems: list[str], where: str):
    """Apply one transform ('codes:A=annual;Q=quarterly', 'split:;', 'date:%d/%m/%Y') to one value."""
    if not transform or value == "":
        return value
    kind, _, spec = transform.partition(":")
    if kind == "codes":
        codes = dict(pair.split("=", 1) for pair in spec.split(";") if "=" in pair)
        if value in codes:
            return codes[value]
        problems.append(f"{where}: unknown code {value!r} (known: {', '.join(codes)})")
        return value
    if kind == "split":
        return [part.strip() for part in value.split(spec) if part.strip()]
    if kind == "date":
        parsed = pd.to_datetime(value, format=spec, errors="coerce")
        if pd.isna(parsed):
            problems.append(f"{where}: date {value!r} does not match {spec}")
            return value
        return parsed.date().isoformat()
    problems.append(f"{where}: unknown transform {transform!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("legacy")
    parser.add_argument("mapping")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    legacy = pd.read_csv(args.legacy, dtype=str).fillna("")
    mapping = pd.read_csv(args.mapping, dtype=str).fillna("")
    placed = mapping[mapping["schema_field"] != ""]
    unplaced = mapping[mapping["schema_field"] == ""]

    problems: list[str] = []
    empty: Counter = Counter()
    records = []
    for _, row in legacy.iterrows():
        key = row.get("Code") or row.get(mapping["legacy_column"].iloc[0], "?")
        rec = {}
        for m in placed.itertuples():
            value = apply(
                m.transform,
                row.get(m.legacy_column, ""),
                problems,
                f"{key}/{m.legacy_column}",
            )
            if value in ("", []):
                empty[m.schema_field] += 1
            else:
                rec[m.schema_field] = value
        records.append(rec)
    if args.output:
        args.output.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    proposed = placed[placed["status"] == "proposed"]
    print(
        f"{len(legacy)} legacy rows -> {len(records)} records; {len(mapping)} mapping rows: {(mapping['status'] == 'confirmed').sum()} confirmed, {len(proposed) + len(unplaced)} proposed"
    )
    if not proposed.empty:
        print("\nproposed mappings applied, to confirm or change:")
        for m in proposed.itertuples():
            print(
                f"  {m.legacy_column!r} -> {m.schema_field} (confidence {m.confidence}): {m.note}"
            )
    if not unplaced.empty:
        print("\nunplaced columns (values not migrated):")
        for m in unplaced.itertuples():
            values = [v for v in legacy.get(m.legacy_column, pd.Series(dtype=str)) if v]
            print(f"  {m.legacy_column!r}: {len(values)} non-empty value(s); {m.note}")
            for v in values:
                print(f"      - {v}")
    if empty:
        print(
            "\nempty after migration: "
            + ", ".join(f"{k} ({n})" for k, n in empty.most_common())
        )
    if problems:
        print("\nvalues left as they were:")
        for p in problems:
            print(f"  {p}")
    if args.output:
        print(f"\nrecords written to {args.output}; run the completeness check next")
    return 0


if __name__ == "__main__":
    sys.exit(main())
