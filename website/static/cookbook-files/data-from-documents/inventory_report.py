"""Report on a document inventory: where the data are, and what is left to do.

Reads an inventory CSV with one row per document (document_id, title,
year, type, pages, text_layer, tables, figures, extracted, data_sources)
and prints counts by type and year, the number of tables and figures, the
share of documents with a text layer (documents without one need OCR
before any text-based extraction), and the extraction backlog ordered by
the number of tables and figures. Uses pandas.

Usage:
    python inventory_report.py document_inventory.csv

What this does not do: it reports what the inventory says. Counting tables
and figures per document is the work of the layout detection in chapter 2;
until then the counts are estimates entered by hand.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inventory")
    args = parser.parse_args(argv)
    docs = pd.read_csv(args.inventory, dtype=str).fillna("")
    if docs.empty:
        sys.exit("no documents in the inventory")
    for col in ("tables", "figures"):
        docs[col] = pd.to_numeric(docs[col], errors="coerce").fillna(0).astype(int)
    docs["has_text"] = docs["text_layer"].str.strip().str.lower() == "yes"
    docs["extracted"] = docs["extracted"].str.strip().str.lower()

    n = len(docs)
    print(
        f"{n} documents, {docs['tables'].sum()} tables, {docs['figures'].sum()} figures; {docs['has_text'].sum()} of {n} with a text layer"
    )

    by_type = (
        docs.groupby("type")
        .agg(
            documents=("document_id", "size"),
            tables=("tables", "sum"),
            figures=("figures", "sum"),
        )
        .sort_values("tables", ascending=False)
    )
    print(f"\n{'type':<22} {'docs':>5} {'tables':>7} {'figures':>8}")
    for t, c in by_type.iterrows():
        print(f"{t:<22} {c.documents:>5} {c.tables:>7} {c.figures:>8}")

    status = docs["extracted"].value_counts().sort_index()
    print("\nextraction status: " + ", ".join(f"{k} {v}" for k, v in status.items()))

    backlog = (
        docs[docs["extracted"] != "yes"]
        .assign(size=lambda d: d["tables"] + d["figures"])
        .sort_values("size", ascending=False, kind="stable")
    )
    print("\nbacklog (largest first):")
    for d in backlog.itertuples():
        print(
            f"  {d.document_id:<18} {d.year}  {d.tables:>4} tables {d.figures:>3} figures  {d.extracted}{'' if d.has_text else '  needs OCR'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
