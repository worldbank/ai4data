"""Report the use of each dataset from harmonized mentions.

Reads harmonized mentions (JSON lines with dataset_id, is_used,
usage_context, as written by harmonize_mentions.py) and a documents table
(CSV: document_id, title, year, type, source, country) and prints, per
dataset: the number of documents that mention it, the number that use it
(is_used true), and the split by year and by document type. Mentions that
matched no canonical name are reported as other organizations' data, which
shows what the organization's data are used together with. Uses pandas.

Usage:
    python use_report.py harmonized.jsonl documents.csv

What this does not do: the counts cover the documents that were collected.
A dataset used in sources outside the collection does not appear, and a
mention the extractor missed is not counted; the report is read with the
coverage of chapter 2 and the recall of chapter 3 in mind.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("harmonized")
    parser.add_argument("documents")
    args = parser.parse_args(argv)
    docs = pd.read_csv(args.documents, dtype=str).fillna("")
    mentions = pd.read_json(args.harmonized, lines=True)
    mentions = mentions.merge(
        docs[["document_id", "year", "type"]], on="document_id", how="left"
    ).fillna({"year": "?", "type": "?"})

    own = mentions[mentions["dataset_id"].notna() & (mentions["dataset_id"] != "")]
    external = mentions[~mentions.index.isin(own.index)]
    per_dataset = own.groupby("dataset_id").agg(
        mentioned=("document_id", "nunique"),
        used=(
            "document_id",
            lambda d: d[own.loc[d.index, "is_used"].astype(bool)].nunique(),
        ),
    )
    per_dataset = per_dataset.sort_values(
        ["used", "mentioned"], ascending=False, kind="stable"
    )

    print(
        f"{len(docs)} documents, {len(mentions)} mentions, {len(per_dataset)} datasets of the organization mentioned\n"
    )
    print(f"{'dataset':<22} {'mentioned in':>12} {'used in':>8}  by year / by type")
    for ds, r in per_dataset.iterrows():
        g = own[own["dataset_id"] == ds]
        years = ", ".join(
            f"{y}: {n}" for y, n in g.groupby("year")["document_id"].nunique().items()
        )
        types = ", ".join(
            f"{t}: {n}" for t, n in g.groupby("type")["document_id"].nunique().items()
        )
        print(f"{ds:<22} {r.mentioned:>12} {r.used:>8}  {years} / {types}")
    if not external.empty:
        print("\nother organizations' data mentioned alongside (co-use):")
        for name, n in (
            external.groupby("text")["document_id"]
            .nunique()
            .sort_values(ascending=False, kind="stable")
            .items()
        ):
            print(f"  {name!r}: {n} document(s)")
    print(
        "\nCounts cover the collected documents only; read them with the coverage of the sources and the recall of the extractor."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
