"""Build the work manifest for a document-extraction pipeline run.

Reads the document inventory (document_id, title, year, type, pages,
text_layer, tables, figures, extracted, data_sources) and the processing
status log (document_id, stage, status, updated), and writes the manifest
of what the next run does: for each document the next stage that is not
done, in priority order. Stages run in a fixed order: text, layout,
extract, verify, publish. A failed stage is retried and marked as such.
Documents already published are skipped, which makes a run resumable.

Priority is by use first (number of data sources the document feeds),
then by what is missing (a document with no text layer needs OCR, which
is the slow step and is scheduled first so that it runs while the rest is
reviewed), then by size (fewer pages first, so that results arrive early).
Prints the manifest and the totals: pages to OCR, pages to lay out,
tables to extract. Uses pandas.

Usage:
    python build_manifest.py document_inventory.csv processing_status.csv -o manifest.csv

What this does not do: it orders work and counts pages; it does not run
anything. Cost per page comes from the organization's own measurements of
each stage (recipe 9.2), and the priority rule is the one the
organization writes, which this script makes explicit and repeatable.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

STAGES = ["text", "layout", "extract", "verify", "publish"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inventory")
    parser.add_argument("status")
    parser.add_argument("-o", "--output")
    args = parser.parse_args(argv)
    docs = pd.read_csv(args.inventory, dtype=str).fillna("")
    status = pd.read_csv(args.status, dtype=str).pivot_table(
        index="document_id", columns="stage", values="status", aggfunc="last"
    )
    status = (
        status.reindex(columns=STAGES).reindex(docs["document_id"]).fillna("pending")
    )

    rows = []
    for d in docs.itertuples():
        states = status.loc[d.document_id]
        if states["publish"] == "done":
            continue
        next_stage = next(s for s in STAGES if states[s] != "done")
        retry = states[next_stage] == "failed"
        needs_ocr = next_stage == "text" and d.text_layer.strip().lower() == "no"
        sources = [s for s in d.data_sources.split(";") if s.strip()]
        rows.append(
            {
                "document_id": d.document_id,
                "next_stage": next_stage,
                "retry": "yes" if retry else "",
                "ocr": "yes" if needs_ocr else "",
                "pages": int(d.pages),
                "tables": int(d.tables),
                "data_sources": len(sources),
                "reason": ("retry after failure; " if retry else "")
                + ("OCR needed; " if needs_ocr else "")
                + f"feeds {len(sources)} data source(s)",
            }
        )
    manifest = pd.DataFrame(
        rows,
        columns=[
            "document_id",
            "next_stage",
            "retry",
            "ocr",
            "pages",
            "tables",
            "data_sources",
            "reason",
        ],
    )
    manifest = manifest.sort_values(
        ["data_sources", "ocr", "pages"], ascending=[False, False, True], kind="stable"
    )
    manifest.to_csv(args.output or sys.stdout, index=False)

    stage_index = manifest["next_stage"].map(STAGES.index)
    print(
        f"{len(docs)} documents in the inventory, {len(docs) - len(manifest)} published and skipped, {len(manifest)} in the manifest",
        file=sys.stderr,
    )
    print(
        f"pages to OCR {manifest.loc[manifest['ocr'] == 'yes', 'pages'].sum()}, pages to lay out {manifest.loc[stage_index <= 1, 'pages'].sum()}, "
        f"tables to extract {manifest.loc[stage_index <= 2, 'tables'].sum()}, retries {(manifest['retry'] == 'yes').sum()}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
