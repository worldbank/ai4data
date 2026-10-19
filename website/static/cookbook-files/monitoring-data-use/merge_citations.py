"""Merge citation events from identifier services and text mining into one
list of citing documents per dataset.

Reads citation events (CSV: source, dataset_id, dataset_doi, citing_id,
citing_title, citing_year, relation) collected from DataCite, Crossref,
OpenAlex, or any Scholix-conformant service, and from the mention
extraction of this cookbook, and merges them: one row per dataset and
citing document, with the sources that found it. DOIs are compared
case-insensitively. Reports per dataset the unique citing documents, how
many each source found, how many were found by one source only (the ones
to check), and the overlap between the identifier services and text
mining. Uses pandas.

Usage:
    python merge_citations.py citation_events.csv

What this does not do: it merges by identifier. Two records of the same
paper with different identifiers (a preprint and the journal version) stay
separate until a mapping says otherwise; the review in the harmonization
chapter handles that. A citation event says a document references the
dataset; what it did with the data is the typology's question.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

IDENTIFIER_SOURCES = {"datacite", "crossref", "openalex"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("events")
    args = parser.parse_args(argv)
    events = pd.read_csv(args.events, dtype=str).fillna("")
    events["citing_key"] = events["citing_id"].str.strip().str.lower()
    sources = sorted(events["source"].unique())

    # one row per (dataset, citing document) with the set of sources that found it
    merged = events.groupby(["dataset_id", "citing_key"]).agg(
        title=("citing_title", "first"),
        year=("citing_year", "first"),
        sources=("source", lambda s: frozenset(s)),
    )
    merged["n_sources"] = merged["sources"].map(len)
    found_by = pd.DataFrame(
        {s: merged["sources"].map(lambda ss, s=s: s in ss) for s in sources}
    )

    print(
        f"{len(events)} events -> {len(merged)} unique citing documents across {merged.index.get_level_values(0).nunique()} datasets\n"
    )
    print(
        f"{'dataset':<22} {'unique':>6} "
        + " ".join(f"{s:>12}" for s in sources)
        + f" {'one source':>11}"
    )
    for dataset, g in merged.groupby(level=0):
        per = found_by.loc[g.index].sum()
        print(
            f"{dataset:<22} {len(g):>6} "
            + " ".join(f"{per[s]:>12}" for s in sources)
            + f" {(g['n_sources'] == 1).sum():>11}"
        )

    ids = merged["sources"].map(lambda ss: bool(ss & IDENTIFIER_SOURCES))
    mined = merged["sources"].map(lambda ss: "text-mining" in ss)
    print(
        f"\nfound through identifiers {ids.sum()}, through text mining {mined.sum()}, by both {(ids & mined).sum()}, by text mining only {(mined & ~ids).sum()}"
    )
    single = merged[merged["n_sources"] == 1]
    if not single.empty:
        print(
            "\nfound by one source only (check the record and add the identifier where it is missing):"
        )
        for (dataset, _), m in single.iterrows():
            print(
                f"  {dataset}: {m.title} ({m.year}) found by {next(iter(m.sources))} only"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
