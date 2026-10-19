"""Score retrieval runs by slice: language, paraphrase, and run.

Reads retrieval results (CSV: run, query_id, language, paraphrase yes|no,
expected_id, ranked_ids separated by ";") and reports Recall@1, Recall@3,
and MRR for every run overall and per slice, so that a change that
helps English paraphrases and hurts French official titles is visible.
Uses pandas.

Usage:
    python retrieval_by_slice.py retrieval_runs.csv

What this does not do: ten queries per run demonstrate the slices; a
slice needs thirty or more queries before a difference between runs
means anything, and the comparison script of the gates chapter puts an
interval around the difference. Queries with no correct answer (NONE)
are scored by the refusal measures, not here.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def rank_of_expected(row: pd.Series) -> float:
    """1-based rank of the expected id in the ranked list, or NaN when absent."""
    ranked = [x for x in row["ranked_ids"].split(";") if x]
    return (
        ranked.index(row["expected_id"]) + 1
        if row["expected_id"] in ranked
        else float("nan")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.results, dtype=str)
    df = df[df["expected_id"] != "NONE"].copy()
    df["rank"] = df.apply(rank_of_expected, axis=1)
    df["R@1"] = (df["rank"] == 1).astype(float)
    df["R@3"] = (df["rank"] <= 3).astype(float)
    df["MRR"] = (1 / df["rank"]).fillna(0.0)

    # one row per (slice, run): "all", each language, each paraphrase value
    slices = pd.concat(
        [
            df.assign(slice="all"),
            df.assign(slice="language=" + df["language"]),
            df.assign(slice="paraphrase=" + df["paraphrase"]),
        ]
    )
    table = slices.groupby(["slice", "run"], sort=False).agg(
        n=("rank", "size"), r1=("R@1", "mean"), r3=("R@3", "mean"), mrr=("MRR", "mean")
    )
    order = {
        s: i for i, s in enumerate(slices["slice"].unique())
    }  # slices in order of appearance, then runs
    table = table.sort_index(key=lambda idx: idx.map(lambda v: order.get(v, v)))

    runs = sorted(df["run"].unique())
    print(f"{len(df)} scored queries, runs: {', '.join(runs)}\n")
    print(f"{'slice':<16} {'run':<4} {'n':>3} {'R@1':>6} {'R@3':>6} {'MRR':>6}")
    for (name, run), r in table.iterrows():
        print(
            f"{name:<16} {run:<4} {int(r.n):>3} {r.r1:>6.2f} {r.r3:>6.2f} {r.mrr:>6.2f}"
        )
    print(
        "\nA slice with fewer than thirty queries shows direction; the gate script puts an interval around the difference."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
