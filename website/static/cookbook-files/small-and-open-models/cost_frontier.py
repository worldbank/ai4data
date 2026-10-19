"""Find the models on the cost-quality frontier for a task.

Reads candidate models (CSV: model, size_b, hosting local|hosted,
suite_score, national_language_score, cost_per_1k_queries,
latency_ms_p50, license) and prints the frontier: the models that no
other model beats on both the suite score and the cost. Then applies a
required score (overall and on the national language) and prints the
cheapest model that meets it, with the saving against the best-scoring
model. Uses pandas.

Usage:
    python cost_frontier.py candidates.csv --required 0.90 --required-national 0.85

What this does not do: scores come from the organization's own suite on
its own questions; the numbers here are illustrative. Cost per query is
measured, including local server and staff time, as the running chapter
describes; a frontier drawn from list prices is a sketch.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def on_frontier(df: pd.DataFrame) -> pd.Series:
    """True for candidates no other candidate beats on both score and cost."""
    score, cost = df["suite_score"].to_numpy(), df["cost_per_1k_queries"].to_numpy()
    dominated = [
        ((score >= s) & (cost <= c) & ((score > s) | (cost < c))).any()
        for s, c in zip(score, cost)
    ]
    return ~pd.Series(dominated, index=df.index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("candidates")
    parser.add_argument(
        "--required", type=float, default=0.0, help="required suite score"
    )
    parser.add_argument(
        "--required-national",
        type=float,
        default=0.0,
        help="required score on the national language",
    )
    args = parser.parse_args(argv)
    df = pd.read_csv(args.candidates)
    df["frontier"] = on_frontier(df)

    print(
        f"{len(df)} candidates; {int(df['frontier'].sum())} on the cost-quality frontier\n"
    )
    print(
        f"{'model':<14} {'hosting':<7} {'score':>6} {'national':>8} {'cost/1k':>8} {'p50 ms':>7}  {'frontier':<8} licence"
    )
    for r in df.sort_values("cost_per_1k_queries", kind="stable").itertuples():
        print(
            f"{r.model:<14} {r.hosting:<7} {r.suite_score:>6.2f} {r.national_language_score:>8.2f} {r.cost_per_1k_queries:>8.2f} {r.latency_ms_p50:>7}  {'yes' if r.frontier else '':<8} {r.license}"
        )
    eligible = df[
        (df["suite_score"] >= args.required)
        & (df["national_language_score"] >= args.required_national)
    ]
    if eligible.empty:
        print(
            f"\nno candidate meets the required scores ({args.required:.2f} overall, {args.required_national:.2f} national); raise the model size or improve the adaptation"
        )
        return 1
    cheapest = eligible.loc[eligible["cost_per_1k_queries"].idxmin()]
    best = df.loc[df["suite_score"].idxmax()]
    print(
        f"\ncheapest candidate meeting {args.required:.2f} overall and {args.required_national:.2f} national: {cheapest.model} at {cheapest.cost_per_1k_queries:.2f} per 1,000 queries"
    )
    print(
        f"best-scoring candidate: {best.model} at {best.cost_per_1k_queries:.2f}; choosing the cheapest passing model saves {(1 - cheapest.cost_per_1k_queries / best.cost_per_1k_queries):.0%} per query"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
