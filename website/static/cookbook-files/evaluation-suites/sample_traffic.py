"""Draw the monthly review sample from logged traffic, stratified.

Reads a traffic log (CSV: query_id, timestamp, language, intent,
answered yes|no, user_rating up|down|empty) and draws a review sample
with a fixed seed: every language and intent is represented in
proportion to traffic with a minimum per stratum, and every query the
system declined or a user rated down is included, because those are
where the errors are. Writes the sample and prints what it contains.
Uses pandas.

Usage:
    python sample_traffic.py traffic_log.csv --size 12 --min-per-stratum 1 -o review_sample.csv

What this does not do: a sample reviewed by people finds the errors in
the sample. The rates it yields apply to the month's traffic with the
uncertainty of its size, and the oversampled declined and down-rated
queries are reported separately so that they do not inflate the error
rate of ordinary traffic.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log")
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--min-per-stratum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("-o", "--output")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.log, dtype=str).fillna("")

    flagged = (df["answered"] == "no") | (df["user_rating"] == "down")
    always = df[flagged].assign(reason="declined or rated down")
    pool = df[~flagged]
    remaining = max(args.size - len(always), 0)

    drawn = []
    for (language, intent), group in pool.groupby(["language", "intent"], sort=True):
        share = round(
            remaining
            * (df[["language", "intent"]].eq([language, intent]).all(axis=1)).sum()
            / len(df)
        )
        take = min(len(group), max(args.min_per_stratum, share))
        drawn.append(
            group.sample(n=take, random_state=args.seed).assign(
                reason=f"stratum {language}/{intent}"
            )
        )
    sample = pd.concat([always, *drawn]).sort_values("query_id")
    if args.output:
        sample.to_csv(args.output, index=False)

    print(
        f"{len(df)} logged queries -> sample of {len(sample)}: {len(always)} declined or rated down (always included), {len(sample) - len(always)} drawn by stratum"
    )
    by = sample.groupby(["language", "intent"]).size()
    print(
        "by stratum: "
        + ", ".join(f"{lang}/{intent} {n}" for (lang, intent), n in by.items())
    )
    print(
        "report the error rate of the stratified part and of the flagged part separately"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
