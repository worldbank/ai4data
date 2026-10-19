"""Compare the composition of a dataset with population benchmarks.

Reads a dataset (CSV) and a benchmarks file with one row per variable and
category (variable, category, population_share, source). For each
variable it prints the dataset's share of each category, unweighted and
weighted where a weight column is given, next to the population share
and the difference in percentage points. It flags categories whose share
differs from the population by more than a threshold and categories with
fewer records than a minimum. Exit status 1 when any category is below
the minimum count. Uses pandas.

Usage:
    python representativeness_report.py data.csv census_benchmarks.csv \
        --weight weight --min-count 30 --max-diff 5

What this does not do: it compares marginal shares, one variable at a
time. Joint coverage (women aged 55 to 64 in the Northern Region) needs
cross-tabulations the chapter describes, and representativeness of the
label (which occupations appear) is reported separately. A benchmark is
only as current as its source.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data")
    parser.add_argument("benchmarks")
    parser.add_argument("--weight", help="survey weight column")
    parser.add_argument(
        "--min-count",
        type=int,
        default=30,
        help="categories with fewer records are flagged",
    )
    parser.add_argument(
        "--max-diff",
        type=float,
        default=5.0,
        help="difference in percentage points that is flagged",
    )
    args = parser.parse_args(argv)
    data = pd.read_csv(args.data, dtype=str)
    bench = pd.read_csv(args.benchmarks, dtype={"category": str})
    n = len(data)

    print(
        f"{n} records; benchmarks for {bench['variable'].nunique()} variable(s)"
        + (f"; weighted by '{args.weight}'" if args.weight else "")
    )
    small, off = [], []
    for var, b in bench.groupby("variable", sort=False):
        if var not in data:
            print(f"\n{var}: not in the dataset")
            continue
        counts = data[var].value_counts()
        weighted = (
            data.groupby(var)[args.weight].apply(lambda s: pd.to_numeric(s).sum())
            if args.weight
            else None
        )
        table = (
            pd.DataFrame({"n": counts, "dataset": counts / n})
            .join(
                b.set_index("category")["population_share"].rename("population"),
                how="outer",
            )
            .fillna({"n": 0, "dataset": 0.0})
        )
        if args.weight:
            table["weighted"] = (
                (weighted / weighted.sum()).reindex(table.index).fillna(0.0)
            )
        table["diff"] = (table["dataset"] - table["population"]) * 100
        print(f"\n{var}  (benchmark: {b['source'].iloc[0]})")
        print(
            f"  {'category':<10}{'n':>6}{'dataset':>10}"
            + (f"{'weighted':>10}" if args.weight else "")
            + f"{'population':>12}{'diff pp':>9}"
        )
        for cat, r in table.sort_index().iterrows():
            flags = []
            if r["n"] < args.min_count:
                flags.append(f"fewer than {args.min_count} records")
                small.append(f"{var}={cat} ({int(r['n'])})")
            if pd.notna(r["diff"]) and abs(r["diff"]) > args.max_diff:
                flags.append("differs from the population")
                off.append(f"{var}={cat} ({r['diff']:+.1f} pp)")
            pop = f"{r['population']:.3f}" if pd.notna(r["population"]) else "-"
            diff = f"{r['diff']:+.1f}" if pd.notna(r["diff"]) else "-"
            line = f"  {cat:<10}{int(r['n']):>6}{r['dataset']:>10.3f}" + (
                f"{r['weighted']:>10.3f}" if args.weight else ""
            )
            line += f"{pop:>12}{diff:>9}"
            print(line + ("  " + "; ".join(flags) if flags else ""))

    print()
    print(
        f"categories below {args.min_count} records: {len(small)}"
        + (": " + ", ".join(small) if small else "")
    )
    print(
        f"categories more than {args.max_diff:g} pp from the population (unweighted): {len(off)}"
        + (": " + ", ".join(off) if off else "")
    )
    print(
        "A dataset is not a sample of the population; the card states these differences and the weights, and the user decides."
    )
    return 1 if small else 0


if __name__ == "__main__":
    sys.exit(main())
