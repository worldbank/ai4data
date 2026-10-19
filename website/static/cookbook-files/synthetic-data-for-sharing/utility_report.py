"""Compare a synthetic table with the real one on marginals, associations,
and a target analysis.

Reads the real and the synthetic CSV (same columns) and reports:

    marginals      for each categorical column the total variation
                   distance between the two distributions; for each
                   numeric column the difference of the medians and of
                   the interquartile ranges relative to the real ones
    associations   for each pair of categorical columns the difference
                   in Cramér's V; for numeric pairs the difference in
                   Pearson correlation
    target         a statistic users compute (here the employment rate
                   by sex and the median income by education) in both
                   files side by side

Columns with at most 12 distinct values are treated as categorical.
Empty values are their own category for categorical columns and are
dropped for numeric ones. Uses pandas and scipy (Cramér's V).

Usage:
    python utility_report.py real_sample.csv synthetic_sample.csv

What this does not do: these are general-purpose measures. The utility
that matters is whether the analyses users will run give the same
answers, which the target section shows for two statistics; the
organization adds the analyses its users run. A synthetic file that
scores well on marginals and badly on associations is the usual
failure of independent synthesis, which the running example shows.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations

import pandas as pd
from scipy.stats.contingency import association

CATEGORICAL_MAX = 12


def kinds(df: pd.DataFrame) -> dict[str, str]:
    """Categorical when few distinct values or non-numeric; numeric otherwise."""
    out = {}
    for c in df.columns:
        present = df[c][df[c] != ""]
        numeric = (
            len(present) > 0 and pd.to_numeric(present, errors="coerce").notna().all()
        )
        out[c] = (
            "categorical"
            if df[c].nunique() <= CATEGORICAL_MAX or not numeric
            else "numeric"
        )
    return out


def tvd(a, b) -> float:
    """Total variation distance between two categorical distributions (0 identical, 1 disjoint)."""
    pa, pb = (
        pd.Series(list(a)).value_counts(normalize=True),
        pd.Series(list(b)).value_counts(normalize=True),
    )
    return 0.5 * (pa.subtract(pb, fill_value=0).abs().sum())


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    return (
        association(table.to_numpy(), method="cramer") if min(table.shape) > 1 else 0.0
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("real")
    parser.add_argument("synthetic")
    args = parser.parse_args(argv)
    real, synth = (
        pd.read_csv(p, dtype=str, keep_default_na=False)
        for p in (args.real, args.synthetic)
    )
    kind = kinds(real)
    cats = [c for c, k in kind.items() if k == "categorical"]
    nums = [c for c, k in kind.items() if k == "numeric"]
    num = {
        name: df[nums].apply(pd.to_numeric, errors="coerce")
        for name, df in (("real", real), ("synthetic", synth))
    }
    print(f"real {len(real)} rows, synthetic {len(synth)} rows, {len(kind)} columns\n")

    print("marginals")
    for c in kind:
        if kind[c] == "categorical":
            print(f"  {c:<12} categorical  TVD {tvd(real[c], synth[c]):.3f}")
        else:
            qa, qb = (
                num["real"][c].quantile([0.25, 0.5, 0.75]),
                num["synthetic"][c].quantile([0.25, 0.5, 0.75]),
            )
            med = (qb[0.5] - qa[0.5]) / qa[0.5] if qa[0.5] else 0.0
            iqr_a, iqr_b = qa[0.75] - qa[0.25], qb[0.75] - qb[0.25]
            print(
                f"  {c:<12} numeric      median diff {med:+.1%}, IQR diff {(iqr_b - iqr_a) / iqr_a if iqr_a else 0:+.1%}"
            )

    print("\nassociations (real vs synthetic)")
    for a, b in combinations(cats, 2):
        va, vb = cramers_v(real[a], real[b]), cramers_v(synth[a], synth[b])
        print(
            f"  {a} x {b:<12} Cramer's V {va:.2f} vs {vb:.2f}{'  <- association lost' if va - vb > 0.1 else ''}"
        )
    for a, b in combinations(nums, 2):
        ra, rb = (
            num["real"][[a, b]].corr().iloc[0, 1],
            num["synthetic"][[a, b]].corr().iloc[0, 1],
        )
        print(f"  {a} x {b:<12} correlation {ra:+.2f} vs {rb:+.2f}")

    print("\ntarget analyses")
    for sex in ("1", "2"):
        for name, df, numbers in (
            ("real", real, num["real"]),
            ("synthetic", synth, num["synthetic"]),
        ):
            group = df[(df["sex"] == sex) & numbers["age"].between(15, 64)]
            rate = (group["lfs_status"] == "1").mean() if len(group) else 0.0
            print(
                f"  employment rate, sex {sex}, ages 15-64: {name:<9} {rate:.3f} (n={len(group)})"
            )
    for educ in ("0", "1", "2", "3"):
        medians = {
            name: numbers.loc[df["educ"] == educ, "income"].median()
            for name, df, numbers in (
                ("real", real, num["real"]),
                ("synthetic", synth, num["synthetic"]),
            )
        }
        print(
            f"  median income, education {educ}: real {medians['real']:.0f}, synthetic {medians['synthetic']:.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
