"""Assign records to training, validation, and test splits by group.

Reads a CSV, assigns each record to a split by hashing a group key (for
example the household identifier), so that all records of a group fall in
one split and the assignment is the same on every run and every machine.
Writes a file with one row per record (identifier and split), and prints
the share of records per split, a check that no group appears in two
splits, and the distribution of a stratifying variable in each split.
Uses pandas and scikit-learn (GroupShuffleSplit).

Usage:
    python make_splits.py data.csv --id record_id --group hhid \
        --stratify region --test 0.2 --validation 0.1 -o splits.csv

With --time-column and --test-from, records whose value in that column is
at or after the given value go to the test split and the rest are split
by group between training and validation.

What this does not do: it does not balance the splits by label or by
stratifier (a grouped split cannot, exactly), it does not check for
leakage through other columns (a derived variable that encodes the
label), and it does not decide the split sizes. The chapter explains the
choices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def assign(
    df: pd.DataFrame,
    group_col: str,
    test: float,
    validation: float,
    seed: int,
    time_col: str | None,
    test_from: str | None,
) -> pd.Series:
    """A split name per record: whole groups go together, drawn with a fixed seed."""
    split = pd.Series("train", index=df.index)
    rest = df
    if time_col and test_from:
        is_test = df[time_col] >= test_from
        split[is_test] = "test"
        rest = df[~is_test]
    elif test > 0:
        _, test_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed).split(
                rest, groups=rest[group_col]
            )
        )
        split.iloc[test_idx] = "test"
        rest = df[split == "train"]
    if validation > 0 and len(rest):
        share = validation / (1 - test) if not (time_col and test_from) else validation
        _, val_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=share, random_state=seed + 1).split(
                rest, groups=rest[group_col]
            )
        )
        split[rest.index[val_idx]] = "validation"
    return split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data")
    parser.add_argument("--id", required=True, help="column that identifies a record")
    parser.add_argument(
        "--group",
        required=True,
        help="column whose values must not be split (household, firm, document)",
    )
    parser.add_argument(
        "--stratify", help="column whose distribution is reported per split"
    )
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--validation", type=float, default=0.1)
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed; change it to draw a different assignment",
    )
    parser.add_argument("--time-column", help="column for a time-based test split")
    parser.add_argument(
        "--test-from", help="records with time-column >= this value form the test split"
    )
    parser.add_argument("-o", "--output", default="splits.csv")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.data, dtype=str)
    df["split"] = assign(
        df,
        args.group,
        args.test,
        args.validation,
        args.seed,
        args.time_column,
        args.test_from,
    )

    counts = df["split"].value_counts()
    print(
        f"{len(df)} records, {df[args.group].nunique()} groups by '{args.group}', seed {args.seed}"
    )
    for name in ("train", "validation", "test"):
        print(
            f"  {name:<11} {counts.get(name, 0):>5}  {counts.get(name, 0) / len(df):5.2f}"
        )
    leaks = int((df.groupby(args.group)["split"].nunique() > 1).sum())
    print(f"groups in more than one split: {leaks}" + (" (leakage)" if leaks else ""))

    if args.stratify:
        print(f"distribution of '{args.stratify}' per split (share of the split):")
        table = pd.crosstab(
            df[args.stratify], df["split"], normalize="columns"
        ).reindex(columns=["train", "validation", "test"], fill_value=0.0)
        print(
            "  "
            + f"{'category':<12}"
            + "".join(f"{name:>12}" for name in table.columns)
        )
        for cat, row in table.iterrows():
            print(f"  {cat:<12}" + "".join(f"{v:>12.2f}" for v in row))

    df[[args.id, "split"]].to_csv(args.output, index=False)
    print(f"splits written to {Path(args.output).name}")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
