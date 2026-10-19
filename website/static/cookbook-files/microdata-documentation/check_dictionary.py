"""Check a data dictionary before it is published.

Reads a data dictionary with one row per variable (CSV) and reports what an
AI system or a user would find missing or misleading:

errors (exit status 1)
  - a variable without a label
  - a label that repeats the variable name
  - duplicate variable names within a file
  - a categorical variable without value labels

warnings
  - a numeric variable without a statement of its missing-value codes
  - a variable without a universe
  - a collected variable without its question text
  - a variable without a concept, where a concept map could link it to a
    classification

Columns expected: file_id, name, label, type (numeric | categorical |
string), universe, question, values ("code=label;code=label"), missing
("code;code"), concept. Other columns are ignored. Uses pandas.

What this does not check: whether the labels are correct, whether the
value labels match the data, or whether the universe statements agree with
the skip patterns. Those need the data file and the questionnaire.

Usage:
    python check_dictionary.py lfs_2025q2_dictionary.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

FIELDS = ("label", "universe", "question", "values", "missing", "concept")


def is_identifier(name: str) -> bool:
    return name.lower().endswith("id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dictionary", help="CSV with one row per variable")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.dictionary, dtype=str).fillna("")
    if df.empty:
        sys.exit("no variables found")
    for col in FIELDS:
        if col not in df:
            df[col] = ""
    df = df.apply(lambda s: s.str.strip())
    df["type"] = df["type"].str.lower()
    df["ref"] = df["file_id"] + "/" + df["name"]
    df["is_id"] = df["name"].map(is_identifier)

    errors: list[str] = []
    warnings: list[str] = []
    dup = df.groupby("ref").size()
    errors += [f"{ref}: name appears {n} times" for ref, n in dup[dup > 1].items()]
    for r in df.drop_duplicates("ref").itertuples():
        if not r.label:
            errors.append(f"{r.ref}: no label")
        elif r.label.lower() == r.name.lower():
            errors.append(f"{r.ref}: label repeats the name ({r.label!r})")
        if r.type == "categorical" and not r.values:
            errors.append(f"{r.ref}: categorical variable without value labels")
        if r.type == "numeric" and not r.missing and not r.is_id:
            warnings.append(
                f"{r.ref}: numeric variable without a missing-value statement"
            )
        if not r.universe:
            warnings.append(f"{r.ref}: no universe")
        if not r.question and not r.is_id:
            warnings.append(f"{r.ref}: no question or derivation text")
        if r.type == "categorical" and not r.concept and not r.is_id:
            warnings.append(f"{r.ref}: no concept or classification named")

    print(f"{len(df)} variables in {df['file_id'].nunique()} file(s)")
    filled = (df[list(FIELDS)] != "").sum()
    for col, n in filled.items():
        print(f"  {col:<10} {n:>4} of {len(df)}  {n / len(df):>4.0%}")
    print()
    for w in warnings:
        print(f"warning  {w}")
    for e in errors:
        print(f"error    {e}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    print(
        "Not checked here: whether labels and value labels are correct, or whether universes match the skip patterns."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
