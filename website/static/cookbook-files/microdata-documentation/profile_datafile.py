"""Draft a data dictionary from a data file.

Reads a CSV data file and writes a draft dictionary with one row per
column: the name, an inferred type (numeric, categorical, or string), the
number of records, the number of empty or coded-missing values, the
distinct codes of a categorical column written as ``code=`` with the label
left for the curator, the candidate missing codes (negative integers), and
the range of a numeric column. Labels, universes, questions, and concepts
are left empty: they come from the questionnaire and the curator, and the
draft shows exactly which ones are needed. Uses pandas.

Usage:
    python profile_datafile.py lfs_2025q2_sample.csv draft_dictionary.csv

Exit status 0. The summary on standard output counts the columns by type
and the fields the curator has to fill.

What this does not do: a column with few distinct integer values is
treated as categorical, which is wrong for a small count variable, and a
code list seen in a sample may miss codes that occur only in the full
file. The Metadata Editor imports Stata, SPSS, and other files with their
value labels, which this script cannot read; use it where only a CSV exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CATEGORICAL_MAX = 12
OUT_FIELDS = [
    "name",
    "label",
    "type",
    "universe",
    "question",
    "values",
    "missing",
    "concept",
    "n",
    "n_missing",
    "range",
]


def profile(column: pd.Series) -> dict[str, str]:
    """Infer a column's type, codes, candidate missing codes, and range from its values."""
    present = column[column.str.strip() != ""]
    as_number = pd.to_numeric(present, errors="coerce")
    numeric = bool(len(present)) and as_number.notna().all()
    missing_codes = (
        sorted(
            {
                v
                for v, x in zip(present, as_number)
                if numeric and x < 0 and float(x).is_integer()
            },
            key=float,
        )
        if numeric
        else []
    )
    valid = present[~present.isin(missing_codes)]
    distinct = sorted(valid.unique(), key=lambda v: (float(v) if numeric else 0, v))
    if (
        numeric
        and len(distinct) <= CATEGORICAL_MAX
        and all(float(v).is_integer() for v in distinct)
    ):
        kind = "categorical"
    elif numeric:
        kind = "numeric"
    elif len(distinct) <= CATEGORICAL_MAX:
        kind = "categorical"
    else:
        kind = "string"
    numbers = pd.to_numeric(valid) if numeric else pd.Series(dtype=float)
    return {
        "type": kind,
        "n": str(len(column)),
        "n_missing": str(len(column) - len(valid)),
        "values": ";".join(f"{v}=" for v in distinct) if kind == "categorical" else "",
        "missing": ";".join(missing_codes),
        "range": f"{numbers.min():g} to {numbers.max():g}"
        if kind == "numeric" and len(numbers)
        else "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("datafile", type=Path)
    parser.add_argument("draft", type=Path, help="where to write the draft dictionary")
    args = parser.parse_args(argv)
    data = pd.read_csv(args.datafile, dtype=str, keep_default_na=False)

    drafts = pd.DataFrame(
        [
            {
                "name": name,
                "label": "",
                "universe": "",
                "question": "",
                "concept": "",
                **profile(data[name]),
            }
            for name in data.columns
        ]
    )[OUT_FIELDS]
    drafts.to_csv(args.draft, index=False)

    by_type = drafts["type"].value_counts()
    codes = int(drafts["values"].str.count("=").sum())
    with_missing = drafts.loc[drafts["missing"] != "", "name"].tolist()
    print(
        f"{len(data)} records, {len(data.columns)} columns: "
        + ", ".join(
            f"{by_type.get(k, 0)} {k}" for k in ("categorical", "numeric", "string")
        )
    )
    print(
        f"to fill by the curator: {len(data.columns)} labels, {len(data.columns)} universes, {codes} value labels; questions and concepts where they apply"
    )
    print(
        f"candidate missing codes found in {len(with_missing)} columns: {', '.join(with_missing)}"
    )
    print(f"draft written to {args.draft}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
