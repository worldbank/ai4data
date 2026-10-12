"""Check the licence matrix of published products, and the licences of models.

Reads the licence matrix (one row per published product) and reports a
product without a licence, a licence identifier outside the allowed list
(SPDX identifiers plus "custom"), a licence without a URL, an AI-use
field outside allowed / conditions / prohibited, a row whose AI use is
"conditions" or "prohibited" without stated terms, and a product whose
licence is not machine-readable. With --models it reads the model
licence table and reports a model without a licence class from
permissive / restricted / proprietary-api or without an allowed-use
statement. Exit status 1 when any product or model has a problem.

Usage:
    python check_licences.py licence_matrix.csv --models model_licences.csv

Requires pandas.

What this does not check: whether a custom licence says what the row
claims. The legal adviser reads the text; the script checks that every
product and model has a reviewed row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ALLOWED_LICENCES = {
    "CC-BY-4.0",
    "CC0-1.0",
    "CC-BY-SA-4.0",
    "ODC-By-1.0",
    "ODbL-1.0",
    "custom",
}
AI_USE = {"allowed", "conditions", "prohibited"}
MODEL_CLASSES = {"permissive", "restricted", "proprietary-api"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--models", type=Path, help="model licence table")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.matrix, dtype=str).fillna("")
    problems: list[str] = []
    for _, row in df.iterrows():
        pid = row["product_id"]
        if not row["licence"]:
            problems.append(f"{pid}: no licence")
        elif row["licence"] not in ALLOWED_LICENCES:
            problems.append(
                f"{pid}: licence {row['licence']!r} is not on the allowed list"
            )
        if row["licence"] and not row["licence_url"]:
            problems.append(f"{pid}: licence has no URL")
        if row["ai_use"] not in AI_USE:
            problems.append(
                f"{pid}: AI use {row['ai_use']!r} is not allowed, conditions, or prohibited"
            )
        elif row["ai_use"] in {"conditions", "prohibited"} and not row["ai_use_terms"]:
            problems.append(
                f"{pid}: AI use is {row['ai_use']} and the terms are not stated"
            )
        if row["machine_readable"].lower() != "yes":
            problems.append(f"{pid}: licence is not machine-readable")

    by_licence = df["licence"].replace("", "(none)").value_counts().to_dict()
    by_ai = df["ai_use"].replace("", "(none)").value_counts().to_dict()
    print(
        f"{len(df)} products; licences: "
        + ", ".join(f"{n} {k}" for k, n in sorted(by_licence.items()))
    )
    print("AI use: " + ", ".join(f"{n} {k}" for k, n in sorted(by_ai.items())))
    print(
        f"machine-readable: {(df['machine_readable'].str.lower() == 'yes').sum()}/{len(df)}"
    )

    if args.models:
        models = pd.read_csv(args.models, dtype=str).fillna("")
        for _, row in models.iterrows():
            mid = row["model_id"]
            if row["licence_class"] not in MODEL_CLASSES:
                problems.append(
                    f"{mid}: licence class {row['licence_class']!r} is not permissive, restricted, or proprietary-api"
                )
            if not row["allowed_use"]:
                problems.append(f"{mid}: allowed use not stated")
            if row["licence_class"] == "restricted" and not row["restrictions_noted"]:
                problems.append(
                    f"{mid}: restricted licence without the restrictions noted"
                )
        classes = models["licence_class"].value_counts().to_dict()
        print(
            f"{len(models)} models; classes: "
            + ", ".join(f"{n} {k}" for k, n in sorted(classes.items()))
        )

    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
