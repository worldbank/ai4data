"""Convert a data dictionary CSV into the World Bank microdata schema.

Produces the `data_files` and `variables` sections of a microdata record
(worldbank/metadata-schemas, microdata-schema.json, which follows DDI
Codebook), and merges them into an existing study record when one is given.
The result can be imported into NADA or the Metadata Editor, or validated
with the metadata checker from the dissemination cookbook.

Mapping from the CSV columns to DDI fields:

    name      -> name            values    -> var_catgry [{value, label}]
    label     -> labl            missing   -> var_catgry entries flagged in var_notes
    type      -> var_format.type question  -> var_qstn_qstnlit
    universe  -> var_universe    concept   -> var_concept [{title}]

Uses pandas.

What this does not do: it does not compute summary statistics (var_sumstat)
or category frequencies, which need the data file, and it does not validate
the result against the schema; run check_metadata.py from the dissemination
cookbook for that.

Usage:
    python dictionary_to_ddi.py lfs_2025q2_dictionary.csv --study example_microdata.json -o study_with_variables.json
    python dictionary_to_ddi.py lfs_2025q2_dictionary.csv            # prints the sections only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

FORMAT = {"numeric": "numeric", "categorical": "numeric", "string": "character"}


def parse_values(text: str) -> list[dict[str, str]]:
    """'1=Yes;2=No' -> [{value: '1', label: 'Yes'}, ...]."""
    pairs = (item.split("=", 1) for item in text.split(";") if "=" in item)
    return [{"value": v.strip(), "label": label.strip()} for v, label in pairs]


def variable(i: int, r: pd.Series) -> dict:
    """One DDI variable from one dictionary row."""
    var: dict = {
        "vid": f"V{i}",
        "file_id": r["file_id"],
        "name": r["name"],
        "labl": r["label"],
        "var_format": {"type": FORMAT.get(r["type"].lower(), "character")},
    }
    if r["universe"]:
        var["var_universe"] = r["universe"]
    if r["question"]:
        var["var_qstn_qstnlit"] = r["question"]
    missing = [m.strip() for m in r["missing"].split(";") if m.strip()]
    categories = parse_values(r["values"]) + [
        {"value": code, "label": "Missing"} for code in missing
    ]
    if categories:
        var["var_catgry"] = categories
    if missing:
        var["var_notes"] = "Missing-value codes: " + ", ".join(missing)
    if r["concept"]:
        var["var_concept"] = [{"title": r["concept"]}]
    return var


def convert(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    df = pd.DataFrame(rows).fillna("")
    for col in (
        "label",
        "type",
        "universe",
        "question",
        "values",
        "missing",
        "concept",
    ):
        if col not in df:
            df[col] = ""
    df = df.apply(lambda s: s.astype(str).str.strip())
    counts = df["file_id"].value_counts(sort=False)
    files = [
        {"file_id": f, "file_name": f"{f}.csv", "var_count": int(n)}
        for f, n in counts.items()
    ]
    variables = [variable(i, r) for i, (_, r) in enumerate(df.iterrows(), 1)]
    return files, variables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dictionary", type=Path)
    parser.add_argument(
        "--study", type=Path, help="study record (JSON) to merge the sections into"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="where to write the result (default: stdout)"
    )
    args = parser.parse_args(argv)

    rows = pd.read_csv(args.dictionary, dtype=str).fillna("").to_dict("records")
    files, variables = convert(rows)
    record = json.loads(args.study.read_text(encoding="utf-8")) if args.study else {}
    record.update({"data_files": files, "variables": variables})

    text = json.dumps(record, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(
            f"wrote {args.output}: {len(files)} file(s), {len(variables)} variable(s)"
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
