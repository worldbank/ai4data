"""Compare the data dictionaries of two survey rounds and draft a crosswalk.

Reads the dictionary of an earlier round and of a later round (the CSV
form used in this cookbook: name, label, type, values, question, ...) and
classifies every variable:

    same        same name, same value codes
    recoded     same name, different value codes: a break in the series
                until a correspondence is documented
    retyped     same name, different type
    renamed     different name, same label or question text
    dropped     in the earlier round only
    new         in the later round only

Writes the crosswalk as CSV (old_name, new_name, status, note) and prints
the counts. A duplicated name in either dictionary is reported and the
first occurrence is used. Uses pandas.

Usage:
    python compare_rounds.py lfs_2024q4_dictionary.csv lfs_2025q2_dictionary.csv round_crosswalk.csv

What this does not do: it matches by name and by exact label or question
text. A variable renamed and relabelled at once is reported as dropped and
new, and a code list that kept its codes but changed their meaning is
reported as same; the methodologist's review of the crosswalk catches
both. A documented correspondence between code lists (an XKOS
correspondence table) is the full version of the "recoded" note.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

STATUSES = ("same", "recoded", "retyped", "renamed", "dropped", "new")


def load(path: Path) -> pd.DataFrame:
    """Read a dictionary, keeping the first row of a duplicated name and warning about the rest."""
    df = pd.read_csv(path, dtype=str).fillna("")
    for name, n in df["name"].value_counts().items():
        if n > 1:
            print(
                f"warning: {path.name} lists {name!r} {n} times; first occurrence used"
            )
    return df.drop_duplicates("name").set_index("name")


def compare(
    old: pd.DataFrame, new: pd.DataFrame, earlier: str, later: str
) -> pd.DataFrame:
    """Classify every variable of the two rounds."""
    rows = []
    matched: set[str] = set()
    for name, o in old.iterrows():
        if name in new.index:
            n = new.loc[name]
            matched.add(name)
            if o["type"] != n["type"]:
                rows.append((name, name, "retyped", f"{o['type']} -> {n['type']}"))
            elif o["values"] != n["values"]:
                note = (
                    f"codes missing in {later}"
                    if not n["values"]
                    else f"codes missing in {earlier}"
                    if not o["values"]
                    else f"codes changed: {o['values']} -> {n['values']}"
                )
                rows.append((name, name, "recoded", note))
            else:
                rows.append((name, name, "same", ""))
            continue
        unmatched = new.loc[[k for k in new.index if k not in matched]]
        by_label = unmatched[(unmatched["label"] == o["label"]) & (o["label"] != "")]
        by_question = unmatched[
            (unmatched["question"] == o["question"]) & (o["question"] != "")
        ]
        if not by_label.empty:
            k = by_label.index[0]
            matched.add(k)
            rows.append((name, k, "renamed", "matched on label"))
        elif not by_question.empty:
            k = by_question.index[0]
            matched.add(k)
            rows.append((name, k, "renamed", "matched on question"))
        else:
            rows.append((name, "", "dropped", ""))
    rows += [("", name, "new", "") for name in new.index if name not in matched]
    return pd.DataFrame(rows, columns=["old_name", "new_name", "status", "note"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("earlier", type=Path)
    parser.add_argument("later", type=Path)
    parser.add_argument("crosswalk", type=Path)
    args = parser.parse_args(argv)
    old, new = load(args.earlier), load(args.later)
    crosswalk = compare(old, new, args.earlier.name, args.later.name)
    crosswalk.to_csv(args.crosswalk, index=False)

    print(
        f"{len(old)} variables in {args.earlier.name}, {len(new)} in {args.later.name}"
    )
    for status in STATUSES:
        part = crosswalk[crosswalk["status"] == status]
        if not part.empty:
            names = ", ".join(
                part["old_name"].where(part["old_name"] != "", part["new_name"])
            )
            print(f"{status:<8} {len(part):>3}  {names}")
    print(f"crosswalk written to {args.crosswalk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
