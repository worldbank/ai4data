"""Score model suggestions against curator decisions.

Reads the suggestions a model or a review pipeline produced (JSON lines:
record_id, field, source, current, suggested, issue_category,
issue_severity, model_confidence) and the decisions curators recorded on
them (CSV: record_id, field, decision accept|edit|reject, final_text,
reason, reviewer, date), and reports per field and per source:

    n          suggestions decided
    accepted   taken as proposed
    edited     taken after a change; the similarity between the suggestion
               and the final text shows how much was changed
    rejected   with the curators' reasons listed

The acceptance rate (accepted plus edited over decided) per field and per
source is the measure that chapter 6 tracks over time. Uses pandas and rapidfuzz (text similarity).

Usage:
    python score_suggestions.py suggestions_example.jsonl decisions_example.csv

What this does not do: ten suggestions give a demonstration; a measure
needs a period's decisions, a few hundred, with the reasons coded into an
error taxonomy. The script counts; it does not judge whether curators were
right.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from rapidfuzz import fuzz


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("suggestions")
    parser.add_argument("decisions")
    args = parser.parse_args(argv)
    suggestions = pd.read_json(args.suggestions, lines=True)
    decisions = pd.read_csv(args.decisions, dtype=str).fillna("")
    decisions["decision"] = decisions["decision"].str.strip().str.lower()
    joined = suggestions.merge(
        decisions, on=["record_id", "field"], how="left", suffixes=("", "_decision")
    )
    decided = joined[joined["decision"].notna()].copy()
    decided["source"] = decided.get(
        "source", pd.Series("?", index=decided.index)
    ).fillna("?")

    print(
        f"{len(suggestions)} suggestions, {len(decided)} decided, {len(joined) - len(decided)} undecided"
    )
    for by in ("field", "source"):
        table = pd.crosstab(decided[by], decided["decision"]).reindex(
            columns=["accept", "edit", "reject"], fill_value=0
        )
        table["n"] = table.sum(axis=1)
        table = table.astype({"accept": int, "edit": int, "reject": int, "n": int})
        table["rate"] = (table["accept"] + table["edit"]) / table["n"]
        print(
            f"\n{'by ' + by:<28} {'n':>3} {'accept':>7} {'edit':>5} {'reject':>7} {'rate':>6}"
        )
        for label, g in table.sort_values(
            "n", ascending=False, kind="stable"
        ).iterrows():
            print(
                f"{label:<28} {g['n']:>3} {g['accept']:>7} {g['edit']:>5} {g['reject']:>7} {g['rate']:>6.2f}"
            )
    edited = decided[decided["decision"] == "edit"]
    if not edited.empty:
        similarity = [
            fuzz.ratio(s, f) / 100
            for s, f in zip(edited["suggested"], edited["final_text"])
        ]
        print(
            f"\nedited suggestions: mean similarity to final text {sum(similarity) / len(similarity):.2f}"
        )
    rejected = decided[decided["decision"] == "reject"]
    if not rejected.empty:
        print("\nrejected, with reasons:")
        for r in rejected.itertuples():
            print(f"  {r.record_id}/{r.field} ({r.source}): {r.reason}")
    print("\nThe acceptance rate counts curator decisions; it does not judge them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
