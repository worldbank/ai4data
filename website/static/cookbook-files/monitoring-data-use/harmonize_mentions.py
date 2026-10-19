"""Match extracted dataset mentions to the organization's canonical names.

Reads mentions (JSON lines, one per mention, as produced by the extraction
step) and a canonical names table (CSV: dataset_id, canonical_name,
acronym, producer, variants separated by ";"), and assigns each mention a
dataset_id with a match type and a score:

    exact     the normalized mention equals the canonical name, the acronym,
              or a variant
    contains  a variant or the acronym appears as a whole phrase inside the
              mention ("unemployment rate from the LFS")
    fuzzy     the normalized strings are similar above a threshold
              (difflib ratio), for spelling and word-order differences
    none      no match; the mention refers to another organization's data
              or is a new variant to review

Writes the mentions with the match fields added (JSON lines) and prints a
summary. Uses pandas and rapidfuzz (fuzzy matching).

What this does not do: it matches strings. The program's harmonization
step adds semantic matching with sentence embeddings and a review of
clusters; this script is the baseline that shows what string matching
alone resolves, and what it leaves for review.

Usage:
    python harmonize_mentions.py mentions_example.jsonl canonical_names.csv -o harmonized.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

WORD = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def load_canonical(path: Path) -> list[dict]:
    """The names table with normalized names and acronym per dataset."""
    rows = pd.read_csv(path, dtype=str).fillna("").to_dict("records")
    for r in rows:
        names = [r["canonical_name"], *r["variants"].split(";")]
        r["_names"] = [norm(n) for n in names if n.strip()]
        r["_acronym"] = norm(r["acronym"])
    return rows


def match(
    mention: str, canonical: list[dict], threshold: float
) -> tuple[str | None, str, float]:
    """Return (dataset_id, match type, score): exact, contains, fuzzy, or none."""
    m = norm(mention)
    for r in canonical:  # exact: the mention is a known name, variant, or acronym
        if m in r["_names"] or (r["_acronym"] and m == r["_acronym"]):
            return r["dataset_id"], "exact", 1.0
    best: tuple[str | None, str, float] = (None, "none", 0.0)
    for r in (
        canonical
    ):  # contains: a long name, or the acronym as a whole word, inside the mention
        for name in r["_names"]:
            if (
                len(name) >= 8
                and f" {name} " in f" {m} "
                and len(name) / len(m) > best[2]
            ):
                best = (r["dataset_id"], "contains", round(len(name) / len(m), 2))
        if r["_acronym"] and re.search(rf"\b{re.escape(r['_acronym'])}\b", m):
            score = min(len(r["_acronym"]) / len(m) + 0.3, 0.99)
            if score > best[2]:
                best = (r["dataset_id"], "contains", round(score, 2))
    if best[0]:
        return best
    choices = {
        name: r["dataset_id"] for r in canonical for name in r["_names"]
    }  # fuzzy: closest name by edit similarity
    hit = process.extractOne(
        m, list(choices), scorer=fuzz.ratio, score_cutoff=threshold * 100
    )
    if hit:
        name, score, _ = hit
        return choices[name], "fuzzy", round(score / 100, 2)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mentions", type=Path, help="JSON lines, one mention per line")
    parser.add_argument("canonical", type=Path, help="canonical names CSV")
    parser.add_argument(
        "-o", "--output", type=Path, help="JSON lines with match fields added"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="fuzzy match ratio (default 0.8)"
    )
    args = parser.parse_args(argv)
    canonical = load_canonical(args.canonical)
    mentions = pd.read_json(args.mentions, lines=True).to_dict("records")

    counts = dict.fromkeys(("exact", "contains", "fuzzy", "none"), 0)
    unmatched: list[str] = []
    for mention in mentions:
        dataset_id, kind, score = match(mention["text"], canonical, args.threshold)
        mention.update(
            {"dataset_id": dataset_id, "match_type": kind, "match_score": score}
        )
        counts[kind] += 1
        if kind == "none":
            unmatched.append(f"{mention['document_id']}: {mention['text']!r}")
        print(
            f"{kind:<9} {score:>5.2f}  {mention['document_id']}  {mention['text']!r:<58} -> {dataset_id or '-'}"
        )

    print(
        f"\n{len(mentions)} mentions: "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
    )
    if unmatched:
        print("unmatched (other organizations' data, or variants to add to the table):")
        for u in unmatched:
            print(f"  {u}")
    if args.output:
        args.output.write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in mentions),
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
