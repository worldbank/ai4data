"""Map free-text keywords to a controlled vocabulary.

Reads keywords as curators or models wrote them (CSV: record_id, keyword)
and a vocabulary (CSV: preferred_label, uri, alternates separated by ";")
and assigns each keyword a preferred label by exact match on the preferred
label or an alternate, or by similarity above a threshold (difflib ratio),
or marks it unmapped. Prints the mapping and the unmapped keywords, which
are the candidates for new alternates or new concepts. Uses pandas and rapidfuzz (fuzzy matching).

The vocabulary format is the minimum of a SKOS concept scheme: a preferred
label, a URI, and alternate labels. A full scheme adds definitions,
broader and narrower relations, and labels per language.

Usage:
    python map_vocabulary.py keywords_freetext.csv vocabulary.csv

What this does not do: string matching maps known forms. A keyword that
names a concept in different words ("hunger" for undernourishment) is
unmapped until an alternate is added or a semantic match (embeddings) is
used, and a language model can propose the mapping for a curator to
confirm.
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd
from rapidfuzz import fuzz, process

WORD = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def build_index(vocabulary: pd.DataFrame) -> dict[str, str]:
    """Every normalized preferred label and alternate, mapped to its preferred label."""
    index = {}
    for c in vocabulary.itertuples():
        index[norm(c.preferred_label)] = c.preferred_label
        for alt in c.alternates.split(";"):
            if alt.strip():
                index[norm(alt)] = c.preferred_label
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("keywords")
    parser.add_argument("vocabulary")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args(argv)
    index = build_index(pd.read_csv(args.vocabulary, dtype=str).fillna(""))
    keywords = pd.read_csv(args.keywords, dtype=str).fillna("")

    counts = dict.fromkeys(("exact", "fuzzy", "none"), 0)
    unmapped: list[str] = []
    print(
        f"{'match':<6} {'score':>5}  {'record':<22} {'keyword':<34} -> preferred label"
    )
    for k in keywords.itertuples():
        text = norm(k.keyword)
        if text in index:
            kind, score, label = "exact", 1.0, index[text]
        elif hit := process.extractOne(
            text, list(index), scorer=fuzz.ratio, score_cutoff=args.threshold * 100
        ):
            kind, score, label = "fuzzy", hit[1] / 100, index[hit[0]]
        else:
            kind, score, label = "none", 0.0, "-"
            unmapped.append(f"{k.record_id}: {k.keyword!r}")
        counts[kind] += 1
        print(
            f"{kind:<6} {score:>5.2f}  {k.record_id:<22} {k.keyword!r:<34} -> {label}"
        )
    print(
        f"\n{len(keywords)} keywords: "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
    )
    if unmapped:
        print(
            "unmapped (add an alternate, add a concept, or let a model propose a mapping for review):"
        )
        for u in unmapped:
            print(f"  {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
