"""Score mention extraction against a labelled sample.

Reads the extractor's mentions (JSON lines) and a labelled sample (CSV:
document_id, sentence, mention_text, with mention_text empty when the
sentence contains no dataset mention) and reports precision, recall, and
F1 at the level of (document, mention text), with the false positives and
false negatives listed. Mentions are compared after normalization (lower
case, punctuation removed). Uses pandas.

Usage:
    python score_mentions.py mentions_example.jsonl labelled_sample.csv

What this does not do: fourteen labelled sentences give a demonstration.
A measure needs a sample of a few hundred sentences drawn across document
types, with two annotators and an agreement check, and the comparison
should also score the usage flag and the context, which this script leaves
out.
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

WORD = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mentions")
    parser.add_argument("labelled")
    args = parser.parse_args(argv)
    mentions = pd.read_json(args.mentions, lines=True)
    labelled = pd.read_csv(args.labelled, dtype=str).fillna("")

    found = set(zip(mentions["document_id"], mentions["text"].map(norm)))
    expected = {
        (d, norm(t))
        for d, t in zip(labelled["document_id"], labelled["mention_text"])
        if t.strip()
    }
    tp, fp, fn = found & expected, found - expected, expected - found
    precision = len(tp) / len(found) if found else 0.0
    recall = len(tp) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(
        f"labelled mentions {len(expected)}, extracted {len(found)}, correct {len(tp)}"
    )
    print(f"precision {precision:.2f}  recall {recall:.2f}  F1 {f1:.2f}")
    for title, pairs in (
        ("false positives (extracted, not labelled as a mention):", fp),
        ("false negatives (labelled, not extracted):", fn),
    ):
        if pairs:
            print(title)
            for d, t in sorted(pairs):
                print(f"  {d}: {t!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
