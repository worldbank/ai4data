"""Check translated metadata fields before a native speaker reviews them.

Reads a translations file (CSV: record_id, field, language, source_text,
translated_text, status machine|reviewed, reviewer) and flags, per row,
the problems a reviewer should look at first:

    numbers changed   digits, years, or thresholds in the source are missing
                      or different in the translation (1000 -> 100)
    untranslated      the translation equals the source
    length            the translation is shorter than half or longer than
                      twice the source, which usually means dropped or
                      added content
    unreviewed        status is machine, so the text may not be published

Prints the flags and the counts, and exits 1 when any machine translation
has a numbers or length flag, so that a pipeline holds those back.
Uses pandas.

Usage:
    python check_translations.py translations_example.csv

What this does not do: it checks form. A fluent translation that uses the
wrong statistical term ("chômage" for a concept that the organization
calls differently) passes every check here and is caught by the native
speaker with the vocabulary of the standardization chapter.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

import pandas as pd

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> Counter:
    return Counter(n.replace(",", ".") for n in NUMBER.findall(text))


def flags_for(r: pd.Series) -> list[str]:
    """The checks on one translation: numbers kept, text changed, length plausible, reviewed."""
    src, dst = r["source_text"], r["translated_text"]
    flags = []
    if numbers(src) != numbers(dst):
        missing, added = numbers(src) - numbers(dst), numbers(dst) - numbers(src)
        detail = "; ".join(
            p
            for p in (
                f"source {', '.join(missing)}" if missing else "",
                f"translation {', '.join(added)}" if added else "",
            )
            if p
        )
        flags.append(f"numbers changed ({detail})")
    if src.strip() == dst.strip():
        flags.append("untranslated")
    ratio = len(dst) / len(src) if src else 1.0
    if not 0.5 <= ratio <= 2.0:
        flags.append(f"length ratio {ratio:.2f}")
    if r["status"] != "reviewed":
        flags.append("unreviewed")
    return flags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("translations")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.translations, dtype=str).fillna("")
    df["flags"] = df.apply(flags_for, axis=1)
    df["held"] = df.apply(
        lambda r: (
            r["status"] != "reviewed"
            and any(f.startswith(("numbers", "length")) for f in r["flags"])
        ),
        axis=1,
    )

    print(f"{'record':<22} {'field':<17} {'lang':<4} {'status':<9} flags")
    for r in df.itertuples():
        print(
            f"{r.record_id:<22} {r.field:<17} {r.language:<4} {r.status:<9} {'; '.join(r.flags) if r.flags else 'ok'}"
        )
    counts = Counter(
        f.split(" (")[0].split(" ratio")[0] for flags in df["flags"] for f in flags
    )
    print(
        f"\n{len(df)} translations: "
        + ", ".join(f"{k} {v}" for k, v in counts.most_common())
    )
    print(
        f"{int(df['held'].sum())} machine translation(s) held back for numbers or length; the rest go to the native speaker"
    )
    return 1 if df["held"].any() else 0


if __name__ == "__main__":
    sys.exit(main())
