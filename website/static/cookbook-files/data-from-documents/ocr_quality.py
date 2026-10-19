"""Score the quality of OCR text on a page and list the lines to review.

Reads the OCR output of one page (plain text) and reports the signs of
recognition errors that matter for data extraction:

    digit garbles   letters inside number tokens (1O,245; l,412; 98O)
    broken words    alphabetic tokens with no vowel, tokens that mix letters
                    and digits (Tab1e, popu1ation), or tokens with rare symbols
    table lines     lines with three or more number tokens, which are the
                    lines the extraction will read, with the count of
                    garbles in them

Prints a summary with the garble rate per number token and the lines to
review, and exits 0. A page whose rate exceeds the threshold is marked for
re-OCR with another engine or at a higher resolution. Standard library
only; no dictionary is used, so broken words are found by shape.

Usage:
    python ocr_quality.py ocr_sample_page.txt [--threshold 0.03]

What this does not do: it finds errors by shape, so a wrong digit that is
still a digit (3 read as 8) is invisible to it; the total checks of the
verification chapter catch those. A page in a language with few vowels in
its romanization needs a different broken-word rule.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NUMBER = re.compile(r"^[\d,.\-%OolISB]+$")
GARBLE = re.compile(r"[OolISB]")
WORD = re.compile(r"^[A-Za-z][A-Za-z\-']*$")
RARE = re.compile(r"[~|¬¦`^]")
MIXED = re.compile(r"^(?=.*[A-Za-z]{2,})(?=.*\d)[A-Za-z\d]+$")
VOWELS = re.compile(r"[aeiouyAEIOUY]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("page", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="garble rate above which re-OCR is recommended",
    )
    args = parser.parse_args(argv)
    lines = args.page.read_text(encoding="utf-8").splitlines()

    numbers = garbled = 0
    broken: list[str] = []
    review: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines, start=1):
        tokens = line.split()
        line_numbers = [
            t for t in tokens if NUMBER.match(t) and any(ch.isdigit() for ch in t)
        ]
        line_garbled = [t for t in line_numbers if GARBLE.search(t)]
        numbers += len(line_numbers)
        garbled += len(line_garbled)
        for t in tokens:
            if (
                WORD.match(t)
                and len(t) > 2
                and not VOWELS.search(t)
                or RARE.search(t)
                or (MIXED.match(t) and not NUMBER.match(t))
            ):
                broken.append(f"{i}: {t}")
        if len(line_numbers) >= 3 or line_garbled:
            review.append((i, len(line_garbled), line))

    rate = garbled / numbers if numbers else 0.0
    print(
        f"{len(lines)} lines, {numbers} number tokens, {garbled} garbled ({rate:.1%}), {len(broken)} broken words"
    )
    print("\nlines to review (garbles | line):")
    for i, g, line in review:
        print(f"  {i:>3} {g:>2} | {line[:90]}")
    if broken:
        print("\nbroken words: " + ", ".join(broken))
    verdict = "re-OCR recommended" if rate > args.threshold else "within threshold"
    print(
        f"\nverdict: {verdict} (garble rate {rate:.1%}, threshold {args.threshold:.0%})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
