"""Screen a text before it is entered into an assistant.

Reads a text file and reports the markers that decide which tier the
text belongs to: respondent-level records and identifiers (national
identity numbers, household and respondent references, phone numbers,
email addresses) make it confidential, which only a local assistant may
see; embargo, unpublished, provisional, and draft markers make it
internal, which no external service may see; a text with none of these
is public. Exit status 1 when the text is confidential, so that a script
or a habit stops before the paste.

Usage:
    python screen_input.py draft_note.txt

Standard library only.

What this does not check: meaning. A text that describes a person
without any of these markers is still confidential, and the person who
pastes it is the one who knows that.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CONFIDENTIAL = {
    "national identity number": re.compile(r"\b\d{4}-\d{2}-\d{5}\b"),
    "household or respondent reference": re.compile(
        r"\b(household|respondent|enumerator)\s+\d+", re.IGNORECASE
    ),
    "phone number": re.compile(r"\+?\d[\d ]{8,}\d"),
    "email address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "person named with a record": re.compile(r"\b(name|surname)\s*:", re.IGNORECASE),
}
INTERNAL = {
    "embargo": re.compile(r"\bembargo(ed)?\b", re.IGNORECASE),
    "unpublished or provisional": re.compile(
        r"\b(unpublished|provisional|not (yet )?released|before the release)\b",
        re.IGNORECASE,
    ),
    "draft": re.compile(r"\bdraft\b", re.IGNORECASE),
    "internal or restricted": re.compile(
        r"\b(internal only|restricted|do not circulate|must not leave)\b", re.IGNORECASE
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("text", type=Path)
    args = parser.parse_args(argv)
    text = args.text.read_text(encoding="utf-8")
    lines = text.splitlines()

    def report(name: str, pattern: re.Pattern) -> int:
        hits = [
            (i + 1, m.group(0))
            for i, line in enumerate(lines)
            for m in pattern.finditer(line)
        ]
        for n, hit in hits:
            print(f"  line {n}: {name}: {hit}")
        return len(hits)

    print("confidential markers:")
    c = sum(report(name, p) for name, p in CONFIDENTIAL.items())
    if not c:
        print("  none")
    print("internal markers:")
    i = sum(report(name, p) for name, p in INTERNAL.items())
    if not i:
        print("  none")
    tier = "confidential" if c else "internal" if i else "public"
    advice = {
        "confidential": "do not enter this text into any assistant outside the organization; use the local assistant, or remove the records and identifiers first",
        "internal": "do not enter this text into an external service; the organization's own assistant may be used",
        "public": "any assistant the organization allows may be used",
    }[tier]
    print(f"tier: {tier}")
    print(f"advice: {advice}")
    return 1 if tier == "confidential" else 0


if __name__ == "__main__":
    sys.exit(main())
