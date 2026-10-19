"""Check a dataset card for the sections and statements a release needs.

Reads a dataset card (Markdown with a YAML front matter) and checks: the
front matter names a licence, a language, a task category, and a pretty
name; every required section is present; the out-of-scope section lists
at least one use; the representativeness section contains at least one
number; the citation section contains a DOI or a URL; and the versions
section names the current version. Exit status 1 when anything is
missing. Uses PyYAML for the header.

Usage:
    python check_dataset_card.py dataset_card.md

What this does not do: it checks presence. Whether the stated uses are
sensible, the known gaps complete, and the collection process described
truthfully are the authors' responsibility, reviewed by a person.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

FRONT_MATTER = ["license", "language", "task_categories", "pretty_name"]
REQUIRED = [
    "Dataset summary",
    "Motivation",
    "Composition",
    "Collection process",
    "Preprocessing",
    "Representativeness and known gaps",
    "Intended uses",
    "Out-of-scope uses",
    "Licence and terms",
    "Versions",
    "Citation",
    "Contact",
]


def split_card(text: str) -> tuple[dict, str]:
    """The YAML header as a dict and the Markdown body."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    return (yaml.safe_load(m.group(1)) or {}), m.group(2)


def section(body: str, name: str) -> str:
    m = re.search(
        rf"^##\s+{re.escape(name)}\s*$(.*?)(?=^##\s|\Z)",
        body,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return m.group(1) if m else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("card")
    args = parser.parse_args(argv)
    meta, body = split_card(Path(args.card).read_text(encoding="utf-8"))

    problems: list[str] = []
    if not meta:
        problems.append("no YAML front matter")
    for k in FRONT_MATTER:
        ok = bool(meta.get(k))
        print(f"  {'ok' if ok else 'MISSING':<8} front matter: {k}")
        if not ok:
            problems.append(f"front matter: {k}")
    for name in REQUIRED:
        ok = bool(section(body, name).strip())
        print(f"  {'ok' if ok else 'MISSING':<8} {name}")
        if not ok:
            problems.append(name)
    checks = [
        (
            "out-of-scope section lists a use",
            bool(
                re.search(
                    r"^\s*[-*]\s+\S", section(body, "Out-of-scope uses"), re.MULTILINE
                )
            ),
        ),
        (
            "representativeness section states a number",
            bool(re.search(r"\d", section(body, "Representativeness and known gaps"))),
        ),
        (
            "citation has a DOI or URL",
            bool(re.search(r"10\.\d{4,}/|https?://", section(body, "Citation"))),
        ),
        (
            "versions section names a version",
            bool(re.search(r"\b\d+\.\d+(\.\d+)?\b", section(body, "Versions"))),
        ),
    ]
    for label, ok in checks:
        print(f"  {'ok' if ok else 'MISSING':<8} {label}")
        if not ok:
            problems.append(label)
    print(
        f"result: {'PASS' if not problems else 'FAIL'}"
        + (f" ({len(problems)} problem(s))" if problems else "")
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
