"""Renumber the chapters of a cookbook after a chapter is added, removed, or moved.

Given a cookbook id and the chapter slugs in their new order, rewrites in
each chapter file the front matter title and sidebar label ("N. ..."), the
sidebar position, the H1, and the Recipe titles ("N.M ..."); and in every
page of the cookbook the links of the form ``[chapter N](./slug.mdx)``,
``[Chapter N](./slug.mdx)`` and ``[N. Title](./slug.mdx)``, and the
self-assessment titles (``id: 'slug'`` followed by ``title: 'N. ...'``).
Prints the remaining plain-text mentions of "chapter N" for manual review,
since those cannot be resolved to a slug.

Usage:
    python scripts/docs/renumber_cookbook.py --id microdata-documentation \
        --order find produce variables concepts navigate rounds fitness access
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def renumber_chapter(path: Path, n: int) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r'^(title: ")\d+\. ', rf"\g<1>{n}. ", text, count=1, flags=re.MULTILINE
    )
    text = re.sub(
        r'^(sidebar_label: ")\d+\. ', rf"\g<1>{n}. ", text, count=1, flags=re.MULTILINE
    )
    text = re.sub(
        r"^sidebar_position: \d+$",
        f"sidebar_position: {n + 1}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^# \d+\. ", f"# {n}. ", text, count=1, flags=re.MULTILINE)
    text = re.sub(r'(<Recipe title=")\d+\.(\d+ )', rf"\g<1>{n}.\2", text)
    path.write_text(text, encoding="utf-8")


def relink(path: Path, numbers: dict[str, int]) -> None:
    text = path.read_text(encoding="utf-8")

    def link(m: re.Match) -> str:
        slug = m.group(3)
        if slug not in numbers:
            return m.group(0)
        return f"[{m.group(1)} {numbers[slug]}](./{slug}.mdx)"

    text = re.sub(r"\[(chapter|Chapter) (\d+)\]\(\./([a-z0-9-]+)\.mdx\)", link, text)

    def titled(m: re.Match) -> str:
        slug = m.group(3)
        if slug not in numbers:
            return m.group(0)
        return f"[{numbers[slug]}. {m.group(2)}](./{slug}.mdx)"

    text = re.sub(r"\[(\d+)\. ([^\]]+)\]\(\./([a-z0-9-]+)\.mdx\)", titled, text)

    def assessment(m: re.Match) -> str:
        slug = m.group(1)
        if slug not in numbers:
            return m.group(0)
        return f"id: '{slug}',{m.group(2)}title: '{numbers[slug]}. "

    text = re.sub(r"id: '([a-z0-9-]+)',(\s*)title: '\d+\. ", assessment, text)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--id", required=True)
    parser.add_argument(
        "--order", nargs="+", required=True, help="chapter slugs in their new order"
    )
    args = parser.parse_args(argv)
    folder = ROOT / "cookbook" / args.id
    numbers = {slug: i for i, slug in enumerate(args.order, start=1)}
    for slug, n in numbers.items():
        path = folder / f"{slug}.mdx"
        if not path.exists():
            sys.exit(f"missing chapter file: {path}")
        renumber_chapter(path, n)
    for path in sorted(folder.glob("*.md*")):
        relink(path, numbers)
    for path in sorted(folder.glob("*.md*")):
        for i, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for m in re.finditer(r"[Cc]hapters? \d+", line):
                if "](./" in line[m.end() : m.end() + 5]:
                    continue
                print(f"review {path.relative_to(ROOT)}:{i}: {line.strip()[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
