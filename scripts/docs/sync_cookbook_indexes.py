"""Keep each cookbook's overview tables and landing card in step with its files.

For every cookbook it reads the chapter files (title and question from each
chapter's front matter and **Question:** line) and:

- rewrites the chapter and question columns of the "Chapters and questions"
  table in index.mdx, keeping the practical-topics column by chapter slug,
  and reports chapters missing from the table or rows without a chapter;
- checks the "Files" table in index.mdx against the files in
  website/static/cookbook-files/<id>/: every file has a row, every row names
  a file that exists;
- rewrites the chapter count of the cookbook's card in
  website/src/content/cookbooks.js;
- checks that the self-assessment's question ids are chapter slugs.

With --check it reports what would change and changes nothing; exit status
1 when anything differs or is missing.

Usage:
    python scripts/docs/sync_cookbook_indexes.py [--check] [--id COOKBOOK]

What this does not do: it cannot write the practical-topics column or the
"Used in" text of a new file; it reports them for a person to add.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COOKBOOK = REPO / "cookbook"
STATIC = REPO / "website" / "static" / "cookbook-files"
CARDS = REPO / "website" / "src" / "content" / "cookbooks.js"
SKIP = {"_template", "_shared", "authoring"}
ROW = re.compile(
    r"^\| \[(\d+)\. ([^\]]*)\]\(\./([^)]+)\.mdx\) \| ([^|]*) \| ([^|]*) \|$",
    re.MULTILINE,
)


def front_matter(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def chapters(folder: Path) -> list[tuple[int, str, str, str]]:
    """(number, slug, title, question) for every chapter, in order."""
    out = []
    for p in folder.glob("*.mdx"):
        text = p.read_text(encoding="utf-8")
        if "<Recipe" not in text:
            continue
        title = front_matter(text).get("title", "")
        m = re.match(r"(\d+)\.\s+(.*)", title)
        q = re.search(r"\*\*Question:\*\*\s*(.+)", text)
        if m:
            out.append(
                (
                    int(m.group(1)),
                    p.stem,
                    m.group(2).strip(),
                    q.group(1).strip() if q else "",
                )
            )
    return sorted(out)


def sync_chapter_table(
    index: Path, chs: list[tuple[int, str, str, str]], problems: list[str]
) -> str:
    text = index.read_text(encoding="utf-8")
    rows = {m.group(3): m for m in ROW.finditer(text)}
    for slug in rows:
        if slug not in {c[1] for c in chs}:
            problems.append(
                f"{index.relative_to(REPO)}: table row for '{slug}' has no chapter file"
            )
    for n, slug, title, question in chs:
        if slug not in rows:
            problems.append(
                f"{index.relative_to(REPO)}: chapter {n} ({slug}) has no row in the chapters table; add it with its practical topics"
            )
            continue
        m = rows[slug]
        topics = m.group(5).strip()
        new = f"| [{n}. {title}](./{slug}.mdx) | {question} | {topics} |"
        text = text.replace(m.group(0), new)
    return text


def check_files_table(index: Path, cid: str, problems: list[str]) -> None:
    text = index.read_text(encoding="utf-8")
    section = re.search(r"^## Files\n(.*?)(?=^## )", text, re.MULTILINE | re.DOTALL)
    listed = (
        set(
            re.findall(
                rf"pathname:///cookbook-files/{re.escape(cid)}/([^)\s]+)",
                section.group(1),
            )
        )
        if section
        else set()
    )
    on_disk = (
        {
            p.name
            for p in (STATIC / cid).iterdir()
            if p.is_file() and not p.name.startswith(".")
        }
        if (STATIC / cid).exists()
        else set()
    )
    for name in sorted(on_disk - listed):
        problems.append(
            f"{index.relative_to(REPO)}: {name} is in cookbook-files/{cid}/ and not in the Files table"
        )
    for name in sorted(listed - on_disk):
        problems.append(
            f"{index.relative_to(REPO)}: the Files table links {name}, which is not in cookbook-files/{cid}/"
        )


def sync_card(cid: str, count: int, problems: list[str]) -> str:
    text = CARDS.read_text(encoding="utf-8")
    m = re.search(rf"id: '{re.escape(cid)}',(.*?)chapters: (\d+),", text, re.DOTALL)
    if not m:
        problems.append(f"{CARDS.relative_to(REPO)}: no card for '{cid}'")
        return text
    if int(m.group(2)) != count:
        text = text[: m.start(2)] + str(count) + text[m.end(2) :]
    return text


def check_self_assessment(
    folder: Path, chs: list[tuple[int, str, str, str]], problems: list[str]
) -> None:
    p = folder / "start-here.mdx"
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8")
    ids = re.findall(r"^\s+id: '([^']+)',", text, re.MULTILINE)
    if not ids:
        # the first cookbook keeps its questions inside the component
        comp = re.search(
            r"import SelfAssessment from '@site/src/components/([^']+)'", text
        )
        if comp:
            cp = REPO / "website" / "src" / "components" / comp.group(1)
            cp = cp if cp.suffix else cp.with_suffix(".js")
            if cp.exists():
                ids = re.findall(
                    r"^\s+id: '([^']+)',", cp.read_text(encoding="utf-8"), re.MULTILINE
                )
    slugs = {c[1] for c in chs}
    for i in ids:
        if i not in slugs:
            problems.append(
                f"{p.relative_to(REPO)}: self-assessment id '{i}' is not a chapter slug"
            )
    for s in sorted(slugs - set(ids)):
        problems.append(
            f"{p.relative_to(REPO)}: chapter '{s}' has no self-assessment question"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--id")
    args = parser.parse_args(argv)
    problems: list[str] = []
    changed = 0
    cards = CARDS.read_text(encoding="utf-8")
    for folder in sorted(
        p for p in COOKBOOK.iterdir() if p.is_dir() and p.name not in SKIP
    ):
        if args.id and folder.name != args.id:
            continue
        chs = chapters(folder)
        index = folder / "index.mdx"
        new_index = sync_chapter_table(index, chs, problems)
        if new_index != index.read_text(encoding="utf-8"):
            changed += 1
            print(
                f"{'would update' if args.check else 'updated'} chapters table: {index.relative_to(REPO)}"
            )
            if not args.check:
                index.write_text(new_index, encoding="utf-8")
        check_files_table(index, folder.name, problems)
        check_self_assessment(folder, chs, problems)
        new_cards = sync_card(folder.name, len(chs), problems)
        if new_cards != cards:
            changed += 1
            print(
                f"{'would update' if args.check else 'updated'} chapter count: {folder.name} -> {len(chs)}"
            )
            cards = new_cards
    if not args.check and cards != CARDS.read_text(encoding="utf-8"):
        CARDS.write_text(cards, encoding="utf-8")
    for s in problems:
        print(f"missing  {s}")
    print(f"{changed} change(s), {len(problems)} missing item(s)")
    return 1 if (problems or (args.check and changed)) else 0


if __name__ == "__main__":
    sys.exit(main())
