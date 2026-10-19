"""Write each cookbook's glossary page from the shared glossary.

Reads cookbook/_shared/glossary.md, where every term is defined once as
`**Term.** definition`, optionally followed by `<!-- match: word; word -->`
naming the words that count as a use of the term. For each cookbook it
keeps the terms whose name or match words appear in the cookbook's
chapters, and writes cookbook/<id>/glossary.md with the page's existing
front matter. With --check it reports the pages that would change and
changes nothing, which is how CI confirms the pages are generated.

Usage:
    python scripts/docs/build_glossaries.py [--check]

What this does not do: it does not judge whether a term is used in its
technical sense; a match word that is also an ordinary word (gate, token)
pulls the term into every cookbook that uses the word.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHARED = REPO / "cookbook" / "_shared" / "glossary.md"
SKIP_DIRS = {"_shared", "_template", "authoring"}


def shared_terms() -> list[tuple[str, str, list[str]]]:
    """(term, definition, match words) for every entry of the shared glossary."""
    body = SHARED.read_text(encoding="utf-8").split("# Shared glossary\n", 1)[1]
    out = []
    for m in re.finditer(
        r"^\*\*(.+?)\.\*\*\s*(.*?)(?:\s*<!-- match: (.*?) -->)?\s*(?=\n\*\*|\Z)",
        body,
        re.DOTALL | re.MULTILINE,
    ):
        term, definition, match = m.group(1), " ".join(m.group(2).split()), m.group(3)
        words = [w.strip().lower() for w in match.split(";")] if match else []
        base = term.lower().split(" (")[0]
        words = sorted({base, *base.split(" and "), *words})
        out.append((term, definition, words))
    return out


def front_matter(path: Path, cookbook: str) -> str:
    if path.exists():
        m = re.match(r"^---\n.*?\n---\n", path.read_text(encoding="utf-8"), re.DOTALL)
        if m:
            return m.group(0)
    return f"---\nid: glossary\ntitle: Glossary\nsidebar_position: 91\ndescription: Terms used in the {cookbook} cookbook, in plain language.\n---\n"


def page_for(cookbook_dir: Path, terms: list[tuple[str, str, list[str]]]) -> str:
    text = "".join(
        p.read_text(encoding="utf-8") for p in cookbook_dir.glob("*.mdx")
    ).lower()
    kept = [(t, d) for t, d, words in terms if any(w in text for w in words)]
    body = "\n\n".join(f"**{t}.** {d}" for t, d in kept)
    return (
        front_matter(cookbook_dir / "glossary.md", cookbook_dir.name)
        + "\n# Glossary\n\n"
        + body
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report pages that would change; write nothing",
    )
    args = parser.parse_args(argv)
    terms = shared_terms()
    changed = 0
    for cb in sorted(
        p
        for p in (REPO / "cookbook").iterdir()
        if p.is_dir() and p.name not in SKIP_DIRS
    ):
        target = cb / "glossary.md"
        new = page_for(cb, terms)
        old = target.read_text(encoding="utf-8") if target.exists() else ""
        if new != old:
            changed += 1
            print(
                f"{'would change' if args.check else 'written'}: {target.relative_to(REPO)} ({new.count(chr(10) + '**')} terms)"
            )
            if not args.check:
                target.write_text(new, encoding="utf-8")
    print(
        f"{len(terms)} shared terms; {changed} page(s) {'differ' if args.check else 'written'}"
    )
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
