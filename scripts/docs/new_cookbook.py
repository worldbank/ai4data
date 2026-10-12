"""Create a cookbook from the template and register it on the site.

Copies cookbook/_template/ into cookbook/<id>/, writes one chapter file per
"slug:Title:Question" argument, creates the folder for downloadable files,
adds a sidebar to website/sidebars-cookbook.js, and adds a card to
website/src/content/cookbooks.js.

Usage:
    python scripts/docs/new_cookbook.py \\
        --id microdata-documentation \\
        --title "Practical Guide to AI-Ready Microdata Documentation" \\
        --audience "Data curators in national statistical organizations" \\
        --chapters "variables:Variables and value labels:Can AI interpret the variables?" \\
                   "questionnaire:Questionnaire and concepts:Can AI relate questions to concepts?"

Then edit the generated files; the placeholders left in them are listed at
the end. Run scripts/docs/check_cookbooks.py before opening a pull request.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COOKBOOK = REPO / "cookbook"
TEMPLATE = COOKBOOK / "_template"
SIDEBARS = REPO / "website" / "sidebars-cookbook.js"
CARDS = REPO / "website" / "src" / "content" / "cookbooks.js"
STATIC = REPO / "website" / "static" / "cookbook-files"

SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def camel(slug: str) -> str:
    head, *rest = slug.split("-")
    return head + "".join(p.capitalize() for p in rest)


def js(text: str) -> str:
    """Escape a value for a single-quoted JavaScript string."""
    return text.replace("\\", "\\\\").replace("'", "\\'")


def fill(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def parse_chapter(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":", 2)
    if len(parts) != 3 or not SLUG.match(parts[0]):
        sys.exit(
            f'chapter must be "slug:Title:Question" with a lowercase slug: {spec!r}'
        )
    return parts[0], parts[1].strip(), parts[2].strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--id",
        required=True,
        help="folder and URL slug, for example microdata-documentation",
    )
    parser.add_argument(
        "--title",
        required=True,
        help='for example "Practical Guide to AI-Ready Microdata Documentation"',
    )
    parser.add_argument(
        "--audience",
        required=True,
        help='for example "National statistical organizations"',
    )
    parser.add_argument("--subtitle", help='default: "A cookbook for <audience>."')
    parser.add_argument(
        "--description",
        help="one sentence for the landing card and the overview front matter",
    )
    parser.add_argument(
        "--chapters", nargs="+", required=True, metavar="SLUG:TITLE:QUESTION"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing cookbook folder"
    )
    args = parser.parse_args(argv)

    if not SLUG.match(args.id):
        sys.exit("--id must be lowercase words separated by hyphens")
    dest = COOKBOOK / args.id
    if dest.exists() and not args.force:
        sys.exit(f"{dest} exists; pass --force to overwrite")
    chapters = [parse_chapter(c) for c in args.chapters]
    today = dt.datetime.now(tz=dt.timezone.utc).date()  # noqa: UP017 (Python 3.10 compatible)
    subtitle = (
        args.subtitle
        or f"A cookbook for {args.audience[0].lower() + args.audience[1:]}."
    )
    description = args.description or (
        f"{len(chapters)} questions {args.audience[0].lower() + args.audience[1:]} can ask, "
        "each answered with recipes, maturity levels, tests, and a checklist."
    )

    dest.mkdir(parents=True, exist_ok=True)
    common = {
        "ID": args.id,
        "TITLE": args.title,
        "SUBTITLE": subtitle,
        "DESCRIPTION": description,
        "DATE": today.strftime("%B %Y"),
        "YEAR": str(today.year),
        "INTRO": "{/* Introduction to write. */}",
    }

    rows = "\n".join(
        f"| [{i}. {title}](./{slug}.mdx) | {question} | |"
        for i, (slug, title, question) in enumerate(chapters, 1)
    )
    (dest / "index.mdx").write_text(
        fill((TEMPLATE / "index.mdx").read_text(), {**common, "CHAPTER_ROWS": rows})
    )

    chapter_tpl = (TEMPLATE / "chapter.mdx").read_text()
    for i, (slug, title, question) in enumerate(chapters, 1):
        values = {
            **common,
            "CHAPTER_SLUG": slug,
            "CHAPTER_N": str(i),
            "CHAPTER_TITLE": title,
            "CHAPTER_SHORT": title.split(":")[0].split(" and ")[0],
            "CHAPTER_POSITION": str(i),
            "CHAPTER_QUESTION": question,
            "CHAPTER_DESCRIPTION": f"{title}.",
        }
        (dest / f"{slug}.mdx").write_text(fill(chapter_tpl, values))

    for name in ("standards.md", "glossary.md", "contributing.md"):
        (dest / name).write_text(fill((TEMPLATE / name).read_text(), common))

    static = STATIC / args.id
    static.mkdir(parents=True, exist_ok=True)
    (static / ".gitkeep").touch()

    # Register the sidebar.
    key = camel(args.id)
    sidebars = SIDEBARS.read_text()
    if f"dirName: '{args.id}'" not in sidebars:
        sidebars = sidebars.replace(
            "};\n\nexport default sidebars;",
            f"  {key}: [{{type: 'autogenerated', dirName: '{args.id}'}}],\n}};\n\nexport default sidebars;",
        )
        SIDEBARS.write_text(sidebars)

    # Register the landing card.
    cards = CARDS.read_text()
    if f"id: '{args.id}'" not in cards:
        card = (
            "  {\n"
            f"    id: '{args.id}',\n"
            f"    audience: '{js(args.audience)}',\n"
            f"    title: '{js(args.title)}',\n"
            f"    description:\n      '{js(description)}',\n"
            f"    chapters: {len(chapters)},\n"
            f"    to: '/cookbook/{args.id}/',\n"
            "  },\n"
        )
        cards = cards.replace("\n];\n", "\n" + card + "];\n", 1)
        CARDS.write_text(cards)

    print(f"created cookbook/{args.id}/ with {len(chapters)} chapter(s)")
    print("files to edit:")
    for path in sorted(dest.iterdir()):
        placeholders = len(
            re.findall(r"\{\{[A-Z_]+\}\}|\{/\*.*?\*/\}", path.read_text(), re.DOTALL)
        )
        print(f"  {path.relative_to(REPO)}  ({placeholders} placeholder(s))")
    print(f"  website/static/cookbook-files/{args.id}/  (downloadable files)")
    print(
        "registered in website/sidebars-cookbook.js and website/src/content/cookbooks.js"
    )
    print("next: python scripts/docs/check_cookbooks.py --id", args.id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
