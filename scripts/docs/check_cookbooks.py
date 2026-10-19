"""Check that each cookbook follows the authoring standard.

For every folder under cookbook/ (except _template and authoring) the
checker verifies:

errors (exit status 1)
  - index.mdx exists with title and description, and has the required
    sections (Prerequisites, First steps, Chapters and questions,
    Maturity levels, Files, Citation, Version history)
  - each chapter (a file that uses the Recipe component) has front matter
    (title "N. ...", sidebar_position, description), a **Question:** line,
    the required sections (Rationale, Target state, Maturity levels,
    Recipes, Common mistakes, Checklist), and a Checklist component
  - each Recipe has a title "N.M ...", a level from the shared vocabulary,
    skills, time, and a **Result:** line
  - headings and titles do not use the banned constructions (", not ",
    "rather than", a statement plus a comma and a tag)
  - the cookbook is registered in website/sidebars-cookbook.js and
    website/src/content/cookbooks.js
  - a visible cookbook does not link into a cookbook listed as hidden in
    website/cookbook-visibility.json, and every hidden id names a folder
  - template placeholders ({{...}}), guidance comments, and sample text
    are gone

warnings
  - standards.md, glossary.md, or contributing.md missing
  - a recipe without a "What this does not do" note
  - second person (you, your) in prose outside code blocks
  - Implementation options or Verification section missing
  - no folder under website/static/cookbook-files/<id>/

Usage:
    python scripts/docs/check_cookbooks.py
    python scripts/docs/check_cookbooks.py --id ai-ready-dissemination
    python scripts/docs/check_cookbooks.py --strict   # warnings fail too
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COOKBOOK = REPO / "cookbook"
SIDEBARS = REPO / "website" / "sidebars-cookbook.js"
CARDS = REPO / "website" / "src" / "content" / "cookbooks.js"
STATIC = REPO / "website" / "static" / "cookbook-files"
SKIP = {"_template", "_shared", "authoring"}
VISIBILITY = REPO / "website" / "cookbook-visibility.json"


def hidden_cookbooks() -> set[str]:
    """Ids listed as hidden in website/cookbook-visibility.json."""
    if not VISIBILITY.exists():
        return set()
    return set(json.loads(VISIBILITY.read_text(encoding="utf-8")).get("hidden", []))


def indexed_cookbooks() -> set[str]:
    """Ids listed as indexed (cleared review) in website/cookbook-visibility.json."""
    if not VISIBILITY.exists():
        return set()
    return set(json.loads(VISIBILITY.read_text(encoding="utf-8")).get("indexed", []))


VERSION_LINE = re.compile(r"Version (\S+) \(working draft\), ([A-Z][a-z]+ \d{4})\.")
CARD_VERSION = re.compile(r"id: '([a-z0-9-]+)',\s*version: '([^']+)',\s*date: '([^']+)',")


def check_versions(folders: list[Path], rep: Report) -> None:
    """The version and date on each card match the cookbook's index page.

    The draft banner on every chapter shows the card's version and date, so
    the two have to agree with the "Version X (working draft), Month YYYY."
    line of index.mdx.
    """
    cards = {m.group(1): (m.group(2), m.group(3)) for m in CARD_VERSION.finditer(CARDS.read_text(encoding="utf-8"))}
    for folder in folders:
        index = folder / "index.mdx"
        if not index.exists():
            continue
        m = VERSION_LINE.search(index.read_text(encoding="utf-8"))
        if not m:
            rep.error(index, "no 'Version X (working draft), Month YYYY.' line")
            continue
        card = cards.get(folder.name)
        if card is None:
            rep.error(CARDS, f"card for '{folder.name}' has no version and date")
        elif card != (m.group(1), m.group(2)):
            rep.error(
                CARDS,
                f"card for '{folder.name}' says version {card[0]}, {card[1]}; "
                f"index.mdx says version {m.group(1)}, {m.group(2)}",
            )


LEVELS = {"Foundational", "AI-ready", "AI-native"}
INDEX_SECTIONS = [
    "Prerequisites",
    "First steps",
    "Chapters and questions",
    "Maturity levels",
    "Files",
    "Citation",
    "Version history",
]
CHAPTER_REQUIRED = [
    "Rationale",
    "Target state",
    "Maturity levels",
    "Recipes",
    "Common mistakes",
    "Checklist",
]
CHAPTER_RECOMMENDED = ["Implementation options", "Resources", "Verification"]

BANNED = re.compile(
    r", not |rather than|instead of|not just|isn.t\.|\bmerely\b|\bsimply\b|\bactually\b"
)
COMMA_TAG = re.compile(r", (and|with|but|so|then|because) ")
CLAUSE_TAG = re.compile(
    r"\b(and|with) (what|who|how|where|why|when|whether)\b"
)  # "Five units and what each runs"
SECOND_PERSON = re.compile(r"\b(you|your|yours)\b", re.IGNORECASE)
FRONT = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
RECIPE = re.compile(r"<Recipe\s+([^>]*)>(.*?)</Recipe>", re.DOTALL)
ATTR = re.compile(r'(\w+)="([^"]*)"')
PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")
GUIDANCE = re.compile(
    r"\{/\*.*?\*/\}|<!--.*?-->", re.DOTALL
)  # template guidance comments
TEMPLATE_TEXT = (
    "Imperative title naming the result",
    "one sentence on what exists when the recipe is done",
    "First item",
)


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: Path, msg: str) -> None:
        self.errors.append(f"{where.relative_to(REPO)}: {msg}")

    def warn(self, where: Path, msg: str) -> None:
        self.warnings.append(f"{where.relative_to(REPO)}: {msg}")


def front_matter(text: str) -> dict[str, str]:
    m = FRONT.match(text)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def headings(text: str) -> list[str]:
    return [
        m.group(2).strip() for m in re.finditer(r"^(#{1,4}) (.+)$", text, re.MULTILINE)
    ]


def prose(text: str) -> str:
    """Text without fenced code blocks, front matter, and JSX attribute values."""
    text = FRONT.sub("", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"<[A-Z]\w*[^>]*>", "", text)
    return text


def is_serial_list(title: str) -> bool:
    """'A, B, and C' is a list; 'A thing, with a tag' is not."""
    m = COMMA_TAG.search(title)
    if not m:
        return True
    before = title[: m.start()]
    return "," in before  # a second comma earlier means a serial list


def check_titles(path: Path, text: str, rep: Report) -> None:
    fm = front_matter(text)
    no_code = re.sub(
        r"```.*?```", "", text, flags=re.DOTALL
    )  # a '#' comment in code is not a heading
    titles = (
        headings(no_code)
        + [fm.get("title", "")]
        + [m.group(2) for m in re.finditer(r'(title|label)="([^"]*)"', no_code)]
        + re.findall(
            r"^:::[a-z]+\[([^\]]*)\]", no_code, flags=re.MULTILINE
        )  # admonition titles
    )
    for t in titles:
        if not t:
            continue
        if BANNED.search(t):
            rep.error(path, f"banned construction in title: {t!r}")
        if COMMA_TAG.search(t) and not is_serial_list(t):
            rep.error(path, f"comma-and-tag title: {t!r}")
        if ";" in t:
            rep.error(path, f"two statements joined in a title: {t!r}")
        if CLAUSE_TAG.search(t):
            rep.error(path, f"title with a clause tag: {t!r}")
    body = prose(text)
    for m in BANNED.finditer(body):
        line = body.count("\n", 0, m.start()) + 1
        rep.error(path, f"banned construction near line {line}: {m.group(0)!r}")
    if SECOND_PERSON.search(body):
        rep.warn(
            path,
            "second person (you/your) in prose; use 'the organization' or a passive form",
        )


def check_index(path: Path, rep: Report) -> None:
    text = path.read_text(encoding="utf-8")
    fm = front_matter(text)
    for key in ("title", "description"):
        if not fm.get(key):
            rep.error(path, f"front matter lacks {key}")
    hs = headings(text)
    for section in INDEX_SECTIONS:
        if section not in hs:
            rep.error(path, f"missing section '## {section}'")
    if PLACEHOLDER.search(text):
        rep.error(path, "template placeholders remain")
    check_unfinished(path, text, rep)
    check_titles(path, text, rep)


def check_unfinished(path: Path, text: str, rep: Report) -> None:
    """Template guidance comments and sample text mean the page is unfinished."""
    if GUIDANCE.search(text):
        rep.error(path, "template guidance comments remain; replace them with content")
    body = prose(text)  # the recipe template is allowed inside a code block
    for sample in TEMPLATE_TEXT:
        if sample in body:
            rep.error(path, f"template sample text remains: {sample!r}")


def check_chapter(path: Path, rep: Report) -> None:
    text = path.read_text(encoding="utf-8")
    fm = front_matter(text)
    if not re.match(r"^\d+\. ", fm.get("title", "")):
        rep.error(path, 'front matter title must read "N. Title"')
    for key in ("sidebar_position", "description"):
        if not fm.get(key):
            rep.error(path, f"front matter lacks {key}")
    if "**Question:**" not in text:
        rep.error(path, "missing '**Question:** ...' line under the title")
    hs = headings(text)
    for section in CHAPTER_REQUIRED:
        if section not in hs:
            rep.error(path, f"missing section '## {section}'")
    for section in CHAPTER_RECOMMENDED:
        if section not in hs and not any(h.startswith(section) for h in hs):
            rep.warn(path, f"recommended section '## {section}' missing")
    if "<Checklist" not in text:
        rep.error(path, "missing Checklist component")
    recipes = list(RECIPE.finditer(text))
    if not recipes:
        rep.error(path, "no Recipe component found")
    for m in recipes:
        body = m.group(0)
        tm = re.search(r'title="([^"]+)"', m.group(1))
        rtitle = tm.group(1).split(" ")[0] if tm else "?"
        prose = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
        prose = re.sub(
            r"<Recipe\s+[^>]*>", "", prose
        )  # the tag's attributes are not a sentence
        prose = "\n".join(
            ln for ln in prose.splitlines() if not ln.lstrip().startswith("|")
        )
        prose = re.sub(r"\]\([^)]*\)", "]", prose)  # link targets do not count as words
        # paragraphs, list items, and lines ending in a colon are sentence boundaries
        prose = re.sub(r"\n(\s*(?:\d+\.|-|\*)\s+)", "\n\n", prose)
        units = [u for u in re.split(r"\n\s*\n|:\s*\n", prose) if u.strip()]
        flat = []
        for u in units:
            flat.extend(re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", u)))
        for sent in flat:
            if len(sent.split()) > 45:
                rep.warn(
                    path,
                    f"recipe {rtitle}: sentence of {len(sent.split())} words; split it",
                )
                break
        for cm in re.finditer(r"\n\n```", body):
            paragraph = body[: cm.start()].rstrip().split("\n\n")[-1].strip()
            if paragraph.startswith(("**Result:**", "<Recipe")):
                rep.warn(
                    path, f"recipe {rtitle}: code block without an introducing sentence"
                )
                break
        attrs = dict(ATTR.findall(m.group(1)))
        title = attrs.get("title", "")
        if not re.match(r"^\d+\.\d+ ", title):
            rep.error(path, f'recipe title must read "N.M Title": {title!r}')
        if attrs.get("level") not in LEVELS:
            rep.error(path, f"recipe {title!r}: level must be one of {sorted(LEVELS)}")
        for key in ("skills", "time"):
            if not attrs.get(key):
                rep.error(path, f"recipe {title!r}: missing {key}")
        if "**Result:**" not in m.group(2):
            rep.error(path, f"recipe {title!r}: missing '**Result:**' line")
        if "**What this does not do" not in m.group(
            2
        ) and "**What this does not check" not in m.group(2):
            rep.warn(path, f"recipe {title!r}: no 'What this does not do' note")
    if PLACEHOLDER.search(text):
        rep.error(path, "template placeholders remain")
    check_unfinished(path, text, rep)
    check_titles(path, text, rep)


def check_cookbook(folder: Path, rep: Report) -> None:
    index = folder / "index.mdx"
    if not index.exists():
        rep.error(folder, "index.mdx missing")
    else:
        check_index(index, rep)
    chapters = [
        p
        for p in sorted(folder.glob("*.mdx"))
        if p.name != "index.mdx" and "<Recipe" in p.read_text(encoding="utf-8")
    ]
    if not chapters:
        rep.error(folder, "no chapter files (files that use the Recipe component)")
    for p in chapters:
        check_chapter(p, rep)
    for p in sorted(folder.glob("*.md")) + [
        p
        for p in sorted(folder.glob("*.mdx"))
        if p not in chapters and p.name != "index.mdx"
    ]:
        text = p.read_text(encoding="utf-8")
        if PLACEHOLDER.search(text):
            rep.error(p, "template placeholders remain")
        check_unfinished(p, text, rep)
        check_titles(p, text, rep)
    for name in ("standards.md", "glossary.md"):
        if not (folder / name).exists():
            rep.warn(folder, f"{name} missing")
    if (
        not (folder / "contributing.md").exists()
        and not (folder / "contribute.md").exists()
    ):
        rep.warn(folder, "contributing.md missing")
    cid = folder.name
    if f"dirName: '{cid}'" not in SIDEBARS.read_text():
        rep.error(SIDEBARS, f"no sidebar for '{cid}'")
    if f"id: '{cid}'" not in CARDS.read_text():
        rep.error(CARDS, f"no landing card for '{cid}'")
    if not (STATIC / cid).exists():
        rep.warn(folder, f"no folder website/static/cookbook-files/{cid}/")


# --- cross-references -------------------------------------------------------

CHAPTER_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
MAP_FILE = REPO / "website" / "src" / "content" / "readinessMap.js"


def chapter_map(folder: Path) -> dict[str, tuple[int, str, set[str]]]:
    """slug -> (chapter number, title, recipe numbers) for every chapter of a cookbook."""
    out = {}
    for p in sorted(folder.glob("*.mdx")):
        text = p.read_text(encoding="utf-8")
        if "<Recipe" not in text:
            continue
        title = front_matter(text).get("title", "")
        m = re.match(r"(\d+)\.\s+(.*)", title)
        if not m:
            continue
        recipes = {
            r.group(1) for r in re.finditer(r'<Recipe\s+title="(\d+\.\d+)\s', text)
        }
        out[p.stem] = (int(m.group(1)), m.group(2).strip(), recipes)
    return out


def resolve_target(
    target: str, here: Path, maps: dict[str, dict]
) -> tuple[str, str] | None:
    """(cookbook id, slug) for a chapter link, or None when the link is not a chapter."""
    target = target.split("#")[0]
    if target.startswith("./") and target.endswith(".mdx"):
        return here.parent.name, target[2:-4]
    m = re.match(r"/cookbook/([^/]+)/([^/]+)/?$", target)
    if m and m.group(1) in maps:
        return m.group(1), m.group(2)
    return None


def check_references(
    folder: Path, maps: dict[str, dict], rep: Report, hidden: set[str] | None = None
) -> None:
    """Chapter and recipe numbers in prose and in the index agree with the files."""
    hidden = hidden or set()
    own = maps[folder.name]
    for p in sorted(folder.glob("*.mdx")) + sorted(folder.glob("*.md")):
        text = prose(p.read_text(encoding="utf-8"))
        for m in CHAPTER_LINK.finditer(text):
            label, target = m.group(1), m.group(2)
            into = re.match(r"/cookbook/([^/#?]+)", target.split("#")[0])
            if into and into.group(1) in hidden and folder.name not in hidden:
                rep.error(
                    p,
                    f"links into hidden cookbook '{into.group(1)}' ({target}); "
                    "a public page must not point at an unlisted one",
                )
            hit = resolve_target(target, p, maps)
            if not hit:
                continue
            cb, slug = hit
            if slug not in maps[cb]:
                if (
                    not (COOKBOOK / cb / f"{slug}.mdx").exists()
                    and not (COOKBOOK / cb / f"{slug}.md").exists()
                ):
                    rep.error(p, f"link to a page that does not exist: {target}")
                continue
            number, title, recipes = maps[cb][slug]
            before = text[max(0, m.start() - 60) : m.start()]
            n = re.search(r"\bchapters?\s+(\d+)\b", label, re.IGNORECASE) or (
                re.fullmatch(r"\d+", label.strip())
                and re.search(r"\bchapters\b", before, re.IGNORECASE)
                and re.match(r"(\d+)", label.strip())
            )
            if n and int(n.group(1)) != number:
                rep.error(p, f"'{label}' links to chapter {number} ({slug})")
            r = re.search(r"\brecipe\s+(\d+\.\d+)\b", label, re.IGNORECASE)
            if r and r.group(1) not in recipes:
                rep.error(
                    p, f"'{label}' links to {slug}, which has no recipe {r.group(1)}"
                )
            if re.match(r"\d+\.\s", label) and label.strip() != f"{number}. {title}":
                rep.error(
                    p,
                    f"index row '{label}' differs from the chapter title '{number}. {title}'",
                )
        # bare mentions: "recipe N.M" refers to this cookbook; "chapter N" has to exist
        own_numbers = {v[0]: k for k, v in own.items()}
        own_recipes = {r for _, _, rs in own.values() for r in rs}
        plain = CHAPTER_LINK.sub(" ", text)
        for r in re.finditer(r"\brecipes?\s+(\d+\.\d+)\b", plain, re.IGNORECASE):
            if r.group(1) not in own_recipes:
                rep.error(
                    p,
                    f"mentions recipe {r.group(1)}, which this cookbook does not have",
                )
        for c in re.finditer(r"\bchapters?\s+(\d+)\b", plain, re.IGNORECASE):
            if int(c.group(1)) not in own_numbers:
                rep.warn(
                    p,
                    f"mentions chapter {c.group(1)}, which this cookbook does not have",
                )


def check_readiness_map(maps: dict[str, dict], rep: Report) -> None:
    """Chapter and recipe numbers in the readiness map's labels agree with the files."""
    src = MAP_FILE.read_text(encoding="utf-8")
    cb_prefix = re.search(r"const CB = '/cookbook/([^/']+)/'", src)
    for m in re.finditer(
        r"\{to: (?:`\$\{CB\}([^`#]*)(?:#[^`]*)?`|'([^'#]+)(?:#[^']*)?'), label: '((?:[^'\\]|\\.)*)'\}",
        src,
    ):
        slug_from_cb, path, label = m.group(1), m.group(2), m.group(3)
        if slug_from_cb is not None and cb_prefix:
            cb, slug = cb_prefix.group(1), slug_from_cb
        elif path:
            mm = re.match(r"/cookbook/([^/]+)/([^/]*)/?$", path)
            if not mm:
                continue
            cb, slug = mm.group(1), mm.group(2)
        else:
            continue
        if cb not in maps:
            rep.error(MAP_FILE, f"link to unknown cookbook '{cb}'")
            continue
        if not slug:
            continue
        if slug not in maps[cb]:
            if (
                not (COOKBOOK / cb / f"{slug}.md").exists()
                and not (COOKBOOK / cb / f"{slug}.mdx").exists()
            ):
                rep.error(
                    MAP_FILE,
                    f"link to a page that does not exist: /cookbook/{cb}/{slug}",
                )
            continue
        number, _, recipes = maps[cb][slug]
        n = re.search(r"\bchapters?\s+(\d+)\b", label, re.IGNORECASE)
        if n and int(n.group(1)) != number:
            rep.error(
                MAP_FILE,
                f"label '{label[:60]}' names chapter {n.group(1)}; {cb}/{slug} is chapter {number}",
            )
        r = re.search(r"\brecipe\s+(\d+\.\d+)\b", label, re.IGNORECASE)
        if r and r.group(1) not in recipes:
            rep.error(
                MAP_FILE,
                f"label '{label[:60]}' names recipe {r.group(1)}, which {cb}/{slug} does not have",
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--id", help="check one cookbook")
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as errors"
    )
    args = parser.parse_args(argv)

    folders = [
        p for p in sorted(COOKBOOK.iterdir()) if p.is_dir() and p.name not in SKIP
    ]
    if args.id:
        folders = [p for p in folders if p.name == args.id]
        if not folders:
            sys.exit(f"no cookbook named {args.id!r}")

    rep = Report()
    hidden = hidden_cookbooks()
    for cid in sorted(hidden):
        if not (COOKBOOK / cid).is_dir():
            rep.error(
                VISIBILITY, f"hidden cookbook '{cid}' has no folder under cookbook/"
            )
    for cid in sorted(indexed_cookbooks()):
        if not (COOKBOOK / cid).is_dir():
            rep.error(
                VISIBILITY, f"indexed cookbook '{cid}' has no folder under cookbook/"
            )
        elif cid in hidden:
            rep.error(VISIBILITY, f"cookbook '{cid}' is listed as both hidden and indexed")
    maps = {f.name: chapter_map(f) for f in folders}
    for folder in folders:
        check_cookbook(folder, rep)
        check_references(folder, maps, rep, hidden)
    if not args.id:
        check_readiness_map(maps, rep)
        check_versions(folders, rep)

    for w in rep.warnings:
        print(f"warning  {w}")
    for e in rep.errors:
        print(f"error    {e}")
    print(
        f"\n{len(folders)} cookbook(s): {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)"
    )
    failed = bool(rep.errors) or (args.strict and bool(rep.warnings))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
