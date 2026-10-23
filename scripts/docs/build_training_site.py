#!/usr/bin/env python3
"""Generate the Training section of the docs site from the training/ folder.

The training folder stays the source: its READMEs, participant resources, and
notebooks are what GitHub shows and what reviewers edit. This script writes a
Markdown copy of them into website/generated/training/ (gitignored), which the
`training` docs instance of the site reads. The site build runs it before
Docusaurus, through `npm run convert-notebooks`.

    python3 scripts/docs/build_training_site.py

What the script does to the source:

- The series README becomes the section's index page, and each session README
  becomes the session's overview page. Sections listed in OMIT_SECTIONS are
  left off the site.
- Each notebook becomes a page with its committed outputs, an "Open in Colab"
  button, and a link to the notebook on GitHub. Colab section links
  (`#scrollTo=section-N`) become heading anchors, and Colab links between the
  session's notebooks become links between the pages.
- Relative links to files that are on the site are rewritten to the generated
  pages. Relative links to anything else (data files, scripts, folders) point
  to the file on GitHub. Absolute links to the published site become site-
  internal links.
- Notebook HTML outputs keep their markup. Inline scripts are removed; the
  site's pcnDetail client module provides the behaviour they had in Jupyter.

No dependencies beyond the standard library.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "training"
OUT = REPO / "website" / "generated" / "training"
GITHUB = "https://github.com/worldbank/ai4data"
BLOB = f"{GITHUB}/blob/main"
TREE = f"{GITHUB}/tree/main"
EDIT = f"{GITHUB}/edit/main"
COLAB = "https://colab.research.google.com/github/worldbank/ai4data/blob/main"
SITE = "https://worldbank.github.io/ai4data"

# Sections of a README that concern the team preparing a session and stay on GitHub.
OMIT_SECTIONS = {
    "training/README.md": ["Layout of a session folder", "Review", "Adding a session"],
    "session": ["Roles", "Definition of ready"],
}

# Participant-facing Markdown pages of a session, in sidebar order: (source, output, label).
SESSION_PAGES = [
    ("resources/README.md", "resources/index.md", "Participant guide"),
    ("resources/glossary.md", "resources/glossary.md", None),
    ("resources/evaluation-examples.md", "resources/testing-on-fixed-questions.md", None),
    ("resources/next-steps.md", "resources/next-steps.md", None),
    ("assets/README.md", "worked-example-data.md", "Worked-example data"),
]


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    return re.sub(r"[\s_]+", "-", text).strip("-")


def yaml_str(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def front_matter(**fields) -> str:
    lines = ["---"]
    for key, value in fields.items():
        if value is None:
            continue
        lines.append(f"{key}: {value if isinstance(value, (int, bool)) else yaml_str(str(value))}")
    lines.append("format: md")
    lines.append("---\n")
    return "\n".join(lines) + "\n"


def split_title(markdown: str) -> tuple[str, str]:
    match = re.search(r"^#\s+(.+?)\s*$", markdown, re.MULTILINE)
    if not match:
        return "Untitled", markdown
    body = markdown[: match.start()] + markdown[match.end():]
    return match.group(1), body.lstrip("\n")


def omit_sections(markdown: str, headings: list[str]) -> str:
    """Remove each `## heading` section up to the next heading of level 2 or higher."""
    for heading in headings:
        pattern = rf"^##\s+{re.escape(heading)}\s*\n.*?(?=^#{{1,2}}\s|\Z)"
        markdown = re.sub(pattern, "", markdown, flags=re.MULTILINE | re.DOTALL)
    return markdown.rstrip() + "\n"


class Site:
    """Maps source files to generated pages and rewrites links between them."""

    def __init__(self) -> None:
        self.pages: dict[Path, Path] = {}  # source file -> generated file
        self.colab: dict[str, Path] = {}  # Colab URL of a notebook -> generated file

    def add(self, src: Path, out: Path) -> None:
        self.pages[src.resolve()] = out
        if src.suffix == ".ipynb":
            self.colab[f"{COLAB}/{src.relative_to(REPO).as_posix()}"] = out

    def rewrite_links(self, markdown: str, src: Path, out: Path, *, colab_to_page: bool) -> str:
        def target(url: str) -> str:
            url = url.strip()
            if url.startswith(SITE + "/"):
                path = "/" + url[len(SITE) + 1:]
                # Static files (the documents page) are not routes; pathname:// keeps the link as is.
                return f"pathname://{path}" if re.search(r"\.html(#|$)", path) else path
            if url.startswith(SITE):
                return "/"
            if colab_to_page:
                base, _, frag = url.partition("#")
                if base in self.colab:
                    return self.relative(out, self.colab[base]) + (f"#{frag}" if frag else "")
            if url.startswith("#scrollTo="):
                return "#" + url[len("#scrollTo="):]
            if re.match(r"^[a-z][a-z0-9+.-]*:", url) or url.startswith("#") or url.startswith("/"):
                return url
            path, _, frag = url.partition("#")
            resolved = (src.parent / path).resolve()
            frag = f"#{frag}" if frag else ""
            if resolved in self.pages:
                return self.relative(out, self.pages[resolved]) + frag
            rel = resolved.relative_to(REPO).as_posix()
            return f"{TREE if resolved.is_dir() else BLOB}/{rel}{frag}"

        def link(match: re.Match) -> str:
            return f"{match.group(1)}({target(match.group(2))}{match.group(3) or ''})"

        parts = re.split(r"(^```.*?^```)", markdown, flags=re.MULTILINE | re.DOTALL)
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r"(\]\s?)\(([^()\s]+)(\s+\"[^\"]*\")?\)", link, parts[i])
            parts[i] = re.sub(r'href="([^"]+)"', lambda m: f'href="{target(m.group(1))}"', parts[i])
        return "".join(parts)

    @staticmethod
    def relative(frm: Path, to: Path) -> str:
        rel = Path(*([".."] * (len(frm.parent.relative_to(OUT).parts)))) / to.relative_to(OUT)
        return rel.as_posix() if rel.as_posix().startswith(".") else "./" + rel.as_posix()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_page(site: Site, src: Path, out: Path, *, position: int, label: str | None = None,
                  omit: list[str] | None = None, slug: str | None = None, note: str | None = None) -> None:
    title, body = split_title(src.read_text(encoding="utf-8"))
    if omit:
        body = omit_sections(body, omit)
    if note:
        body = body.rstrip() + "\n\n## Source files\n\n" + note + "\n"
    body = site.rewrite_links(body, src, out, colab_to_page=False)
    rel = src.relative_to(REPO).as_posix()
    write(out, front_matter(title=title, sidebar_label=label, sidebar_position=position, slug=slug,
                            custom_edit_url=f"{EDIT}/{rel}") + body)


# --- Notebooks ----------------------------------------------------------------

DATA_IMAGE = re.compile(r"!\[([^\]]*)\]\(data:image/(svg\+xml|png|jpeg);base64,([A-Za-z0-9+/=]+)\)")
EXT = {"svg+xml": "svg", "png": "png", "jpeg": "jpg"}


def source_text(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def fence(text: str, lang: str = "") -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{lang}\n{text.rstrip()}\n{ticks}"


OUTPUT_CSS = REPO / "website" / "generated" / "training-outputs.css"
STYLES: dict[str, None] = {}  # <style> blocks of notebook outputs, in first-seen order


def html_block(markup: str) -> str:
    """Return HTML that CommonMark keeps as one HTML block and React can render."""
    markup = re.sub(r"<script\b.*?</script>", "", markup, flags=re.DOTALL | re.IGNORECASE)
    # The site's HTML minifier rewrites inline <style> text, which breaks React hydration,
    # so output styles go to one stylesheet that the site loads (see docusaurus.config.js).
    for css in re.findall(r"<style\b[^>]*>(.*?)</style>", markup, flags=re.DOTALL | re.IGNORECASE):
        STYLES.setdefault(css.strip(), None)
    markup = re.sub(r"<style\b.*?</style>", "", markup, flags=re.DOTALL | re.IGNORECASE)
    # A blank line ends a CommonMark HTML block, so lines are joined without blank lines.
    markup = re.sub(r"\n[ \t]*\n+", "\n", markup.strip())
    # Whitespace between table tags becomes text nodes that React rejects inside a table.
    table = r"/?(?:table|thead|tbody|tfoot|tr|td|th|colgroup|col)\b"
    markup = re.sub(rf"(<{table}[^>]*>)\s+(?=<)", r"\1", markup)
    markup = re.sub(rf"\s+(?=<{table})", "", markup)
    return f'<div class="nb-output nb-output--html">\n{markup}\n</div>'


def notebook_page(site: Site, src: Path, out: Path, *, position: int, assets_dir: Path) -> None:
    nb = json.loads(src.read_text(encoding="utf-8"))
    rel = src.relative_to(REPO).as_posix()
    blocks: list[str] = []
    title = None

    def save_image(kind: str, data: bytes) -> str:
        name = hashlib.sha1(data).hexdigest()[:12] + "." + EXT[kind]
        (assets_dir).mkdir(parents=True, exist_ok=True)
        (assets_dir / name).write_bytes(data)
        return Site.relative(out, assets_dir / name)

    for cell in nb["cells"]:
        text = source_text(cell)
        if cell["cell_type"] == "markdown":
            if title is None:
                found, text = split_title(text)
                if found != "Untitled":
                    title = found
            cell_id = cell.get("metadata", {}).get("id") or cell.get("id", "")
            if cell_id.startswith("section-"):
                text = re.sub(r"^(#{2,3}\s+.+?)\s*$", rf"\1 {{#{cell_id}}}", text, count=1, flags=re.MULTILINE)
            text = site.rewrite_links(text, src, out, colab_to_page=True)
            # Embedded images become files next to the page; done after link rewriting,
            # which would otherwise resolve their relative paths against the source folder.
            text = DATA_IMAGE.sub(
                lambda m: f"![{m.group(1)}]({save_image(m.group(2), base64.b64decode(m.group(3)))})", text)
            blocks.append(text)
        elif cell["cell_type"] == "code":
            if text.strip():
                blocks.append(fence(text, "python"))
            for output in cell.get("outputs", []):
                blocks.extend(render_output(output, save_image))
    title = title or src.stem
    header = (
        '<p class="training-actions">'
        f'<a class="button button--primary button--sm" href="{COLAB}/{rel}">Open in Colab</a> '
        f'<a class="button button--secondary button--sm" href="{BLOB}/{rel}">View the notebook on GitHub</a>'
        "</p>\n\n"
        "The outputs on this page are the ones committed with the notebook. Without an API key, "
        "the notebook replays recorded model answers, and the same recordings appear here.\n"
    )
    short = re.sub(r"^\d+\s*·\s*", "", title)
    write(out, front_matter(title=title, sidebar_label=title.split(" · ")[0] + " · " + short,
                            sidebar_position=position, slug=out.stem, custom_edit_url=f"{BLOB}/{rel}")
          + header + "\n" + "\n\n".join(b.strip("\n") for b in blocks if b.strip()) + "\n")


def render_output(output: dict, save_image) -> list[str]:
    kind = output.get("output_type")
    if kind == "stream":
        text = source_text({"source": output.get("text", "")})
        return [fence(text, "text")] if text.strip() else []
    if kind == "error":
        return [fence("\n".join(output.get("traceback", [])) or output.get("evalue", ""), "text")]
    data = output.get("data", {})
    if "application/vnd.jupyter.widget-view+json" in data:
        return []
    if "text/html" in data:
        return [html_block(source_text({"source": data["text/html"]}))]
    for mime, ext in (("image/svg+xml", "svg+xml"), ("image/png", "png"), ("image/jpeg", "jpeg")):
        if mime in data:
            raw = data[mime]
            raw = "".join(raw) if isinstance(raw, list) else raw
            payload = raw.encode() if mime == "image/svg+xml" else base64.b64decode(raw)
            return [f"![Output image]({save_image(ext, payload)})"]
    if "text/markdown" in data:
        return [source_text({"source": data["text/markdown"]})]
    if "text/plain" in data:
        return [fence(source_text({"source": data["text/plain"]}), "text")]
    return []


# --- Build --------------------------------------------------------------------

def session_slug(folder: Path) -> str:
    match = re.match(r"session-(\d+)", folder.name)
    return f"session-{match.group(1)}" if match else folder.name


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    STYLES.clear()
    site = Site()
    sessions = sorted(p for p in SRC.iterdir() if p.is_dir() and re.match(r"session-\d+", p.name))

    # Register every page first so that links between them resolve in any order.
    plan = []
    site.add(SRC / "README.md", OUT / "index.md")
    for folder in sessions:
        dest = OUT / session_slug(folder)
        site.add(folder / "README.md", dest / "index.md")
        notebooks = sorted((folder / "notebooks").glob("[0-9][0-9]-*.ipynb"))
        nb_out = []
        for nb in notebooks:
            title = nb.stem
            first = json.loads(nb.read_text(encoding="utf-8"))["cells"][0]
            found, _ = split_title(source_text(first))
            if found != "Untitled":
                title = found
            out = dest / "notebooks" / f"{slugify(title)}.md"
            site.add(nb, out)
            nb_out.append((nb, out))
        pages = []
        for src_rel, out_rel, label in SESSION_PAGES:
            if (folder / src_rel).exists():
                site.add(folder / src_rel, dest / out_rel)
                pages.append((folder / src_rel, dest / out_rel, label))
        plan.append((folder, dest, nb_out, pages))

    markdown_page(site, SRC / "README.md", OUT / "index.md", position=1, label="Training series",
                  omit=OMIT_SECTIONS["training/README.md"], slug="/",
                  note=f"The pages of this section are generated from the [training folder]({TREE}/training) "
                       "of the repository. The slide decks, the demonstration scripts, and the scripts that "
                       "regenerate the notebooks are kept there.")
    for folder, dest, nb_out, pages in plan:
        title, _ = split_title((folder / "README.md").read_text(encoding="utf-8"))
        write(dest / "_category_.json", json.dumps(
            {"label": title.split(" · ")[0], "position": int(re.sub(r"\D", "", dest.name) or 99) + 1,
             "collapsed": False}, indent=2))
        markdown_page(site, folder / "README.md", dest / "index.md", position=1,
                      label=title.split(" · ", 1)[-1],
                      omit=OMIT_SECTIONS["session"],
                      note=f"The session's slides, demonstration script, worked-example files, and notebook "
                           f"sources are in the [session folder]({TREE}/{folder.relative_to(REPO).as_posix()}) "
                           "on GitHub.")
        write(dest / "notebooks" / "_category_.json",
              json.dumps({"label": "Notebooks", "position": 2, "collapsed": False}, indent=2))
        for i, (nb, out) in enumerate(nb_out, start=1):
            notebook_page(site, nb, out, position=i, assets_dir=out.parent / "img")
        resources = [p for p in pages if p[1].parent.name == "resources"]
        if resources:
            write(dest / "resources" / "_category_.json", json.dumps(
                {"label": "Participant resources", "position": 3, "collapsed": True,
                 "link": {"type": "doc", "id": f"{dest.name}/resources/index"}}, indent=2))
        for i, (src, out, label) in enumerate(pages, start=1):
            markdown_page(site, src, out, position=i if out.parent.name == "resources" else 4, label=label)
        print(f"training site: {dest.relative_to(REPO)} ({len(nb_out)} notebooks, {len(pages)} pages)")
    write(OUTPUT_CSS, "/* Styles of the training notebook outputs, generated by "
          "scripts/docs/build_training_site.py. */\n" + "\n".join(STYLES) + "\n")


if __name__ == "__main__":
    build()
