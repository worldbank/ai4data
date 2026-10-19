#!/usr/bin/env python3
"""Convert selected Jupyter notebooks into Docusaurus-ready Markdown pages.

Generated files are build artifacts (see website/docs/notebooks/.gitignore) and
are regenerated on every docs build, both locally (`npm run convert-notebooks`
in website/) and in CI (.github/workflows/gh-pages.yml).
"""

import re
import sys
from pathlib import Path

from nbconvert import MarkdownExporter
from traitlets.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEST_ROOT = REPO_ROOT / "docs" / "notebooks"

# (source notebook, destination markdown path relative to DEST_ROOT, title)
NOTEBOOKS = [
    (
        REPO_ROOT / "notebooks" / "metadata-quality-assessment-with-llm.ipynb",
        "metadata-quality-assessment-with-llm.md",
    ),
    (
        REPO_ROOT
        / "notebooks"
        / "data-anomaly"
        / "Timeseries_Anomaly_Explanation_with_LLMs.ipynb",
        "data-anomaly/timeseries-anomaly-explanation-with-llms.md",
    ),
]


def extract_title(nb_node) -> str:
    for cell in nb_node.cells:
        if cell.cell_type == "markdown":
            match = re.search(r"^#\s+(.+)$", cell.source, re.MULTILINE)
            if match:
                return match.group(1).strip()
    return "Notebook"


def convert(src: Path, dest_rel: str) -> None:
    dest = DEST_ROOT / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    resources_dir = dest.stem

    exporter = MarkdownExporter(config=Config())

    import nbformat

    nb_node = nbformat.read(src, as_version=4)
    title = extract_title(nb_node)

    # Stale `filenames` metadata (left over from a previous nbconvert run) can collide
    # across outputs; drop it so ExtractOutputPreprocessor derives fresh, unique names.
    for cell in nb_node.cells:
        for output in cell.get("outputs", []):
            output.get("metadata", {}).pop("filenames", None)

    body, resources = exporter.from_notebook_node(
        nb_node,
        resources={
            "output_files_dir": f"{resources_dir}_files",
            "unique_key": resources_dir,
        },
    )

    # Drop the first H1 (Docusaurus renders `title` from front matter as the page heading).
    body = re.sub(r"^#\s+.+\n+", "", body, count=1)

    for filename, data in resources.get("outputs", {}).items():
        out_path = dest.parent / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)

    front_matter = (
        "---\n"
        f"title: {title}\n"
        "format: md\n"
        "---\n\n"
        f"> Generated from [`{src.relative_to(REPO_ROOT)}`]"
        f"(https://github.com/worldbank/ai4data/blob/main/{src.relative_to(REPO_ROOT)}). "
        "Do not edit this page directly — edit the notebook and re-run the docs build.\n\n"
    )

    dest.write_text(front_matter + body, encoding="utf-8")
    print(f"converted {src.relative_to(REPO_ROOT)} -> {dest.relative_to(REPO_ROOT)}")


def main() -> int:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    for src, dest_rel in NOTEBOOKS:
        if not src.exists():
            print(f"missing notebook: {src}", file=sys.stderr)
            return 1
        convert(src, dest_rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
