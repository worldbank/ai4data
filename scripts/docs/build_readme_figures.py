"""Build the SVG figures the repository README embeds.

Usage:
    python scripts/docs/build_readme_figures.py

Writes docs/images/program-pillars.svg and docs/images/program-pillars-dark.svg,
the two-pillar overview of the program, from the groups listed below. The README
shows the dark variant through a <picture> element when the viewer prefers a dark
colour scheme. Exit status 0 on success.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "images"

PILLARS = [
    {
        "title": "AI for Data",
        "lede": "AI applied inside the statistical process, under human review",
        "groups": [
            ("Data production", [
                "Statistical classification and coding",
                "Data Snapshots from PDF documents",
                "Small and agentic AI",
                "Synthetic data (REaLTabFormer)",
            ]),
            ("Data quality and metadata", [
                "Metadata Reviewer and quality scoring",
                "Metadata Augmentation",
                "Anomaly detection and explanation",
                "Responsible AI guidance",
            ]),
            ("Discovery and trustworthy dissemination", [
                "Search by meaning over catalogs",
                "Proof-Carrying Numbers",
                "Monitoring of data use",
            ]),
            ("Methods and open tools", [
                "Inclusive and multilingual models",
                "Open benchmarks and evaluation",
            ]),
        ],
    },
    {
        "title": "Data for AI",
        "lede": "Development data made usable by AI systems through open standards and infrastructure",
        "groups": [
            ("Assessment", [
                "AI-readiness assessment framework",
                "Operational companion",
            ]),
            ("Standards and metadata", [
                "AI-ready data framework",
                "AI-ready metadata and standards: DDI, SDMX, DCAT, Croissant, schema.org",
            ]),
            ("Infrastructure", [
                "Model Context Protocol servers over official statistics",
                "AI-assisted metadata platforms: Metadata Editor, NADA",
            ]),
            ("Semantic knowledge", [
                "Global Question Bank",
                "Ontologies and knowledge graphs",
            ]),
        ],
    },
]

THEMES = {
    "light": {
        "bg": "none", "panel": "#f6f8fa", "panel_border": "#d0d7de",
        "card": "#ffffff", "card_border": "#d8dee4",
        "text": "#1f2328", "muted": "#57606a",
        "accents": ["#2a5aa7", "#13917f"],
    },
    "dark": {
        "bg": "none", "panel": "#161b22", "panel_border": "#30363d",
        "card": "#0d1117", "card_border": "#30363d",
        "text": "#e6edf3", "muted": "#9198a1",
        "accents": ["#79b0ff", "#3fd0b6"],
    },
}

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
W = 1200
GAP = 20
PAD = 22
PANEL_W = (W - GAP) // 2
CARD_PAD = 14
LINE = 20          # line height of item text (14px font)
TITLE_H = 24       # group title line
CHARS = 66         # wrap width for 14px text inside a card


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def card_lines(items: list[str]) -> list[str]:
    return textwrap.wrap(" · ".join(items), CHARS)


def build(theme: dict) -> str:
    # Lay out both panels; all cards in a row share a height so the panels align.
    rows = max(len(p["groups"]) for p in PILLARS)
    row_h = []
    for r in range(rows):
        n = max(len(card_lines(p["groups"][r][1])) for p in PILLARS if r < len(p["groups"]))
        row_h.append(CARD_PAD * 2 + TITLE_H + n * LINE)
    header_h = 78
    panel_h = PAD + header_h + sum(row_h) + (rows - 1) * 12 + PAD
    H = panel_h + 2

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'font-family="{FONT}" role="img" aria-labelledby="t">',
        '<title id="t">The two pillars of the AI for Data – Data for AI program and the workstreams under each</title>',
    ]
    for i, p in enumerate(PILLARS):
        x = i * (PANEL_W + GAP) + 1
        accent = theme["accents"][i]
        out.append(f'<rect x="{x}" y="1" width="{PANEL_W}" height="{panel_h}" rx="12" fill="{theme["panel"]}" stroke="{theme["panel_border"]}"/>')
        out.append(f'<rect x="{x}" y="1" width="{PANEL_W}" height="6" rx="3" fill="{accent}"/>')
        tx = x + PAD
        out.append(f'<text x="{tx}" y="{PAD + 26}" font-size="24" font-weight="700" fill="{accent}">{esc(p["title"])}</text>')
        for j, line in enumerate(textwrap.wrap(p["lede"], 70)):
            out.append(f'<text x="{tx}" y="{PAD + 50 + j * 18}" font-size="14" fill="{theme["muted"]}">{esc(line)}</text>')
        y = PAD + header_h + 1
        for r, (name, items) in enumerate(p["groups"]):
            h = row_h[r]
            out.append(f'<rect x="{tx}" y="{y}" width="{PANEL_W - 2 * PAD}" height="{h}" rx="8" fill="{theme["card"]}" stroke="{theme["card_border"]}"/>')
            out.append(f'<rect x="{tx}" y="{y + 10}" width="4" height="{h - 20}" rx="2" fill="{accent}"/>')
            cx = tx + CARD_PAD + 8
            out.append(f'<text x="{cx}" y="{y + CARD_PAD + 15}" font-size="15" font-weight="600" fill="{theme["text"]}">{esc(name)}</text>')
            for k, line in enumerate(card_lines(items)):
                out.append(f'<text x="{cx}" y="{y + CARD_PAD + TITLE_H + 14 + k * LINE}" font-size="14" fill="{theme["muted"]}">{esc(line)}</text>')
            y += h + 12
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "program-pillars.svg").write_text(build(THEMES["light"]))
    (OUT / "program-pillars-dark.svg").write_text(build(THEMES["dark"]))
    print("wrote", OUT / "program-pillars.svg", "and the dark variant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
