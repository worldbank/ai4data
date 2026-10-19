"""Check a guidance resource against the manifest and the required sections.

A guidance resource is the text an assistant reads before using the
server: what it provides, how to use the tools, how to cite, the licence,
the limitations, and when to refuse. This check reads the guidance (a
Markdown file) and the tool manifest and reports:

    missing sections   required headings absent from the guidance
    tools not named    tools in the manifest the guidance never mentions
    unknown tools      names in the guidance that are neither a tool nor
                       an input or output field of the manifest
    length             words, since a guidance text an assistant has to
                       read on every session should stay short

Exit status 1 when a section is missing or a tool is unnamed or unknown.
Standard library only.

Usage:
    python check_guidance.py guidance_resource.md tool_manifest.json

What this does not do: it checks presence, not quality. Whether the
citation example is right or the limitations are complete is the
curator's judgement, and whether the guidance changes the assistant's
behaviour is measured by the evaluation chapter's question set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED = [
    "What this server provides",
    "How to use the tools",
    "How to cite",
    "Licence",
    "Limitations",
    "When to refuse or decline",
]
MAX_WORDS = 600


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("guidance", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    text = args.guidance.read_text(encoding="utf-8")
    with args.manifest.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    headings = {
        h.strip().lower()
        for h in re.findall(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE)
    }
    missing = [r for r in REQUIRED if r.lower() not in headings]
    tools = {t["name"] for t in manifest.get("tools", [])}
    fields = {"idno"}

    def collect(
        node,
    ) -> None:  # every key and identifier-like string in the manifest is a field name
        if isinstance(node, dict):
            fields.update(k for k in node if re.fullmatch(r"[a-z_][a-z0-9_]*", k))
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)
        elif isinstance(node, str) and re.fullmatch(r"[a-z_][a-z0-9_]*", node):
            fields.add(node)

    collect(manifest)
    named = set(re.findall(r"`([a-z_][a-z0-9_]*)`", text))
    unnamed = sorted(tools - named)
    unknown = sorted(
        n for n in named if "_" in n and n not in tools and n not in fields
    )
    words = len(text.split())

    print(
        f"{args.guidance.name}: {words} words, {len(headings)} headings; manifest {manifest.get('server')} {manifest.get('version')} with {len(tools)} tool(s)"
    )
    for r in REQUIRED:
        print(f"  {'ok     ' if r not in missing else 'MISSING'} {r}")
    print(
        f"tools named: {len(tools) - len(unnamed)}/{len(tools)}"
        + (f"; not named: {', '.join(unnamed)}" if unnamed else "")
    )
    if unknown:
        print(f"unknown tool names in guidance: {', '.join(unknown)}")
    if words > MAX_WORDS:
        print(
            f"warning: {words} words is above the {MAX_WORDS}-word guideline; assistants read this every session"
        )
    failed = bool(missing or unnamed or unknown)
    print("result: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
