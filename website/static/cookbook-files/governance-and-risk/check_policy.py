"""Check an AI-use policy for the parts that make it usable.

Reads a policy in Markdown and reports whether it has the seven sections
(scope, principles, permitted uses, prohibited uses, controls, roles,
review), whether the permitted and prohibited lists have at least one
specific item each, whether an approver and a review date are stated,
and whether any template placeholder such as "[add]" or "[role]" is
left. Exit status 1 when a section or a required line is missing or a
placeholder remains. Standard library only.

Usage:
    python check_policy.py ai_use_policy.md

What this does not check: whether the permitted uses are the right ones
for the organization, and whether the policy is followed. The register
and the decision log show that.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SECTIONS = [
    "Scope",
    "Principles",
    "Permitted uses",
    "Prohibited uses",
    "Controls",
    "Roles",
    "Review",
]
PLACEHOLDER = re.compile(r"\[(add|role|name|date|period|interval|number|office)\]")


def section_text(text: str, name: str) -> str:
    """The body of the H2 section whose heading ends with `name`, or ''."""
    m = re.search(
        rf"^##\s+(?:\d+\.\s+)?{re.escape(name)}\s*$(.*?)(?=^##\s|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return m.group(1) if m else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("policy", type=Path)
    args = parser.parse_args(argv)
    text = args.policy.read_text(encoding="utf-8")
    problems: list[str] = []

    for name in SECTIONS:
        body = section_text(text, name)
        status = "ok     " if body.strip() else "MISSING"
        print(f"  {status} {name}")
        if not body.strip():
            problems.append(f"section missing: {name}")

    for name in ("Permitted uses", "Prohibited uses"):
        items = re.findall(r"^\s*- (.+)$", section_text(text, name), flags=re.MULTILINE)
        items = [i for i in items if not PLACEHOLDER.search(i)]
        print(f"{name.lower()}: {len(items)} item(s)")
        if not items:
            problems.append(f"{name.lower()}: no specific item")

    approver = re.search(r"^\*\*Approved by:\*\*\s*(.+)$", text, flags=re.MULTILINE)
    review = re.search(
        r"^\*\*Review date:\*\*\s*(\d{4}-\d{2}-\d{2})", text, flags=re.MULTILINE
    )
    print(f"approver: {approver.group(1).strip() if approver else 'MISSING'}")
    print(f"review date: {review.group(1) if review else 'MISSING'}")
    if not approver or PLACEHOLDER.search(approver.group(1)):
        problems.append("approver not stated")
    if not review:
        problems.append("review date not stated as YYYY-MM-DD")

    placeholders = sorted({m.group(0) for m in PLACEHOLDER.finditer(text)})
    print(f"placeholders left: {', '.join(placeholders) if placeholders else 'none'}")
    if placeholders:
        problems.append(f"placeholders left: {', '.join(placeholders)}")

    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
