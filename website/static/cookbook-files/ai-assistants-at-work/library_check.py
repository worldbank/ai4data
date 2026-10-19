"""Check the prompt library for the parts every prompt needs.

Reads the prompt library (YAML) and verifies, with a JSON Schema, that
the library has a version and an owner and that every prompt has an id,
a purpose, a data tier from public, internal, and confidential, a
review, a list of placeholders, and a template. It then checks that
every placeholder listed appears in the template and that every brace
placeholder in the template is listed, and that the template has the
three parts (source, task, format or constraints). Exit status 1 when
anything is missing.

Usage:
    python library_check.py prompt_library.yaml

Requires PyYAML and jsonschema.

What this does not check: whether a prompt is a good one. The unit's
session of chapter 6 and the review of each use are where prompts are
improved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

SCHEMA = {
    "type": "object",
    "required": ["version", "owner", "prompts"],
    "properties": {
        "version": {"type": "string"},
        "owner": {"type": "string", "minLength": 1},
        "prompts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "purpose",
                    "data_tier",
                    "review",
                    "placeholders",
                    "template",
                ],
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                    "purpose": {"type": "string", "minLength": 1},
                    "data_tier": {"enum": ["public", "internal", "confidential"]},
                    "review": {"type": "string", "minLength": 1},
                    "placeholders": {"type": "array", "items": {"type": "string"}},
                    "template": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}
PARTS = ["Source:", "Task:", "Format:", "Constraints:"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("library", type=Path)
    args = parser.parse_args(argv)
    lib = yaml.safe_load(args.library.read_text(encoding="utf-8"))
    problems = [
        f"schema: {e.json_path}: {e.message}"
        for e in sorted(
            Draft202012Validator(SCHEMA).iter_errors(lib), key=lambda e: e.path
        )
    ]
    prompts = lib.get("prompts", []) if isinstance(lib, dict) else []
    ids = [p.get("id") for p in prompts]
    for dup in {i for i in ids if ids.count(i) > 1}:
        problems.append(f"duplicate id: {dup}")
    for p in prompts:
        if not isinstance(p, dict) or "template" not in p:
            continue
        in_template = set(re.findall(r"\{(\w+)\}", p["template"]))
        listed = set(p.get("placeholders", []))
        for ph in sorted(listed - in_template):
            problems.append(
                f"{p.get('id')}: placeholder {ph!r} listed and not in the template"
            )
        for ph in sorted(in_template - listed):
            problems.append(
                f"{p.get('id')}: placeholder {ph!r} in the template and not listed"
            )
        for part in PARTS:
            if part not in p["template"]:
                problems.append(f"{p.get('id')}: template has no '{part}' part")
    tiers = {}
    for p in prompts:
        tiers[p.get("data_tier")] = tiers.get(p.get("data_tier"), 0) + 1
    print(
        f"{len(prompts)} prompts; tiers: "
        + ", ".join(
            f"{n} {t}" for t, n in sorted(tiers.items(), key=lambda x: str(x[0]))
        )
    )
    for p in problems:
        print(f"problem: {p}")
    print("result: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
