"""Check a tool manifest against the design rules for a statistics server.

Reads a manifest (JSON: server, version, description, tools, resources) and
checks each tool against the rules of the cookbook's tools chapter:

errors (exit status 1)
  - a tool name that is not a lower-case verb phrase in snake_case
  - a description shorter than 20 words (the description is the model's
    only documentation)
  - a tool that is not marked read-only (readOnlyHint true)
  - an input without a type or a description
  - a data tool (one that returns observations) whose outputs lack a
    provenance field: SERIES, UNIT_MEASURE, RELEASE, SOURCE_URL, license,
    citation
  - a tool without an example call
  - more than ten tools

warnings
  - no guidance resource (a resource whose description tells the model how
    to use the tools)
  - a search tool without a limit input

The manifest is a design document; it is not the server. The checks apply
to whatever the server's tool list reports as well, when exported in the
same form. Uses jsonschema for the manifest's shape.

Usage:
    python check_tool_manifest.py tool_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

PROVENANCE = ["SERIES", "UNIT_MEASURE", "RELEASE", "SOURCE_URL", "license", "citation"]
MAX_TOOLS = 10
MIN_WORDS = 20

# The shape every manifest has to have; the rules below cover what a schema cannot say.
SCHEMA = {
    "type": "object",
    "required": ["server", "version", "tools"],
    "properties": {
        "server": {"type": "string"},
        "version": {"type": "string"},
        "tools": {
            "type": "array",
            "maxItems": MAX_TOOLS,
            "items": {
                "type": "object",
                "required": ["name", "description", "readOnlyHint", "example"],
                "properties": {
                    "name": {
                        "type": "string",
                        "pattern": "^[a-z][a-z0-9]*(_[a-z0-9]+)+$",
                    },
                    "description": {"type": "string"},
                    "readOnlyHint": {"const": True},
                    "inputs": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "required": ["type", "description"],
                        },
                    },
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "example": {
                        "anyOf": [
                            {"type": "string", "minLength": 1},
                            {"type": "object", "minProperties": 1},
                        ]
                    },
                },
            },
        },
        "resources": {"type": "array", "items": {"type": "object"}},
    },
}


def describe(error, manifest: dict) -> str:
    """One line per schema violation, named by the tool it concerns."""
    path = list(error.absolute_path)
    where = ""
    if path[:1] == ["tools"] and len(path) >= 2:
        where = f"{manifest['tools'][path[1]].get('name', '?')}: "
    if error.validator == "pattern":
        return f"{where}name must be a snake_case verb phrase (search_series, get_observations)"
    if error.validator == "const":
        return f"{where}not marked read-only (readOnlyHint true)"
    if error.validator == "maxItems":
        return f"{len(error.instance)} tools; more than {MAX_TOOLS} makes the interface hard for a model to use"
    if len(path) >= 4 and path[2] == "inputs":
        return f"{where}input {path[3]!r} needs a type and a description"
    if error.validator == "required" and "'example'" in error.message:
        return f"{where}no example call"
    return f"{where}{error.message}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    m = json.loads(args.manifest.read_text(encoding="utf-8"))

    errors = [
        describe(e, m)
        for e in sorted(
            Draft202012Validator(SCHEMA).iter_errors(m),
            key=lambda e: list(e.absolute_path),
        )
    ]
    warnings: list[str] = []
    for t in m.get("tools", []):
        name = t.get("name", "?")
        words = len((t.get("description") or "").split())
        if words < MIN_WORDS:
            errors.append(
                f"{name}: description has {words} words; the model needs at least {MIN_WORDS}"
            )
        outputs = t.get("outputs") or []
        if "observations" in outputs:
            missing = [f for f in PROVENANCE if f not in outputs]
            if missing:
                errors.append(f"{name}: data tool lacks provenance fields {missing}")
        if name.startswith("search") and "limit" not in (t.get("inputs") or {}):
            warnings.append(
                f"{name}: a search tool without a limit input returns unbounded lists"
            )
    if not any(
        "how to use" in (r.get("description") or "").lower()
        for r in m.get("resources", [])
    ):
        warnings.append(
            "no guidance resource; a resource that tells the model how to use the tools reduces misuse"
        )

    print(
        f"{m.get('server', '?')} {m.get('version', '')}: {len(m.get('tools', []))} tool(s), {len(m.get('resources', []))} resource(s)"
    )
    for w in warnings:
        print(f"warning  {w}")
    for e in errors:
        print(f"error    {e}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    print(
        "Not checked here: whether the server behaves as the manifest says; the evaluation in chapter 5 does that."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
