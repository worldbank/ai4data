"""Check a server listing (server.json) before publishing it to a registry.

Reads a server.json in the MCP registry format and reports the required
fields (a reverse-DNS name such as org.example.stats/server, a semantic
version), the fields a client needs to connect (a remote with transport
streamable-http or sse and an https URL, or a package with a registry type
and a transport), and the fields a person needs to trust the listing
(title, description, website, repository). Exit status 1 when a required
or connection field is missing. Uses jsonschema for the listing's shape.

Usage:
    python check_server_listing.py server_listing.json

What this does not do: it checks the listing's shape against the format
documented by the registry at the time of writing; the registry validates
against its own schema (the ``$schema`` URL) on submission, and the
version of that schema changes. Whether the endpoint answers is the
operation chapter's check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

# The parts of the registry's server.json format that a listing needs; the
# registry's own dated schema (named in $schema) is the full definition.
SCHEMA = {
    "type": "object",
    "required": ["name", "version"],
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[a-z0-9][a-z0-9.-]*\\.[a-z0-9.-]+/[A-Za-z0-9._-]+$",
        },
        "version": {
            "type": "string",
            "pattern": "^\\d+\\.\\d+\\.\\d+(?:[-+][0-9A-Za-z.-]+)?$",
        },
        "remotes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "url"],
                "properties": {
                    "type": {"enum": ["streamable-http", "sse"]},
                    "url": {"type": "string", "pattern": "^https://"},
                },
            },
        },
        "packages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["registryType", "identifier", "version", "transport"],
            },
        },
    },
    "anyOf": [
        {"required": ["remotes"], "properties": {"remotes": {"minItems": 1}}},
        {"required": ["packages"], "properties": {"packages": {"minItems": 1}}},
    ],
}
MESSAGES = {
    (
        "name",
        "pattern",
    ): "name must be reverse-DNS namespace plus server name, for example org.example.stats/statistics-example",
    ("version", "pattern"): "version must be a semantic version such as 1.0.0",
}


def describe(error) -> str:
    path = list(error.absolute_path)
    key = (path[0] if path else "", error.validator)
    if key in MESSAGES:
        return MESSAGES[key]
    if error.validator == "anyOf":
        return "a listing needs at least one remote (hosted endpoint) or one package (installable server)"
    if path[:1] == ["remotes"] and error.validator == "enum":
        return f"remotes[{path[1]}].type must be one of ['sse', 'streamable-http']"
    if path[:1] == ["remotes"] and error.validator == "pattern":
        return f"remotes[{path[1]}].url must be an https URL"
    if path[:1] == ["packages"] and error.validator == "required":
        return f"packages[{path[1]}] lacks {error.message.split()[0].strip(chr(39))}"
    return ".".join(map(str, path)) + ": " + error.message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("listing", type=Path)
    args = parser.parse_args(argv)
    s = json.loads(args.listing.read_text(encoding="utf-8"))

    errors = [
        describe(e)
        for e in sorted(
            Draft202012Validator(SCHEMA).iter_errors(s),
            key=lambda e: list(e.absolute_path),
        )
    ]
    warnings = [
        f"{key} missing; clients show it to users"
        for key in ("title", "description", "websiteUrl")
        if not s.get(key)
    ]
    if not (s.get("repository") or {}).get("url"):
        warnings.append(
            "repository.url missing; readers cannot inspect the server's code"
        )
    if "$schema" not in s:
        warnings.append(
            "$schema missing; the registry validates against a dated schema"
        )
    if len(s.get("description", "")) > 300:
        warnings.append(
            f"description is {len(s['description'])} characters; keep it under 300"
        )

    remotes, packages = s.get("remotes", []), s.get("packages", [])
    print(
        f"{args.listing.name}: {s.get('name')} {s.get('version')}; {len(remotes)} remote(s), {len(packages)} package(s)"
    )
    for e in errors:
        print(f"error    {e}")
    for w in warnings:
        print(f"warning  {w}")
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
