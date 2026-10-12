"""Build a prompt from the prompt library with the values a person supplies.

Reads the prompt library (YAML) and prints one prompt with its
placeholders filled. A value that names a file that exists is replaced
by the file's contents, so that a table or a script can be passed by
path. The output starts with a header that names the prompt, its data
tier, and its review, so that the person sees the rule before the
text. Exit status 1 when the prompt does not exist or a placeholder is
not supplied.

Usage:
    python build_prompt.py prompt_library.yaml --prompt summarize_release --var table=release_table.csv --var "audience=the Friday briefing"

Requires PyYAML.

What this does not do: it does not send the prompt anywhere, and it does
not screen the values for confidential content; screen_input.py does
that, and the data tier in the header says which assistant may receive
the prompt.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("library", type=Path)
    parser.add_argument("--prompt", required=True, help="the prompt id")
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a placeholder value; a path to an existing file is read",
    )
    args = parser.parse_args(argv)
    lib = yaml.safe_load(args.library.read_text(encoding="utf-8"))
    prompts = {p["id"]: p for p in lib.get("prompts", [])}
    if args.prompt not in prompts:
        print(f"problem: no prompt {args.prompt!r}; prompts are {', '.join(prompts)}")
        return 1
    p = prompts[args.prompt]
    values = {}
    for v in args.var:
        name, _, value = v.partition("=")
        path = Path(value)
        values[name] = (
            path.read_text(encoding="utf-8").rstrip() if path.is_file() else value
        )
    missing = [ph for ph in p.get("placeholders", []) if ph not in values]
    if missing:
        print(f"problem: placeholder(s) not supplied: {', '.join(missing)}")
        return 1
    text = re.sub(
        r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), p["template"]
    )
    print(f"prompt: {p['id']} (data tier: {p['data_tier']})")
    print(f"review: {p['review']}")
    print("-" * 60)
    print(text.rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
