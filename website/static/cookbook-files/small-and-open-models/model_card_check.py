"""Check a model card for what an organization needs before it deploys.

Reads a model card (JSON) and reports whether it states: the identifier,
version, and a hash of the weights; a licence with a URL; the languages;
the intended and out-of-scope uses; the training data (what, from where,
and whether confidential data were used); an evaluation with a suite
name, a date, and at least one number; known limitations; hardware
requirements; a contact. Exit status 1 when any is missing. Uses jsonschema for the card's shape.

Usage:
    python model_card_check.py model_card_example.json

What this does not do: it checks that the card says these things, not
that they are true. The evaluation numbers are re-produced on the
organization's own suite before deployment, and the licence is read by
someone who can say what it permits.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

REQUIRED = [
    "model_id",
    "version",
    "weights_sha256",
    "license",
    "license_url",
    "languages",
    "intended_use",
    "out_of_scope_use",
    "training_data",
    "evaluation",
    "known_limitations",
    "hardware",
    "contact",
]
NON_EMPTY = {
    "anyOf": [
        {"type": "string", "minLength": 1},
        {"type": "array", "minItems": 1},
        {"type": "object", "minProperties": 1},
        {"type": "number"},
    ]
}
SCHEMA = {
    "type": "object",
    "required": REQUIRED,
    "properties": {k: NON_EMPTY for k in REQUIRED}
    | {"weights_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
}
DATA_PROVENANCE = re.compile(
    r"\b(no external|left the organization|public|licensed|consent)\b", re.IGNORECASE
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("card", type=Path)
    args = parser.parse_args(argv)
    card = json.loads(args.card.read_text(encoding="utf-8"))

    missing: set[str] = set()
    problems: list[str] = []
    for e in Draft202012Validator(SCHEMA).iter_errors(card):
        if e.validator == "required":
            missing.update(k for k in REQUIRED if f"'{k}'" in e.message)
        elif e.validator == "pattern":
            problems.append("weights_sha256 is not a 64-character hex digest")
        else:
            missing.add(str(e.absolute_path[0]))
    ev = card.get("evaluation") or {}
    if ev and not (
        ev.get("suite")
        and ev.get("date")
        and any(isinstance(v, (int, float)) for v in ev.values())
    ):
        problems.append("evaluation lacks a suite name, a date, or a number")
    if card.get("training_data") and not DATA_PROVENANCE.search(card["training_data"]):
        problems.append(
            "training_data does not say whether confidential or external data were used"
        )

    print(
        f"{card.get('model_id', '?')} {card.get('version', '?')}: licence {card.get('license', '?')}, languages {', '.join(card.get('languages', []))}"
    )
    for key in REQUIRED:
        print(f"  {'MISSING' if key in missing else 'ok     '} {key}")
    for p in problems:
        print(f"  problem {p}")
    failed = bool(missing or problems)
    print("result: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
