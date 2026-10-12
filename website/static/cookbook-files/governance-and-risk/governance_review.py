"""Run every governance check on the folder and map the results to the assessment questions.

Runs the policy, register, risk, decision log, licence, and statement
checks on the files in a folder (the file names of this cookbook) and
prints one line per artefact with its result and the AI-readiness
assessment questions it is evidence for. Exit status 1 when any check
fails or a file is missing.

Usage:
    python governance_review.py . --as-of 2026-01-15 --release EX-REP-2025-02

Requires pandas (through the checks it runs).

What this does not do: it does not read the artefacts for meaning. A
policy with seven sections and nothing in them would fail the item
counts and pass little else; the review meeting reads the text.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

ITEMS = [
    (
        "AI-use policy",
        "ai_use_policy.md",
        "check_policy",
        ["{p}/ai_use_policy.md"],
        "1.1.1, 1.1.4, 1.2.7",
    ),
    (
        "Component register",
        "component_register.csv",
        "check_register",
        ["{p}/component_register.csv", "--as-of", "{as_of}"],
        "1.1.2, 1.1.3, 1.2.7",
    ),
    (
        "Risk register",
        "risk_register.csv",
        "risk_score",
        [
            "{p}/risk_register.csv",
            "{p}/controls_by_tier.csv",
            "--register",
            "{p}/component_register.csv",
        ],
        "1.1.3",
    ),
    (
        "Decision log",
        "decision_log.csv",
        "check_decision_log",
        ["{p}/decision_log.csv", "{p}/component_register.csv", "--as-of", "{as_of}"],
        "1.1.2, 1.2.7",
    ),
    (
        "Licence matrix and model licences",
        "licence_matrix.csv",
        "check_licences",
        ["{p}/licence_matrix.csv", "--models", "{p}/model_licences.csv"],
        "1.2.5, 2.6.1, 2.6.2, 2.6.3, 2.6.4",
    ),
    (
        "Statement of model use",
        "evaluation_summary.csv",
        "statement_of_model_use",
        [
            "{p}/component_register.csv",
            "{p}/evaluation_summary.csv",
            "--release",
            "{release}",
            "--check",
        ],
        "1.1.4, 1.2.7",
    ),
]
DOCUMENTS = [
    ("Board terms of reference", "board_terms.md", "1.1.2"),
    ("Incident report", "incident_report.md", "1.1.3, 1.2.7"),
    ("Provider register", "providers.csv", "1.2.5, 1.5.4"),
    ("Cost record", "cost_record.csv", "1.5.4"),
    ("Terms of use for machine clients", "terms_machine_clients.md", "2.6.4"),
]


def run(folder: Path, module: str, argv: list[str]) -> int:
    spec = importlib.util.spec_from_file_location(module, folder / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--as-of", required=True, help="date (YYYY-MM-DD) of the review"
    )
    parser.add_argument(
        "--release", required=True, help="release whose statement is checked"
    )
    args = parser.parse_args(argv)
    folder = args.folder
    failed = 0
    print(f"governance review as of {args.as_of}")
    print(f"{'artefact':<34} {'result':<8} evidence for questions")
    for name, file, module, call, questions in ITEMS:
        if not (folder / file).exists():
            print(f"{name:<34} {'MISSING':<8} {questions}")
            failed += 1
            continue
        call = [
            a.format(p=str(folder), as_of=args.as_of, release=args.release)
            for a in call
        ]
        rc = run(folder, module, call)
        print(f"{name:<34} {'PASS' if rc == 0 else 'FAIL':<8} {questions}")
        failed += rc != 0
    for name, file, questions in DOCUMENTS:
        present = (folder / file).exists()
        print(f"{name:<34} {'present' if present else 'MISSING':<8} {questions}")
        failed += not present
    print(
        f"\n{len(ITEMS) + len(DOCUMENTS) - failed} of {len(ITEMS) + len(DOCUMENTS)} items in order"
    )
    print("result: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
