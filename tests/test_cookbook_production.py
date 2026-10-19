"""Tests for the scripts shipped with the language-models-in-production cookbook."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = (
    REPO / "website" / "static" / "cookbook-files" / "language-models-in-production"
)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts():
    return {
        n: load_script(n)
        for n in [
            "map_components",
            "coding_eval",
            "check_suggestions",
            "commentary_check",
            "batch_cost",
            "check_statement",
        ]
    }


def test_map_flags_unreviewed_answers(scripts, capsys):
    assert scripts["map_components"].main([str(SCRIPTS / "gsbpm_map.csv")]) == 1
    out = capsys.readouterr().out
    assert "10 components across 5 phases" in out
    assert "Respondent chatbot: answer output reaches people with no review" in out
    assert "confidential data to a hosted model" not in out


def test_coding_eval_levels_and_thresholds(scripts, capsys):
    assert scripts["coding_eval"].main([str(SCRIPTS / "occupation_coding.csv")]) == 0
    out = capsys.readouterr().out
    assert "1-digit 0.85 (17/20)" in out and "4-digit 0.75 (15/20)" in out
    assert "0.80       0.70               1.00      0.30" in out
    assert "r14 'helps at home': 9999 -> 5322 (0.35)" in out


def test_suggestions_checked_against_rules(scripts, capsys):
    assert (
        scripts["check_suggestions"].main(
            [str(SCRIPTS / "edit_rules.csv"), str(SCRIPTS / "suggested_values.csv")]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "H0009-2   hours             45 -> 45     still fails: E04" in out
    assert "H0011-1   hours_main        50 -> 50     still fails: E05" in out
    assert "4 suggestions pass the rules and go to the editor; 2 still fail" in out


def test_commentary_check_flags_unverified_numbers_and_claims(scripts, capsys):
    assert (
        scripts["commentary_check"].main(
            [str(SCRIPTS / "release_table.csv"), str(SCRIPTS / "commentary_draft.md")]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "verified       4.31  LF_EMPLOYED value in millions" in out
    assert "verified     44,000  LF_EMPLOYED change in units" in out
    assert "UNVERIFIED   38,000" in out
    assert (
        "6 verified, 1 unverified, 1 claim(s) needing a source; provisional values are stated as provisional"
        in out
    )


def test_batch_cost(scripts, capsys):
    assert (
        scripts["batch_cost"].main(
            [
                "--records",
                "250000",
                "--tokens-in",
                "180",
                "--tokens-out",
                "12",
                "--hosted-in",
                "0.5",
                "--hosted-out",
                "2.0",
                "--batch-discount",
                "0.5",
                "--local-tokens-per-second",
                "900",
                "--local-hourly-cost",
                "1.2",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert (
        "hosted (batch discount 0.50): cost 14.25" in out
        and "local (900 tokens/s): 14.8 hours" in out
    )


def test_statement_check(scripts, tmp_path, capsys):
    assert (
        scripts["check_statement"].main([str(SCRIPTS / "model_use_statement.md")]) == 0
    )
    assert "result: PASS" in capsys.readouterr().out
    broken = tmp_path / "s.md"
    broken.write_text(
        (SCRIPTS / "model_use_statement.md")
        .read_text(encoding="utf-8")
        .replace("## Human oversight", "## Oversight"),
        encoding="utf-8",
    )
    assert scripts["check_statement"].main([str(broken)]) == 1
    assert "MISSING Human oversight" in capsys.readouterr().out
