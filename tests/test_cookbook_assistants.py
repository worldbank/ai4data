"""Tests for the scripts shipped with the ai-assistants-at-work cookbook."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "ai-assistants-at-work"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts():
    names = [
        "task_card",
        "screen_input",
        "build_prompt",
        "library_check",
        "check_numbers",
        "use_summary",
    ]
    return {n: load_script(n) for n in names}


def test_task_card_for_a_role(scripts, capsys):
    assert (
        scripts["task_card"].main(
            [str(SCRIPTS / "task_card.csv"), "--role", "Data curator"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Task card: Data curator" in out
    assert (
        "T07  Propose a variable label from a questionnaire text (draft; local assistant only)"
        in out
    )
    assert (
        "16 tasks for 6 roles and all staff; 14 allowed, 2 stay with the person" in out
    )


def test_task_card_rejects_missing_review(scripts, capsys, tmp_path):
    text = (SCRIPTS / "task_card.csv").read_text(encoding="utf-8")
    broken = tmp_path / "tasks.csv"
    broken.write_text(
        text.replace("The curator approves the draft before it enters the record", ""),
        encoding="utf-8",
    )
    assert scripts["task_card"].main([str(broken), "--role", "Data curator"]) == 1
    assert "T06: allowed task without a review" in capsys.readouterr().out


def test_screen_input_finds_markers(scripts, capsys):
    assert scripts["screen_input"].main([str(SCRIPTS / "draft_note.txt")]) == 1
    out = capsys.readouterr().out
    assert "national identity number: 4471-20-88193" in out
    assert "embargo: EMBARGOED" in out
    assert "tier: confidential" in out


def test_screen_input_internal_after_removal(scripts, capsys, tmp_path):
    lines = (SCRIPTS / "draft_note.txt").read_text(encoding="utf-8").splitlines()
    kept = [
        line
        for line in lines
        if "4471" not in line
        and "0147" not in line
        and "@" not in line
        and "Household" not in line
        and "income" not in line
    ]
    cleaned = tmp_path / "note.txt"
    cleaned.write_text("\n".join(kept), encoding="utf-8")
    assert scripts["screen_input"].main([str(cleaned)]) == 0
    assert "tier: internal" in capsys.readouterr().out


def test_build_prompt_fills_table(scripts, capsys):
    assert (
        scripts["build_prompt"].main(
            [
                str(SCRIPTS / "prompt_library.yaml"),
                "--prompt",
                "summarize_release",
                "--var",
                f"table={SCRIPTS / 'release_table.csv'}",
                "--var",
                "audience=the Friday briefing",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "prompt: summarize_release (data tier: internal)" in out
    assert "LF_UNEMP_PCT,2025-Q2,6.1,6.3" in out
    assert "for the Friday briefing" in out


def test_build_prompt_missing_placeholder(scripts, capsys):
    assert (
        scripts["build_prompt"].main(
            [str(SCRIPTS / "prompt_library.yaml"), "--prompt", "summarize_release"]
        )
        == 1
    )
    assert "placeholder(s) not supplied: table, audience" in capsys.readouterr().out


def test_library_check_passes_and_fails(scripts, capsys, tmp_path):
    assert scripts["library_check"].main([str(SCRIPTS / "prompt_library.yaml")]) == 0
    assert (
        "5 prompts; tiers: 1 confidential, 3 internal, 1 public"
        in capsys.readouterr().out
    )
    text = (SCRIPTS / "prompt_library.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "library.yaml"
    broken.write_text(
        text.replace(
            "placeholders: [table, audience]",
            "placeholders: [table, audience, references]",
        ),
        encoding="utf-8",
    )
    assert scripts["library_check"].main([str(broken)]) == 1
    assert (
        "placeholder 'references' listed and not in the template"
        in capsys.readouterr().out
    )


def test_check_numbers_finds_planted_errors(scripts, capsys):
    assert (
        scripts["check_numbers"].main(
            [str(SCRIPTS / "assistant_answer.md"), str(SCRIPTS / "release_table.csv")]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "8 numbers; 6 verified, 2 unverified" in out
    assert "problem: 4,321 is not a value" in out
    assert "problem: 0.3 is not a value" in out


def test_use_summary_flags_breaches(scripts, capsys):
    assert (
        scripts["use_summary"].main(
            [str(SCRIPTS / "use_log.csv"), str(SCRIPTS / "task_card.csv")]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "20 uses, 6 units, 2025-11-03 to 2025-11-14" in out
    assert "reviewed by a named person: 19 of 20" in out
    assert (
        "2025-11-10 Labour statistics T02: confidential text at a hosted assistant"
        in out
    )
