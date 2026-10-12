"""Tests for the scripts shipped with the governance-and-risk cookbook."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "governance-and-risk"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts():
    names = [
        "check_policy",
        "check_register",
        "risk_score",
        "check_decision_log",
        "cost_summary",
        "check_licences",
        "statement_of_model_use",
        "governance_review",
    ]
    return {n: load_script(n) for n in names}


def test_policy_passes(scripts, capsys):
    assert scripts["check_policy"].main([str(SCRIPTS / "ai_use_policy.md")]) == 0
    out = capsys.readouterr().out
    assert "permitted uses: 7 item(s)" in out
    assert "approver: Director General" in out
    assert "result: PASS" in out


def test_policy_fails_on_placeholders(scripts, capsys, tmp_path):
    text = (SCRIPTS / "ai_use_policy.md").read_text(encoding="utf-8")
    broken = tmp_path / "policy.md"
    broken.write_text(text.replace("Director General", "[role]", 1), encoding="utf-8")
    assert scripts["check_policy"].main([str(broken)]) == 1
    out = capsys.readouterr().out
    assert "approver not stated" in out


def test_register_lists_due_reviews(scripts, capsys):
    assert (
        scripts["check_register"].main(
            [str(SCRIPTS / "component_register.csv"), "--as-of", "2026-01-15"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "6 components; 5 Internal, 1 Provider A; 2 Low, 1 Medium, 3 High" in out
    assert "C03 Occupation coding: due 2026-01-06" in out


def test_register_rejects_confidential_transfer(scripts, capsys, tmp_path):
    text = (SCRIPTS / "component_register.csv").read_text(encoding="utf-8")
    broken = tmp_path / "register.csv"
    broken.write_text(
        text.replace(
            "Confidential,Coders decide the routed share; a 5 percent sample is re-coded,No",
            "Confidential,Coders decide the routed share; a 5 percent sample is re-coded,Yes",
        ),
        encoding="utf-8",
    )
    assert scripts["check_register"].main([str(broken)]) == 1
    assert "C03: confidential data leave the organization" in capsys.readouterr().out


def test_risk_scores_and_tiers(scripts, capsys):
    assert (
        scripts["risk_score"].main(
            [
                str(SCRIPTS / "risk_register.csv"),
                str(SCRIPTS / "controls_by_tier.csv"),
                "--register",
                str(SCRIPTS / "component_register.csv"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "C05  Release commentary drafts         3    3    2    18  High" in out
    assert "C04  Edit failure explanations         2    2    2     8  Medium" in out
    assert "High (C03, C05, C06):" in out


def test_decision_log_overdue_follow_ups(scripts, capsys):
    assert (
        scripts["check_decision_log"].main(
            [
                str(SCRIPTS / "decision_log.csv"),
                str(SCRIPTS / "component_register.csv"),
                "--as-of",
                "2026-01-15",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "10 decisions: 1 policy, 6 approval, 2 incident, 1 review" in out
    assert "incidents: 1 opened, 1 closed" in out
    assert "D004 (C03)" in out and "OVERDUE" in out


def test_cost_summary(scripts, capsys):
    assert (
        scripts["cost_summary"].main(
            [str(SCRIPTS / "cost_record.csv"), str(SCRIPTS / "providers.csv")]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "2025-12: 31 of 242 (13%)" in out
    assert "C05: +244%" in out


def test_licences_find_planted_defects(scripts, capsys):
    assert (
        scripts["check_licences"].main(
            [
                str(SCRIPTS / "licence_matrix.csv"),
                "--models",
                str(SCRIPTS / "model_licences.csv"),
            ]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "EX-YB-2023: no licence" in out
    assert "EX-REP-2025-03: licence is not machine-readable" in out
    assert "4 models; classes: 2 permissive, 1 proprietary-api, 1 restricted" in out


def test_statement_generates_and_checks(scripts, capsys):
    assert (
        scripts["statement_of_model_use"].main(
            [
                str(SCRIPTS / "component_register.csv"),
                str(SCRIPTS / "evaluation_summary.csv"),
                "--release",
                "EX-REP-2025-02",
                "--check",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "- Occupation coding (GSBPM 5.2): propose ISCO-08 codes" in out
    assert "accuracy at four digits on the automatically coded share 0.93" in out
    assert "check: PASS" in out


def test_governance_review_counts_items(scripts, capsys):
    assert (
        scripts["governance_review"].main(
            [str(SCRIPTS), "--as-of", "2026-01-15", "--release", "EX-REP-2025-02"]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "Licence matrix and model licences  FAIL" in out
    assert "10 of 11 items in order" in out
