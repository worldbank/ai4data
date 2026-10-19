"""Tests for the scripts shipped with the assessment-to-pathway cookbook."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "assessment-to-pathway"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts():
    names = [
        "readiness_profile",
        "gap_table",
        "sequence_plan",
        "effort_estimate",
        "pathway_document",
        "check_evidence",
        "compare_profiles",
    ]
    return {n: load_script(n) for n in names}


def test_profile_levels_and_overall(scripts, capsys):
    assert (
        scripts["readiness_profile"].main([str(SCRIPTS / "question_scores.csv")]) == 0
    )
    out = capsys.readouterr().out
    assert (
        "1.3   Skills and capacity                                5  0.40  A      3"
        in out
    )
    assert "overall: 25.6 of 100" in out
    assert "questions scored A: 24 of 69" in out


def test_profile_rejects_bad_score(scripts, capsys, tmp_path):
    text = (SCRIPTS / "question_scores.csv").read_text(encoding="utf-8")
    broken = tmp_path / "scores.csv"
    broken.write_text(
        text.replace(
            "1.1.1,1.1,1,Strategy and governance,AI strategy,1,",
            "1.1.1,1.1,1,Strategy and governance,AI strategy,7,",
        ),
        encoding="utf-8",
    )
    assert scripts["readiness_profile"].main([str(broken)]) == 1
    assert "1.1.1: score '7' is not 0 to 3" in capsys.readouterr().out


def test_gap_table_orders_by_gap(scripts, capsys):
    assert (
        scripts["gap_table"].main(
            [str(SCRIPTS / "question_scores.csv"), str(SCRIPTS / "targets.csv")]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.splitlines()[1].startswith("1.2     B (0.86)      C    1      6")
    assert "dimensions with a gap: 9 of 12; questions below target: 46" in out


def test_sequence_waves_and_foundations(scripts, capsys):
    assert (
        scripts["sequence_plan"].main(
            [
                str(SCRIPTS / "question_scores.csv"),
                str(SCRIPTS / "targets.csv"),
                str(SCRIPTS / "dependencies.csv"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "wave 3:\n  2.5" in out
    assert "after 2.1, 2.2, 2.4, 2.6, 1.1" in out
    assert "foundations in the first two waves or already at target: 5 of 5" in out


def test_sequence_detects_cycle(scripts, capsys, tmp_path):
    text = (SCRIPTS / "dependencies.csv").read_text(encoding="utf-8")
    broken = tmp_path / "dependencies.csv"
    broken.write_text(
        text.replace("2.1,,Metadata standards", "2.1,2.3,Metadata standards"),
        encoding="utf-8",
    )
    assert (
        scripts["sequence_plan"].main(
            [
                str(SCRIPTS / "question_scores.csv"),
                str(SCRIPTS / "targets.csv"),
                str(broken),
            ]
        )
        == 1
    )
    assert "dependency cycle" in capsys.readouterr().out


def test_effort_totals(scripts, capsys):
    assert (
        scripts["effort_estimate"].main(
            [
                str(SCRIPTS / "question_scores.csv"),
                str(SCRIPTS / "targets.csv"),
                str(SCRIPTS / "recipe_effort.csv"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "total: 9 recipes, 75 to 149 person-days" in out
    assert "of which outside the cookbook series: 1 items, 10 to 20 person-days" in out
    assert "kinds of cost beyond staff time: Gateway; Hosting; Legal fees" in out


def test_pathway_document_sections(scripts, capsys):
    assert (
        scripts["pathway_document"].main(
            [
                str(SCRIPTS / "question_scores.csv"),
                str(SCRIPTS / "targets.csv"),
                str(SCRIPTS / "dependencies.csv"),
                str(SCRIPTS / "recipe_effort.csv"),
                "--organization",
                "National Statistics Office (example)",
                "--assessed",
                "2025-06-20",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    for section in (
        "## Summary",
        "## Readiness profile",
        "## Targets and gaps",
        "## Sequence",
        "## Effort and roles by wave",
        "## Evidence and re-assessment",
        "## Governance of the pathway",
    ):
        assert section in out
    assert "### Wave 3" in out and "| 3 | 2.5 |" in out


def test_evidence_register_complete(scripts, capsys):
    assert (
        scripts["check_evidence"].main(
            [
                str(SCRIPTS / "question_scores.csv"),
                str(SCRIPTS / "targets.csv"),
                str(SCRIPTS / "evidence_register.csv"),
                "--as-of",
                "2026-01-15",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "questions below target: 46; with evidence rows: 46" in out
    assert "overdue on 2026-01-15: 0" in out


def test_evidence_register_overdue(scripts, capsys):
    assert (
        scripts["check_evidence"].main(
            [
                str(SCRIPTS / "question_scores.csv"),
                str(SCRIPTS / "targets.csv"),
                str(SCRIPTS / "evidence_register.csv"),
                "--as-of",
                "2026-04-01",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "overdue on 2026-04-01:" in out
    assert "1.1.1: AI strategy document" in out


def test_compare_profiles(scripts, capsys):
    assert (
        scripts["compare_profiles"].main(
            [
                str(SCRIPTS / "question_scores.csv"),
                str(SCRIPTS / "scores_after.csv"),
                str(SCRIPTS / "targets.csv"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "targets reached: 12 of 12; questions moved up: 51, down: 0" in out
    assert "after: pillar 1 1.53, pillar 2 1.78, overall 55.3 of 100" in out
