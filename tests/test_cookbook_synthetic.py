"""Tests for the scripts shipped with the synthetic-data-for-sharing cookbook."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "synthetic-data-for-sharing"


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
            "utility_report",
            "risk_report",
            "relational_check",
            "check_synthesis_record",
        ]
    }


def test_utility_report_shows_lost_association(scripts, capsys):
    assert (
        scripts["utility_report"].main(
            [str(SCRIPTS / "real_sample.csv"), str(SCRIPTS / "synthetic_sample.csv")]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "real 300 rows, synthetic 300 rows, 6 columns" in out
    assert "sex x lfs_status   Cramer's V 0.13 vs 0.05" in out
    assert "employment rate, sex 1, ages 15-64: real      0.556" in out
    assert scripts["utility_report"].tvd(
        ["a", "a", "b"], ["a", "b", "b"]
    ) == pytest.approx(1 / 3)


def test_risk_report_finds_exact_copies(scripts, capsys):
    rc = scripts["risk_report"].main(
        [
            str(SCRIPTS / "real_sample.csv"),
            str(SCRIPTS / "synthetic_sample.csv"),
            "--quasi",
            "region",
            "sex",
            "age",
            "educ",
            "--sensitive",
            "income",
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "exact copies: 15 synthetic record(s) identical to a real record" in out
    assert "nearest-neighbour accuracy 0.42 vs majority baseline 0.56" in out


def test_relational_check_reports_faults_and_distributions(scripts, capsys):
    rc = scripts["relational_check"].main(
        [
            str(SCRIPTS / "households_synth.csv"),
            str(SCRIPTS / "persons_synth.csv"),
            "--real-households",
            str(SCRIPTS / "households_real.csv"),
            "--real-persons",
            str(SCRIPTS / "persons_real.csv"),
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "3 fault(s):" in out and "person H999/1 belongs to no household" in out
    assert (
        "household H011 size 3 but 2 persons" in out
        and "child 3 aged 61 is not younger than the head (56)" in out
    )
    assert (
        scripts["relational_check"].main(
            [str(SCRIPTS / "households_real.csv"), str(SCRIPTS / "persons_real.csv")]
        )
        == 0
    )
    assert "no integrity or structure faults" in capsys.readouterr().out


def test_synthesis_record_check(scripts, tmp_path, capsys):
    assert (
        scripts["check_synthesis_record"].main([str(SCRIPTS / "synthesis_record.md")])
        == 0
    )
    assert "result: PASS" in capsys.readouterr().out
    broken = tmp_path / "r.md"
    broken.write_text(
        (SCRIPTS / "synthesis_record.md")
        .read_text(encoding="utf-8")
        .replace("Seed 2025.", "")
        .replace("## Contact", "## Questions"),
        encoding="utf-8",
    )
    assert scripts["check_synthesis_record"].main([str(broken)]) == 1
    out = capsys.readouterr().out
    assert "MISSING Contact" in out and "MISSING seed stated in the method" in out
