"""Tests for the scripts shipped with the evaluation-suites cookbook."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "evaluation-suites"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts():
    names = [
        "label_agreement",
        "retrieval_by_slice",
        "answer_scores",
        "extraction_scores",
        "judge_agreement",
        "compare_runs",
        "sample_traffic",
        "report_card",
    ]
    return {n: load_script(n) for n in names}


def test_label_agreement(scripts, capsys):
    assert (
        scripts["label_agreement"].main([str(SCRIPTS / "labels_two_annotators.csv")])
        == 0
    )
    out = capsys.readouterr().out
    assert "20 items: agreement 0.85 (17/20), Cohen's kappa 0.82" in out
    assert "3 of 3 disagreements involve NONE" in out


def test_retrieval_by_slice(scripts, capsys):
    assert (
        scripts["retrieval_by_slice"].main([str(SCRIPTS / "retrieval_runs.csv")]) == 0
    )
    out = capsys.readouterr().out
    assert "all              v1    10   0.60   0.80   0.70" in out
    assert "all              v2    10   0.90   1.00   0.95" in out
    assert "language=fr      v1     2   0.00   0.50   0.25" in out


def test_answer_scores(scripts, capsys):
    assert scripts["answer_scores"].main([str(SCRIPTS / "answers_run.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "a03: expected [41.0], answer gave [35.0, 2020.0]" in out
    assert (
        "a06: should have declined" in out
        and "a07: cited EDU_STATS_2024, which was not retrieved" in out
    )
    assert "number 8 is not" not in out and "number 1000 is not" not in out
    assert scripts["answer_scores"].numbers(
        "7,4 % en 2025 (released 2025-08-15), per 1,000 births, 1 234,5"
    ) == [7.4, 2025.0, 1234.5]


def test_extraction_scores(scripts, capsys):
    assert (
        scripts["extraction_scores"].main([str(SCRIPTS / "extraction_gold_pred.csv")])
        == 0
    )
    out = capsys.readouterr().out
    assert "exact span and type            precision 0.50  recall 0.50  F1 0.50" in out
    assert "overlapping span, same type    precision 0.70  recall 0.70  F1 0.70" in out
    assert (
        "error taxonomy: correct 5, boundary 2, missed 2, spurious 2, wrong type 1"
        in out
    )


def test_judge_agreement_flags_length_bias(scripts, capsys):
    assert scripts["judge_agreement"].main([str(SCRIPTS / "judge_vs_human.csv")]) == 0
    out = capsys.readouterr().out
    assert (
        "agreement 0.75, kappa 0.44; grader passes what humans fail 4, fails what humans pass 1"
        in out
    )
    assert "length bias" in out


def test_compare_runs_sends_small_regressions_to_review(scripts, capsys):
    rc = scripts["compare_runs"].main(
        [
            str(SCRIPTS / "run_scores_v1.csv"),
            str(SCRIPTS / "run_scores_v2.csv"),
            "--tolerance",
            "0.05",
        ]
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "gate: REVIEW" in out and "language=es" in out
    assert (
        scripts["compare_runs"].main(
            [str(SCRIPTS / "run_scores_v2.csv"), str(SCRIPTS / "run_scores_v2.csv")]
        )
        == 0
    )
    assert "gate: PASS" in capsys.readouterr().out


def test_sample_traffic_includes_failures(scripts, tmp_path, capsys):
    out_path = tmp_path / "sample.csv"
    assert (
        scripts["sample_traffic"].main(
            [
                str(SCRIPTS / "traffic_log.csv"),
                "--size",
                "12",
                "--min-per-stratum",
                "1",
                "-o",
                str(out_path),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert (
        "24 logged queries -> sample of 16: 6 declined or rated down (always included), 10 drawn by stratum"
        in out
    )
    rows = out_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 17 and any("t009" in r for r in rows)


def test_report_card(scripts, capsys):
    assert (
        scripts["report_card"].main(
            [str(SCRIPTS / "run_scores_v2.csv"), "--run-name", "v2"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "all                 30      0.73   [0.57, 0.87]" in out
