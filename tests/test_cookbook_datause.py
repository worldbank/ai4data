"""Tests for the scripts shipped with the monitoring-data-use cookbook.

The scripts live under website/static/cookbook-files/monitoring-data-use
and are loaded by path. They use the standard library only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "monitoring-data-use"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harmonize():
    return load_script("harmonize_mentions")


@pytest.fixture(scope="module")
def score():
    return load_script("score_mentions")


@pytest.fixture(scope="module")
def report():
    return load_script("use_report")


def test_harmonization_resolves_variants_and_leaves_external_data(
    harmonize, tmp_path, capsys
):
    out = tmp_path / "harmonized.jsonl"
    assert (
        harmonize.main(
            [
                str(SCRIPTS / "mentions_example.jsonl"),
                str(SCRIPTS / "canonical_names.csv"),
                "-o",
                str(out),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "12 mentions: exact 8, contains 2, fuzzy 0, none 2" in text
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    by_text = {r["text"]: r for r in rows}
    assert (
        by_text["HBS"]["dataset_id"] == "EX-HBS-2021"
        and by_text["HBS"]["match_type"] == "exact"
    )
    assert by_text["the labour survey"]["dataset_id"] == "EX-LFS"
    assert by_text["unemployment rate from the LFS"]["dataset_id"] == "LF_UNEMP_PCT"
    assert by_text["Demographic and Health Survey 2016"]["dataset_id"] is None
    assert by_text["World Development Indicators"]["match_type"] == "none"


def test_match_function_types(harmonize):
    canonical = harmonize.load_canonical(SCRIPTS / "canonical_names.csv")
    assert harmonize.match("Labour Force Survey", canonical, 0.8) == (
        "EX-LFS",
        "exact",
        1.0,
    )
    dataset, kind, _ = harmonize.match("Labor Force Surveys", canonical, 0.8)
    assert (dataset, kind) == ("EX-LFS", "fuzzy")
    assert harmonize.match("some unrelated phrase", canonical, 0.8) == (
        None,
        "none",
        0.0,
    )


def test_scoring_against_the_labelled_sample(score, capsys):
    assert (
        score.main(
            [
                str(SCRIPTS / "mentions_example.jsonl"),
                str(SCRIPTS / "labelled_sample.csv"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "labelled mentions 12, extracted 12, correct 11" in out
    assert "precision 0.92  recall 0.92  F1 0.92" in out
    assert "D06: 'food security report'" in out
    assert "D04: 'population and housing census 2014'" in out


def test_use_report_counts_documents_not_mentions(harmonize, report, tmp_path, capsys):
    harmonized = tmp_path / "harmonized.jsonl"
    harmonize.main(
        [
            str(SCRIPTS / "mentions_example.jsonl"),
            str(SCRIPTS / "canonical_names.csv"),
            "-o",
            str(harmonized),
        ]
    )
    capsys.readouterr()
    assert report.main([str(harmonized), str(SCRIPTS / "documents.csv")]) == 0
    out = capsys.readouterr().out
    assert "6 documents, 12 mentions, 5 datasets of the organization mentioned" in out
    line = next(line for line in out.splitlines() if line.startswith("EX-HBS-2021"))
    assert line.split()[1:3] == [
        "2",
        "2",
    ]  # two documents mention it, two use it (two mentions in D01 count once)
    line = next(line for line in out.splitlines() if line.startswith("EX-REP-2025-03"))
    assert line.split()[1:3] == ["2", "0"]  # mentioned in D02 and D06, used in none
    assert "'World Development Indicators': 1 document(s)" in out


@pytest.fixture(scope="module")
def access():
    return load_script("count_access")


@pytest.fixture(scope="module")
def citations():
    return load_script("merge_citations")


@pytest.fixture(scope="module")
def robots():
    return load_script("robots_check")


def test_access_count_excludes_robots_and_double_clicks(access, capsys):
    assert access.main([str(SCRIPTS / "access_log.csv")]) == 0
    out = capsys.readouterr().out
    assert "30 log rows; excluded: robots 4, double-clicks 6, failed requests 1" in out
    hbs = next(
        line
        for line in out.splitlines()
        if line.startswith("EX-HBS-2021            2026-09")
    )
    assert hbs.split()[2:] == ["4", "3", "1", "1", "0", "4"]
    fs = next(
        line
        for line in out.splitlines()
        if line.startswith("FS_UNDERNOURISH_PCT    2026-09")
    )
    assert fs.split()[2:] == ["2", "1", "1", "3", "3", "3"]


def test_citation_merge_dedupes_across_sources(citations, capsys):
    assert citations.main([str(SCRIPTS / "citation_events.csv")]) == 0
    out = capsys.readouterr().out
    assert "12 events -> 7 unique citing documents across 4 datasets" in out
    assert (
        "found through identifiers 5, through text mining 4, by both 2, by text mining only 2"
        in out
    )
    assert (
        "National nutrition strategy 2025 to 2030 (2025) found by text-mining only"
        in out
    )


def test_robots_check_blocks_disallowed_urls(robots, capsys):
    rc = robots.main(
        [
            str(SCRIPTS / "robots_example.txt"),
            str(SCRIPTS / "web_sources.csv"),
            "--agent",
            "ai4data-monitor",
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "crawl delay 5" in out and "5 URLs, 3 disallowed" in out
    assert (
        "allowed   https://ministry.example.gov/publications/strategy-2025.pdf" in out
    )
    assert "DISALLOWED  https://ministry.example.gov/search?q=nutrition" in out
    assert (
        robots.main(
            [
                str(SCRIPTS / "robots_example.txt"),
                str(SCRIPTS / "web_sources.csv"),
                "--agent",
                "otherbot",
            ]
        )
        == 1
    )
