"""Tests for the scripts shipped with the metadata-curation-with-llms cookbook."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "metadata-curation-with-llms"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def score():
    return load_script("score_suggestions")


@pytest.fixture(scope="module")
def agreement():
    return load_script("assess_agreement")


@pytest.fixture(scope="module")
def vocab():
    return load_script("map_vocabulary")


def test_acceptance_rates_per_field_and_source(score, capsys):
    assert (
        score.main(
            [
                str(SCRIPTS / "suggestions_example.jsonl"),
                str(SCRIPTS / "decisions_example.csv"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "10 suggestions, 10 decided, 0 undecided" in out
    assert "measurement_unit" in out and "rejected, with reasons:" in out
    assert "HE_U5_MORT_RATE/measurement_unit (reviewer)" in out
    field_lines = out.split("by field")[1].split("by source")[0].splitlines()
    keywords = next(line for line in field_lines if line.startswith("keywords"))
    assert keywords.split()[-1] == "1.00"


def test_agreement_flags_the_lenient_dimension(agreement, capsys):
    assert (
        agreement.main(
            [str(SCRIPTS / "model_scores.csv"), str(SCRIPTS / "curator_scores.csv")]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "5 records, 4 dimensions" in out
    consistency = next(
        line for line in out.splitlines() if line.startswith("consistency")
    )
    assert "model more lenient" in consistency
    assert "1 dimension(s) miscalibrated" in out


def test_vocabulary_mapping_counts(vocab, capsys):
    assert (
        vocab.main(
            [str(SCRIPTS / "keywords_freetext.csv"), str(SCRIPTS / "vocabulary.csv")]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "14 keywords: exact 10, fuzzy 2, none 2" in out
    assert "FS_UNDERNOURISH_PCT: 'hunger'" in out
    assert "LF_UNEMP_PCT: 'jobless'" in out
    assert "'unemployment rates'" in out and "-> Unemployment" in out


@pytest.fixture(scope="module")
def migrate():
    return load_script("migrate_records")


@pytest.fixture(scope="module")
def translations():
    return load_script("check_translations")


def test_migration_applies_transforms_and_reports_gaps(migrate, tmp_path, capsys):
    out_path = tmp_path / "records.json"
    assert (
        migrate.main(
            [
                str(SCRIPTS / "legacy_catalog.csv"),
                str(SCRIPTS / "field_mapping.csv"),
                "-o",
                str(out_path),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "5 legacy rows -> 5 records; 11 mapping rows: 8 confirmed, 3 proposed" in out
    assert (
        "'Notes': 2 non-empty value(s)" in out
        and "empty after migration: definition_long (1)" in out
    )
    assert "PV_HEADCOUNT_NPL_PCT/Frequency: unknown code 'Other'" in out
    records = {
        r["idno"]: r
        for r in __import__("json").loads(out_path.read_text(encoding="utf-8"))
    }
    assert records["FS_UNDERNOURISH_PCT"]["periodicity"] == "annual"
    assert records["FS_UNDERNOURISH_PCT"]["date_last_update"] == "2025-03-14"
    assert records["FS_UNDERNOURISH_PCT"]["sources"] == [
        "Food balance sheets",
        "household budget survey",
    ]
    assert (
        "definition_long" not in records["LF_UNEMP_PCT"]
        and records["PV_HEADCOUNT_NPL_PCT"]["periodicity"] == "Other"
    )


def test_translation_check_holds_back_changed_numbers_and_short_text(
    translations, capsys
):
    assert translations.main([str(SCRIPTS / "translations_example.csv")]) == 1
    out = capsys.readouterr().out
    assert "numbers changed (source 1000; translation 100)" in out
    assert "untranslated; unreviewed" in out and "length ratio 0.45" in out
    assert (
        "10 translations: unreviewed 6, numbers changed 1, untranslated 1, length 1"
        in out
    )
    assert "2 machine translation(s) held back" in out
