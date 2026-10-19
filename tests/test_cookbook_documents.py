"""Tests for the scripts shipped with the data-from-documents cookbook.

The scripts live under website/static/cookbook-files/data-from-documents and
are loaded by path. The table record is validated with the metadata checker
from the dissemination cookbook against the pinned schema fixtures.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "data-from-documents"
DISSEMINATION = (
    REPO / "website" / "static" / "cookbook-files" / "ai-ready-dissemination"
)
SCHEMAS = REPO / "tests" / "fixtures" / "wb-schemas"


def load_script(folder: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verify():
    return load_script(SCRIPTS, "verify_extraction")


@pytest.fixture(scope="module")
def tidy():
    return load_script(SCRIPTS, "snapshot_to_tidy")


@pytest.fixture(scope="module")
def inventory():
    return load_script(SCRIPTS, "inventory_report")


def test_chart_values_match_printed_labels(verify, capsys):
    assert verify.main([str(SCRIPTS / "example_snapshot.json")]) == 0
    out = capsys.readouterr().out
    assert "4 verified, 0 flagged, 0 estimated" in out


def test_table_check_flags_the_planted_error(verify, capsys):
    assert verify.main([str(SCRIPTS / "example_table.json")]) == 1
    out = capsys.readouterr().out
    assert "flagged   Eastern Region/Prevalence (%)" in out
    assert "8.2 vs computed 9.19" in out
    assert "total 5640 vs sum of rows 5640" in out
    assert "14 verified, 1 flagged, 0 estimated" in out


def test_chart_without_printed_labels_is_estimated(verify):
    results = verify.verify_chart(
        {
            "categories": ["A", "B"],
            "series": [{"name": "S", "values": [1.5, 2.5]}],
            "value_labels_printed": False,
        }
    )
    assert [status for _, status, _ in results] == ["estimated", "estimated"]


def test_tidy_file_carries_provenance_and_statuses(verify, tidy, tmp_path, capsys):
    verify.main([str(SCRIPTS / "example_table.json")])
    report = tmp_path / "report.txt"
    report.write_text(capsys.readouterr().out, encoding="utf-8")
    out = tmp_path / "T3.1.csv"
    assert (
        tidy.main(
            [
                str(SCRIPTS / "example_table.json"),
                "--verification",
                str(report),
                "-o",
                str(out),
            ]
        )
        == 0
    )
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 15
    assert rows[0]["document_id"] == "EX-REP-2025-03" and rows[0]["page"] == "24"
    assert rows[0]["bbox"] == "0.08 0.21 0.92 0.52"
    statuses = {(r["row"], r["column"]): r["status"] for r in rows}
    assert statuses[("Eastern Region", "Prevalence (%)")] == "flagged"
    assert statuses[("Central Region", "Prevalence (%)")] == "verified"
    assert statuses[("National", "Population (thousands)")] == "verified"
    assert all(r["unit"] for r in rows)


def test_inventory_report_orders_the_backlog(inventory, capsys):
    assert inventory.main([str(SCRIPTS / "document_inventory.csv")]) == 0
    out = capsys.readouterr().out
    assert "8 documents, 315 tables, 88 figures; 7 of 8 with a text layer" in out
    backlog = [
        line.strip()
        for line in out.split("backlog (largest first):")[1].strip().splitlines()
    ]
    assert backlog[0].startswith("EX-YB-2023")
    assert "needs OCR" in next(line for line in backlog if "EX-CEN-2014-01" in line)


def test_table_record_is_valid_and_complete(capsys):
    pytest.importorskip("jsonschema")
    checker = load_script(DISSEMINATION, "check_metadata")
    code = checker.main(
        [
            str(SCRIPTS / "table_record.json"),
            str(SCRIPTS / "profile_table.json"),
            "--level",
            "ai-ready",
            "--schema-dir",
            str(SCHEMAS),
        ]
    )
    out = capsys.readouterr().out
    assert "1 of 1 records are valid instances of the schema" in out
    assert "1 of 1 records complete" in out
    assert code == 0


@pytest.fixture(scope="module")
def ocr():
    return load_script(SCRIPTS, "ocr_quality")


@pytest.fixture(scope="module")
def stack():
    return load_script(SCRIPTS, "stack_series")


@pytest.fixture(scope="module")
def manifest():
    return load_script(SCRIPTS, "build_manifest")


def test_ocr_quality_flags_garbled_numbers_and_recommends_re_ocr(ocr, capsys):
    assert ocr.main([str(SCRIPTS / "ocr_sample_page.txt")]) == 0
    out = capsys.readouterr().out
    assert "14 lines, 33 number tokens, 3 garbled (9.1%), 4 broken words" in out
    assert "2: Tab1e" in out and "9: popu1ation" in out
    assert "verdict: re-OCR recommended" in out
    assert "631,2O5" in out and "l,412,006" in out


def test_stack_series_confirms_and_revises_overlaps(stack, tmp_path, capsys):
    out_path = tmp_path / "series.csv"
    assert (
        stack.main(
            [
                str(SCRIPTS / "series_mapping.csv"),
                str(SCRIPTS / "tidy_fs2024.csv"),
                str(SCRIPTS / "tidy_fs2025.csv"),
                "-o",
                str(out_path),
            ]
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "18 tidy rows from 2 documents -> 12 series values; 0 rows unmatched" in err
    assert "single 6, confirmed 4, revised 2" in err
    rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
    revised = {(r["area"], r["period"]): r for r in rows if r["status"] == "revised"}
    assert (
        revised[("Northern Region", "2023")]["value"] == "9.6"
        and revised[("Northern Region", "2023")]["previous_value"] == "9.7"
    )
    assert all(r["unit"] == "% of population" for r in rows)


def test_manifest_skips_published_and_orders_by_priority(manifest, tmp_path, capsys):
    out_path = tmp_path / "manifest.csv"
    assert (
        manifest.main(
            [
                str(SCRIPTS / "document_inventory.csv"),
                str(SCRIPTS / "processing_status.csv"),
                "-o",
                str(out_path),
            ]
        )
        == 0
    )
    err = capsys.readouterr().err
    assert (
        "8 documents in the inventory, 2 published and skipped, 6 in the manifest"
        in err
    )
    assert (
        "pages to OCR 220, pages to lay out 621, tables to extract 286, retries 2"
        in err
    )
    rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
    assert [r["document_id"] for r in rows[:3]] == [
        "EX-REP-2024-03",
        "EX-REP-2025-03",
        "EX-CEN-2014-01",
    ]
    census = next(r for r in rows if r["document_id"] == "EX-CEN-2014-01")
    assert (
        census["next_stage"] == "text"
        and census["retry"] == "yes"
        and census["ocr"] == "yes"
    )
