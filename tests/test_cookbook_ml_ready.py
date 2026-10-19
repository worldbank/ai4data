"""Tests for the scripts shipped with the ml-ready-datasets cookbook."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "ml-ready-datasets"
NAMES = [
    "make_splits",
    "representativeness_report",
    "build_croissant",
    "croissant_check",
    "check_dataset_card",
    "release_manifest",
]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts():
    return {n: load_script(n) for n in NAMES}


def test_make_splits_is_grouped_and_reproducible(scripts, tmp_path, capsys):
    out = tmp_path / "splits.csv"
    args = [
        str(SCRIPTS / "lfs_occupation_ml.csv"),
        "--id",
        "record_id",
        "--group",
        "hhid",
        "--stratify",
        "region",
        "-o",
        str(out),
    ]
    assert scripts["make_splits"].main(args) == 0
    text = capsys.readouterr().out
    assert "600 records, 344 groups by 'hhid', seed 1" in text
    assert "groups in more than one split: 0" in text
    assert out.read_text(encoding="utf-8") == (SCRIPTS / "splits.csv").read_text(
        encoding="utf-8"
    )
    # a different seed gives a different assignment
    out2 = tmp_path / "splits2.csv"
    assert (
        scripts["make_splits"].main(args[:-2] + ["--seed", "2", "-o", str(out2)]) == 0
    )
    assert out2.read_text(encoding="utf-8") != out.read_text(encoding="utf-8")


def test_representativeness_report_flags_differences(scripts, capsys):
    rc = scripts["representativeness_report"].main(
        [
            str(SCRIPTS / "lfs_occupation_ml.csv"),
            str(SCRIPTS / "census_benchmarks.csv"),
            "--weight",
            "weight",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert (
        "R01           86     0.143     0.225       0.270    -12.7  differs from the population"
        in out
    )
    assert "categories below 30 records: 0" in out
    rc = scripts["representativeness_report"].main(
        [
            str(SCRIPTS / "lfs_occupation_ml.csv"),
            str(SCRIPTS / "census_benchmarks.csv"),
            "--min-count",
            "100",
        ]
    )
    assert rc == 1
    assert "region=R01 (86)" in capsys.readouterr().out


def test_build_and_check_croissant(scripts, tmp_path, capsys):
    for name in ["lfs_occupation_ml.csv", "dictionary.csv", "splits.csv"]:
        shutil.copy(SCRIPTS / name, tmp_path / name)
    out = tmp_path / "record.json"
    args = [
        str(tmp_path / "lfs_occupation_ml.csv"),
        str(tmp_path / "dictionary.csv"),
        "--name",
        "x",
        "--description",
        "y",
        "--url",
        "https://stats.example/d",
        "--license",
        "https://creativecommons.org/licenses/by/4.0/",
        "--version",
        "1.0.0",
        "--date-published",
        "2026-10-09",
        "--cite-as",
        "z",
        "--splits",
        str(tmp_path / "splits.csv"),
        "-o",
        str(out),
    ]
    assert scripts["build_croissant"].main(args) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert [d["@id"] for d in rec["distribution"]] == ["data-file", "splits-file"]
    assert len(rec["recordSet"][0]["field"]) == 12
    assert rec["recordSet"][1]["dataType"] == "cr:Split"
    assert scripts["croissant_check"].main([str(out)]) == 0
    assert "0 error(s), 0 warning(s)" in capsys.readouterr().out
    # the shipped record matches the shipped files
    assert (
        scripts["croissant_check"].main(
            [str(SCRIPTS / "lfs_occupation_ml.croissant.json")]
        )
        == 0
    )
    # an edited file fails the checksum
    (tmp_path / "splits.csv").write_text("record_id,split\n", encoding="utf-8")
    assert scripts["croissant_check"].main([str(out)]) == 1
    assert "sha256 does not match splits.csv" in capsys.readouterr().out


def test_check_dataset_card(scripts, tmp_path, capsys):
    assert scripts["check_dataset_card"].main([str(SCRIPTS / "dataset_card.md")]) == 0
    assert "result: PASS" in capsys.readouterr().out
    broken = tmp_path / "card.md"
    text = (SCRIPTS / "dataset_card.md").read_text(encoding="utf-8")
    broken.write_text(
        text.replace("## Out-of-scope uses", "## Other uses").replace(
            "license: cc-by-4.0\n", ""
        ),
        encoding="utf-8",
    )
    assert scripts["check_dataset_card"].main([str(broken)]) == 1
    out = capsys.readouterr().out
    assert (
        "MISSING  front matter: license" in out and "MISSING  Out-of-scope uses" in out
    )


def test_release_manifest_build_and_verify(scripts, tmp_path, capsys):
    names = ["lfs_occupation_ml.csv", "splits.csv", "dictionary.csv"]
    for name in names:
        shutil.copy(SCRIPTS / name, tmp_path / name)
    manifest = tmp_path / "manifest.json"
    assert (
        scripts["release_manifest"].main(
            ["build", "--version", "1.0.0", "--date", "2026-10-09", "-o", str(manifest)]
            + [str(tmp_path / n) for n in names]
        )
        == 0
    )
    assert scripts["release_manifest"].main(["verify", str(manifest)]) == 0
    assert "result: PASS" in capsys.readouterr().out
    (tmp_path / "splits.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "dictionary.csv").unlink()
    assert scripts["release_manifest"].main(["verify", str(manifest)]) == 1
    out = capsys.readouterr().out
    assert "CHANGED   splits.csv" in out and "MISSING   dictionary.csv" in out
    # the shipped manifest matches the shipped files
    assert (
        scripts["release_manifest"].main(["verify", str(SCRIPTS / "manifest.json")])
        == 0
    )
