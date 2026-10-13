"""Tests for the end-to-end reference implementation (reference/end_to_end).

Runs the pipeline as a subprocess, which starts the MCP server as its own
subprocess, so the test covers the protocol round trip. Requires jsonschema,
mcp, and pcn-core (see reference/end_to_end/requirements.txt).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "reference" / "end_to_end"
SAMPLE = E2E / "sample_run"

pytest.importorskip("pcn_core")
pytest.importorskip("mcp")


def run(out: Path, *args: str) -> tuple[int, str, dict]:
    proc = subprocess.run(
        [sys.executable, str(E2E / "run.py"), "--out", str(out), *args],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=E2E,
        check=False,
    )
    report = (
        json.loads((out / "report.json").read_text(encoding="utf-8"))
        if (out / "report.json").exists()
        else {}
    )
    return proc.returncode, proc.stdout + proc.stderr, report


def test_clean_run_publishes_validates_serves_and_verifies(tmp_path):
    code, console, r = run(tmp_path / "clean")
    assert code == 0, console
    assert r["result"] == "PASS"
    # Validation withholds the one record that lacks a definition, and only that one.
    withheld = [x["idno"] for x in r["records"] if not x["published"]]
    assert withheld == ["LF_UNEMP_PCT"]
    assert all(not x["structure_errors"] for x in r["records"])
    # Publication writes the records that passed, plus values, schema.org, and the index.
    assert r["published"] == [
        "FS_UNDERNOURISH_PCT",
        "ED_PRIMARY_NER_PCT",
        "HE_U5_MORT_RATE",
        "PV_HEADCOUNT_NPL_PCT",
        "EX-LFS-2025-Q2",
    ]
    assert (
        tmp_path / "clean" / "catalog" / "jsonld" / "EX-LFS-2025-Q2.jsonld"
    ).exists()
    assert not (
        tmp_path / "clean" / "catalog" / "records" / "LF_UNEMP_PCT.json"
    ).exists()
    # The MCP round trip: six calls, and the only failure is the withheld series.
    assert len(r["tool_calls"]) == 6
    failed = [c for c in r["tool_calls"] if not c["ok"]]
    assert [c["arguments"]["series"] for c in failed] == ["LF_UNEMP_PCT"]
    # Every tagged number verifies; the answer says the withheld series is absent.
    assert r["counts"] == {"verified": 3, "flagged": 0, "unmarked": 0}
    assert "No published series for the unemployment rate" in r["answer"]
    assert "verified number matches the record it cites" in r["caveat"]


def test_planted_errors_are_flagged_and_unmarked(tmp_path):
    code, console, r = run(tmp_path / "planted", "--inject-error")
    assert code == 1, console
    assert r["result"] == "FAIL"
    assert r["counts"] == {"verified": 1, "flagged": 1, "unmarked": 1}
    by_outcome = {o["outcome"]: o for o in r["verification"]}
    assert by_outcome["flagged"]["number"] == "7.2"
    assert by_outcome["flagged"]["claim_id"] == "FS_UNDERNOURISH_PCT:NAT:2024"
    assert by_outcome["unmarked"]["number"] == "41"
    assert by_outcome["unmarked"]["claim_id"] is None


def test_external_answer_is_verified_against_the_same_claims(tmp_path):
    answer = tmp_path / "answer.md"
    answer.write_text(
        'Undernourishment stood at <claim id="FS_UNDERNOURISH_PCT:NAT:2024" policy="rounded" decimals="0">8</claim> percent in 2024, '
        'and at <claim id="FS_UNDERNOURISH_PCT:NAT:2023" policy="exact">7.9</claim> percent the year before. '
        'A made-up record says <claim id="NO_SUCH:NAT:2024" policy="exact">3</claim>.\n',
        encoding="utf-8",
    )
    code, console, r = run(tmp_path / "ext", "--answer", str(answer))
    assert code == 1, console
    outcomes = [(o["number"], o["outcome"]) for o in r["verification"]]
    assert ("8", "verified") in outcomes  # 7.6 rounds to 8 at zero decimals
    assert ("7.9", "verified") in outcomes
    assert ("3", "flagged") in outcomes  # unknown claim identifier fails closed
    assert r["answer_source"] == "from answer.md"


def test_nada_dry_run_builds_requests_without_sending(tmp_path):
    code, console, r = run(
        tmp_path / "nada", "--nada-url", "https://catalog.example", "--nada-dry-run"
    )
    assert code == 0, console
    assert [n["status"] for n in r["nada"]] == ["dry-run"] * 5
    urls = [n["url"] for n in r["nada"]]
    assert urls[0] == "https://catalog.example/index.php/api/datasets/create/timeseries"
    assert (
        urls[-1]
        == "https://catalog.example/index.php/api/datasets/create/survey/EX-LFS-2025-Q2"
    )


def test_committed_sample_matches_a_fresh_run(tmp_path):
    for name, args in (("clean", ()), ("planted", ("--inject-error",))):
        committed = (SAMPLE / name / "report.md").read_text(encoding="utf-8")
        _, _, _ = run(tmp_path / name, *args)
        fresh = (tmp_path / name / "report.md").read_text(encoding="utf-8")
        assert fresh == committed, (
            f"sample_run/{name}/report.md is stale; rerun run.py --out sample_run/{name} {' '.join(args)}"
        )
