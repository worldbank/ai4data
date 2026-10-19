"""Tests for the scripts shipped with the serving-statistics-to-agents cookbook."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = (
    REPO / "website" / "static" / "cookbook-files" / "serving-statistics-to-agents"
)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest_check():
    return load_script("check_tool_manifest")


@pytest.fixture(scope="module")
def agent_eval():
    return load_script("agent_eval")


def test_example_manifest_passes(manifest_check, capsys):
    assert manifest_check.main([str(SCRIPTS / "tool_manifest.json")]) == 0
    out = capsys.readouterr().out
    assert "statistics-example 1.0.0: 4 tool(s), 3 resource(s)" in out
    assert "0 error(s), 0 warning(s)" in out


def test_manifest_rules_catch_design_faults(manifest_check, tmp_path, capsys):
    m = json.loads((SCRIPTS / "tool_manifest.json").read_text(encoding="utf-8"))
    m["tools"][0]["name"] = "SearchSeries"
    m["tools"][0]["description"] = "Search."
    m["tools"][2]["readOnlyHint"] = False
    m["tools"][2]["outputs"].remove("SOURCE_URL")
    del m["tools"][3]["example"]
    m["resources"] = []
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(m), encoding="utf-8")
    assert manifest_check.main([str(bad)]) == 1
    out = capsys.readouterr().out
    for expected in (
        "SearchSeries: name must be a snake_case verb phrase",
        "description has 1 words",
        "get_observations: not marked read-only",
        "lacks provenance fields ['SOURCE_URL']",
        "list_code_values: no example call",
        "no guidance resource",
    ):
        assert expected in out


def test_agent_eval_scores_the_example_traces(agent_eval, capsys):
    assert (
        agent_eval.main(
            [str(SCRIPTS / "agent_questions.csv"), str(SCRIPTS / "agent_traces.jsonl")]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "tools   7/8" in out
    assert "value   4/5" in out
    assert "behave  7/8" in out
    assert (
        "a05: value failed" in out
    )  # the French answer gives 6.3 for a quarter whose value is 6.1
    assert (
        "a07: tools failed" in out and "a07: behave failed" in out
    )  # answered a regional figure the series does not have


def test_number_parsing_handles_french_decimals_and_thousands(agent_eval):
    assert agent_eval.numbers_in("6,3 % et 12,500 ménages en 2025") == [
        6.3,
        12500.0,
        2025.0,
    ]
    assert agent_eval.numbers_in("7.6% in 2024") == [7.6, 2024.0]


@pytest.fixture(scope="module")
def guidance():
    return load_script("check_guidance")


@pytest.fixture(scope="module")
def quota():
    return load_script("quota_audit")


@pytest.fixture(scope="module")
def listing():
    return load_script("check_server_listing")


def test_guidance_passes_and_names_every_tool(guidance, tmp_path, capsys):
    assert (
        guidance.main(
            [str(SCRIPTS / "guidance_resource.md"), str(SCRIPTS / "tool_manifest.json")]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "tools named: 4/4" in out and "result: PASS" in out
    broken = tmp_path / "guidance.md"
    broken.write_text(
        (SCRIPTS / "guidance_resource.md")
        .read_text(encoding="utf-8")
        .replace("## Limitations", "## Notes")
        .replace("`get_observations`", "`fetch_values`"),
        encoding="utf-8",
    )
    assert guidance.main([str(broken), str(SCRIPTS / "tool_manifest.json")]) == 1
    out = capsys.readouterr().out
    assert (
        "MISSING Limitations" in out
        and "not named: get_observations" in out
        and "unknown tool names in guidance: fetch_values" in out
    )


def test_quota_audit_reports_unknown_clients_and_missing_terms(quota, capsys):
    assert (
        quota.main(
            [str(SCRIPTS / "client_registry.csv"), str(SCRIPTS / "agent_requests.csv")]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "27 requests from 6 clients; registry lists 5" in out
    assert (
        "unknown clients: k-0000" in out
        and "registered without accepted terms: k-2b8e" in out
    )
    anon = next(line for line in out.splitlines() if line.startswith("pub-anon"))
    assert anon.split()[2:6] == ["10", "8", "30", "2"]


def test_server_listing_check(listing, tmp_path, capsys):
    assert listing.main([str(SCRIPTS / "server_listing.json")]) == 0
    assert "0 error(s), 0 warning(s)" in capsys.readouterr().out
    bad = json.loads((SCRIPTS / "server_listing.json").read_text(encoding="utf-8"))
    bad["name"] = "statistics-example"
    bad["remotes"][0]["url"] = "http://stats.example/mcp"
    del bad["title"]
    path = tmp_path / "server.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    assert listing.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert (
        "name must be reverse-DNS" in out
        and "remotes[0].url must be an https URL" in out
        and "title missing" in out
    )
