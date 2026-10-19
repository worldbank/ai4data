"""Tests for the scripts shipped with the AI-ready dissemination cookbook.

The scripts live under website/static/cookbook-files/ai-ready-dissemination
so that readers can download them. They are loaded here by path. The World
Bank schema files the structure layer needs are a pinned snapshot in
tests/fixtures/wb-schemas, so the tests run without network access.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "ai-ready-dissemination"
SCHEMAS = REPO / "tests" / "fixtures" / "wb-schemas"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve postponed annotations through sys.modules, so the
    # module has to be registered before it executes.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check():
    return load_script("check_metadata")


@pytest.fixture(scope="module")
def score():
    return load_script("score_retrieval")


@pytest.fixture(scope="module")
def verify():
    return load_script("verify_numbers")


def run_check(check, capsys, *args: str) -> tuple[int, str]:
    code = check.main([*args, "--schema-dir", str(SCHEMAS)])
    return code, capsys.readouterr().out


# --- check_metadata: the four example types ---------------------------------

EXAMPLES = [
    ("example_catalog.csv", "profile_indicator.json", 5, 4),
    ("example_microdata.json", "profile_microdata.json", 1, 1),
    ("example_geospatial.json", "profile_geospatial.json", 1, 1),
    ("example_document.json", "profile_document.json", 1, 1),
]


@pytest.mark.parametrize(("records", "profile", "n", "complete"), EXAMPLES)
def test_examples_validate_and_score_at_foundational(
    check, capsys, records, profile, n, complete
):
    pytest.importorskip("jsonschema")
    code, out = run_check(check, capsys, str(SCRIPTS / records), str(SCRIPTS / profile))
    assert f"{n} of {n} records are valid instances of the schema" in out
    assert f"{complete} of {n} records complete" in out
    assert "Validity: 0 issue(s)" in out
    assert code == (0 if complete == n else 1)


def test_indicator_example_names_the_one_gap_with_its_reason(check, capsys):
    _, out = run_check(
        check,
        capsys,
        str(SCRIPTS / "example_catalog.csv"),
        str(SCRIPTS / "profile_indicator.json"),
    )
    assert "LF_UNEMP_PCT: series_description.definition_long" in out
    assert "why: Search by meaning" in out


def test_ai_ready_level_is_cumulative_and_lists_more_fields(check, capsys):
    _, foundational = run_check(
        check,
        capsys,
        str(SCRIPTS / "example_catalog.csv"),
        str(SCRIPTS / "profile_indicator.json"),
    )
    _, ai_ready = run_check(
        check,
        capsys,
        str(SCRIPTS / "example_catalog.csv"),
        str(SCRIPTS / "profile_indicator.json"),
        "--level",
        "ai-ready",
    )
    assert "(9 fields)" in foundational
    assert "(21 fields)" in ai_ready
    assert "0 of 5 records complete" in ai_ready
    assert "missing series_description.methodology" in ai_ready


def test_bad_record_trips_every_validity_rule(check, capsys, tmp_path):
    header = (
        "idno,name,definition_long,measurement_unit,periodicity,time_period_start,"
        "time_period_end,geographic_units,sources,date_last_update,definition_references"
    )
    row = "BAD ID,Unemployment rate,Unemployment rate,n/a,yearly,2025,2015,national,LFS,20/08/2025,stats.example/def"
    bad = tmp_path / "bad.csv"
    bad.write_text(f"{header}\n{row}\n", encoding="utf-8")
    code, out = run_check(
        check, capsys, str(bad), str(SCRIPTS / "profile_indicator.json")
    )
    assert code == 1
    assert "Validity: 8 issue(s)" in out
    for expected in (
        "identifier has characters outside",
        "date_last_update '20/08/2025' is not YYYY-MM-DD",
        "time_periods.end (2015) is before",
        "definition_references.uri 'stats.example/def' is not a URL",
        "periodicity 'yearly' is not in the profile vocabulary",
        "definition_long has fewer than 12 words",
        "definition_long repeats series_description.name",
        "measurement_unit is a placeholder ('n/a')",
    ):
        assert expected in out


def test_duplicate_identifiers_are_reported(check, capsys, tmp_path):
    src = (SCRIPTS / "example_catalog.csv").read_text(encoding="utf-8").splitlines()
    dup = tmp_path / "dup.csv"
    dup.write_text("\n".join([*src, src[1]]) + "\n", encoding="utf-8")
    _, out = run_check(check, capsys, str(dup), str(SCRIPTS / "profile_indicator.json"))
    assert "FS_UNDERNOURISH_PCT: identifier appears 2 times" in out


# --- check_metadata: units ---------------------------------------------------


def test_csv_rows_become_nested_schema_records(check):
    import json

    with (SCRIPTS / "profile_indicator.json").open(encoding="utf-8") as fh:
        profile = json.load(fh)
    store = check.SchemaStore(SCHEMAS)
    records = check.load_records(
        SCRIPTS / "example_catalog.csv", profile, store, "timeseries-schema.json"
    )
    sd = records[0]["series_description"]
    assert sd["idno"] == "FS_UNDERNOURISH_PCT"
    assert sd["time_periods"] == [{"start": "2010", "end": "2024"}]
    assert sd["geographic_units"] == [{"name": "national"}, {"name": "region"}]
    assert sd["definition_references"] == [
        {"uri": "https://stats.example/def/FS_UNDERNOURISH_PCT"}
    ]
    assert (
        "definition_long" not in records[3]["series_description"]
    )  # empty cell is absent, not ""


def test_schema_store_resolves_references_across_files(check):
    store = check.SchemaStore(SCHEMAS)
    # microdata -> allOf ddi-schema.json -> study_desc -> study_info -> coll_dates (array)
    assert store.is_array("microdata-schema.json", "study_desc.study_info.coll_dates")
    assert (
        store.item_key("microdata-schema.json", "study_desc.study_info.coll_dates")
        == "start"
    )
    assert store.item_key("timeseries-schema.json", "series_description.name") is None
    assert (
        store.node_at("timeseries-schema.json", "series_description.no_such_field")
        is None
    )


def test_path_helpers_flatten_through_lists(check):
    record = {"a": {"b": [{"c": "x"}, {"c": ""}, {"c": "y"}]}, "n": 0, "e": []}
    assert check.values_at(record, ["a", "b", "c"]) == ["x", "", "y"]
    assert check.present(record, "a.b.c")
    assert check.first(record, "a.b.c") == "x"
    assert check.strings_at(record, "a.b.c") == ["x", "y"]
    assert check.present(record, "n")  # zero is a value
    assert not check.present(record, "e")  # empty list is not
    assert not check.present(record, "a.z")


def test_unknown_type_or_level_exit_with_code_2(check, capsys, tmp_path):
    import json

    profile = tmp_path / "p.json"
    profile.write_text(
        json.dumps({"type": "spreadsheet", "levels": {"foundational": []}}),
        encoding="utf-8",
    )
    assert check.main([str(SCRIPTS / "example_document.json"), str(profile)]) == 2
    code = check.main(
        [
            str(SCRIPTS / "example_document.json"),
            str(SCRIPTS / "profile_document.json"),
            "--level",
            "x",
        ]
    )
    assert code == 2
    assert "unknown" in capsys.readouterr().err


def test_structure_layer_is_skipped_without_jsonschema(check, monkeypatch):
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    store = check.SchemaStore(SCHEMAS)
    result = check.layer_structure(
        [{}], store, "timeseries-schema.json", "series_description.idno"
    )
    assert result.valid is None
    assert "layer 1 skipped" in result.problems[0]


# --- score_retrieval ----------------------------------------------------------


def test_keyword_baseline_scores_match_the_cookbook_text(score, capsys):
    assert (
        score.main(
            [str(SCRIPTS / "eval_questions.csv"), str(SCRIPTS / "example_catalog.csv")]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "en      10   0.80   0.80   0.80" in out
    assert "fr       2   0.00   0.00   0.00" in out
    assert "es       2   0.00   0.00   0.00" in out


def test_search_ranks_exact_title_first_and_is_unicode_aware(score):
    catalog = pd.read_csv(SCRIPTS / "example_catalog.csv", dtype=str).to_dict("records")
    assert score.search("unemployment rate", catalog)[0] == "LF_UNEMP_PCT"
    assert score.tokens("Taux de chômage élevé") == {"taux", "de", "chômage", "élevé"}


# --- verify_numbers -----------------------------------------------------------


def test_number_extraction_skips_unit_text_and_trailing_commas(verify):
    text = "41 per 1,000 live births in 2020, and 12,500 households (7.6%)."
    assert list(verify.find_numbers(text)) == ["41", "2020", "12,500", "7.6"]
    assert verify.parse("12,500") == 12500.0


def test_tolerance_is_the_larger_of_absolute_and_relative(verify):
    assert verify.matches(7.60, 7.6, 0.05, 0.002)
    assert not verify.matches(7.2, 7.6, 0.05, 0.002)
    assert verify.matches(12480, 12500, 0.05, 0.002)  # within 0.2 percent
    assert not verify.matches(12400, 12500, 0.05, 0.002)


@pytest.mark.parametrize(
    ("answer", "code", "unverified"),
    [
        ("Undernourishment was 8.4% in 2022 and 7.6% in 2024.", 0, 0),
        ("Undernourishment fell from 8.4% in 2022 to 7.2% in 2024.", 1, 1),
    ],
)
def test_verify_main_reads_stdin_and_sets_exit_code(
    verify, capsys, monkeypatch, answer, code, unverified
):
    monkeypatch.setattr(sys, "stdin", io.StringIO(answer))
    assert verify.main(["-", str(SCRIPTS / "example_values.csv")]) == code
    out = capsys.readouterr().out
    assert f"{unverified} unverified" in out
    assert "8.4  verified   FS_UNDERNOURISH_PCT NAT 2022" in out


# --- mcp_server_example -------------------------------------------------------


def test_mcp_tools_return_provenance_with_values():
    pytest.importorskip("mcp")
    server = load_script("mcp_server_example")
    hits = server.search_series("unemployment", limit=2)
    assert hits[0]["idno"] == "LF_UNEMP_PCT"
    values = server.get_values("LF_UNEMP_PCT")
    assert values["UNIT_MEASURE"] == "% of labour force"
    assert values["SOURCE_URL"].startswith("https://")
    assert values["observations"][-1]["OBS_STATUS"] == "P"
    assert "error" in server.get_values("NO_SUCH_SERIES")
