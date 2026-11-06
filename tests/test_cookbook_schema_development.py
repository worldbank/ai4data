"""Check the offline schema-development exercises and faithful teaching inputs."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "website/static/cookbook-files/ai-assisted-metadata-schema-development"


def _load_script(name: str) -> ModuleType:
    """Load a downloadable exercise without importing the application package."""
    spec = importlib.util.spec_from_file_location(name, ASSETS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts() -> dict[str, ModuleType]:
    """Load the three standalone scripts once for the focused tests.

    Returns
    -------
    dict[str, ModuleType]
        Modules keyed by their downloadable script names.
    """
    return {
        name: _load_script(name)
        for name in ["check_manifest", "inspect_discovery", "profile_observations"]
    }


def _observations() -> list[dict]:
    """Read the real three-observation exercise input."""
    return json.loads((ASSETS / "geographic_observations.json").read_text())


def _register() -> list[dict[str, str]]:
    """Read the publication-safe source-image register."""
    with (ASSETS / "source_images.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _responses() -> list[dict]:
    """Read the selected recorded discovery responses."""
    return [
        json.loads(line)
        for line in (ASSETS / "discovery_excerpt.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_exercise_inputs_match_authoritative_evidence() -> None:
    """New input files preserve the selected Task 1 observations exactly."""
    evidence = json.loads((ASSETS / "evidence.json").read_text())
    examples = {row["id"]: row for row in evidence["examples"]}
    assert (
        _observations()
        == examples["exact-name-profile"]["inputs"]["recorded_observations"]
    )
    titles = examples["independent-naming"]["outputs"]["recorded_observations"]
    fields = [
        "metadata_field",
        "observed_value",
        "description",
        "source_level",
        "discovery_value",
        "reasoning",
    ]
    for record in _responses():
        assert record["evidence_type"] == "teaching_excerpt"
        assert record["parsed_output"]["fields"] == [
            {key: row[key] for key in fields}
            for row in titles
            if row["snapshot_id"] == record["snapshot_id"]
        ]


def test_real_profile_counts_and_evidence(scripts: dict[str, ModuleType]) -> None:
    """The geographic exercise keeps its counts, denominator and inference text."""
    rows = _observations()
    before = copy.deepcopy(rows)
    (profile,) = scripts["profile_observations"].aggregate_observations(rows)
    assert (
        profile["count"],
        profile["n_snapshots"],
        profile["total_snapshots"],
        profile["support"],
    ) == (3, 3, 3, 1.0)
    assert profile["evidence"] == rows == before
    assert "inferred" in profile["evidence"][0]["observed_value"]


def test_occurrences_and_distinct_support_differ(
    scripts: dict[str, ModuleType],
) -> None:
    """Repeated observations from one artifact do not increase snapshot support."""
    first, second, third = _observations()
    second["metadata_field"] = "geographic_location"
    third["metadata_field"] = "Geographic_Scope"
    rows = [first, copy.deepcopy(first), second, third]
    profiles = {
        p["metadata_field"]: p
        for p in scripts["profile_observations"].aggregate_observations(rows)
    }
    assert set(profiles) == {
        "geographic_scope",
        "geographic_location",
        "Geographic_Scope",
    }
    assert profiles["geographic_scope"]["count"] == 2
    assert profiles["geographic_scope"]["n_snapshots"] == 1
    assert profiles["geographic_scope"]["support"] == pytest.approx(1 / 3)
    assert sum(p["count"] for p in profiles.values()) == 4


@pytest.mark.parametrize("bad", [[], {}, [None], [{}]])
def test_profile_rejects_malformed_input(
    scripts: dict[str, ModuleType], bad: object
) -> None:
    """Empty and malformed input cannot produce misleading empty profiles."""
    with pytest.raises((TypeError, ValueError)):
        scripts["profile_observations"].aggregate_observations(bad)


def test_profile_rejects_unknown_source_level(scripts: dict[str, ModuleType]) -> None:
    """Evidence-source labels must belong to the demonstrated contract."""
    rows = _observations()
    rows[0]["source_level"] = "unknown"
    with pytest.raises(ValueError, match="source level"):
        scripts["profile_observations"].aggregate_observations(rows)


def test_profile_writes_separate_json(
    scripts: dict[str, ModuleType], tmp_path: Path
) -> None:
    """A CLI run creates valid new JSON while leaving its observation input intact."""
    source = ASSETS / "geographic_observations.json"
    before = source.read_bytes()
    output = tmp_path / "outputs" / "profiles.json"
    assert (
        scripts["profile_observations"].main([str(source), "--output", str(output)])
        == 0
    )
    assert json.loads(output.read_text())[0]["evidence"] == _observations()
    assert source.read_bytes() == before


@pytest.mark.parametrize("use_input", [False, True])
def test_profile_refuses_existing_paths(
    scripts: dict[str, ModuleType], tmp_path: Path, use_input: bool
) -> None:
    """An existing output or the input itself cannot be overwritten."""
    source = tmp_path / "input.json"
    source.write_text(json.dumps(_observations()))
    output = source if use_input else tmp_path / "existing.json"
    if not use_input:
        output.write_text("preserve this output")
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        scripts["profile_observations"].main([str(source), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == before


def test_bad_input_does_not_create_output_directory(
    scripts: dict[str, ModuleType], tmp_path: Path
) -> None:
    """Input validation occurs before output-directory creation."""
    source = tmp_path / "bad.json"
    source.write_text("[]")
    output = tmp_path / "new-directory" / "result.json"
    with pytest.raises(SystemExit):
        scripts["profile_observations"].main([str(source), "--output", str(output)])
    assert not output.parent.exists()


def test_selected_manifest_counts(scripts: dict[str, ModuleType]) -> None:
    """The eight-row excerpt is distinct from the full study allocation."""
    assert scripts["check_manifest"].summarize_manifest(_register()) == {
        "development_snapshots": 6,
        "development_parents": 6,
        "heldout_snapshots": 2,
        "heldout_parents": 2,
        "shared_parents": 0,
    }


def test_manifest_checks_parent_scope_and_overlap(
    scripts: dict[str, ModuleType],
) -> None:
    """Parents are corpus-scoped, and overlapping snapshot identities are rejected."""
    first, second = _register()[:2]
    first.update(split="development", parent_document_id="same")
    second.update(split="heldout", parent_document_id="same")
    assert first["corpus"] != second["corpus"]
    assert (
        scripts["check_manifest"].summarize_manifest([first, second])["shared_parents"]
        == 0
    )
    second["corpus"] = first["corpus"]
    assert (
        scripts["check_manifest"].summarize_manifest([first, second])["shared_parents"]
        == 1
    )
    second["snapshot_id"] = first["snapshot_id"]
    with pytest.raises(ValueError, match="overlapping"):
        scripts["check_manifest"].summarize_manifest([first, second])


@pytest.mark.parametrize("bad", [[], [{"snapshot_id": "missing-values"}]])
def test_manifest_rejects_incomplete_register(
    scripts: dict[str, ModuleType], bad: list[dict]
) -> None:
    """Absent identities and empty registers fail validation."""
    with pytest.raises(ValueError):
        scripts["check_manifest"].summarize_manifest(bad)


def test_recorded_response_inspection(scripts: dict[str, ModuleType]) -> None:
    """The selected fields and complete-response counts retain separate units."""
    assert scripts["inspect_discovery"].inspect_records(_responses(), _register()) == (
        3,
        0,
        4,
        43,
    )


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "heldout",
        "source_level",
        "truncated",
        "missing_parsed",
        "missing_model",
    ],
)
def test_response_inspection_failure_modes(
    scripts: dict[str, ModuleType], change: str
) -> None:
    """Identity, provenance, contract and completion failures remain distinguishable."""
    records = _responses()
    if change == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif change == "heldout":
        heldout = next(row for row in _register() if row["split"] == "heldout")
        records[0].update(snapshot_id=heldout["snapshot_id"], source=heldout["corpus"])
    elif change == "source_level":
        records[0]["parsed_output"]["fields"][0]["source_level"] = "unverified"
    elif change == "missing_parsed":
        records[0]["parsed_output"] = None
    elif change == "missing_model":
        records[0]["model"] = ""
    else:
        records[0].update(
            status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
        assert scripts["inspect_discovery"].inspect_records(records, _register()) == (
            2,
            1,
            3,
            29,
        )
        return
    with pytest.raises(ValueError):
        scripts["inspect_discovery"].inspect_records(records, _register())
