"""Tests for the confirmatory Schema Validation 3 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from schema_development.paths import ROOT
from schema_development.validation2.validation import CandidateGap as CandidateGapV2
from schema_development.validation3 import (
    CandidateGap,
    SchemaValidationResult,
    run_validation,
)
from schema_development.validation3.validation import (
    _frozen_schema_json,
    _schema_context,
)


class FakeResponses:
    """Record parsed Responses API requests and return configured results."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        """Record a request and return the next configured response."""
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _write_inputs(root: Path) -> dict[str, Path]:
    """Create a minimal Validation 3 filesystem layout."""
    snapshots = root / "snapshots"
    metadata = root / "metadata"
    snapshot_path = snapshots / "prwp" / "figure" / "document_1_figure_000.png"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"png-data")
    second_path = snapshots / "unhcr" / "table" / "document_2_table_001.png"
    second_path.parent.mkdir(parents=True)
    second_path.write_bytes(b"png-data")
    for source, document in (("prwp", "document_1"), ("unhcr", "document_2")):
        metadata_path = metadata / source / f"{document}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text('{"title": "Parent report"}', encoding="utf-8")
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "model": "gpt-5.5",
                "service_tier": "flex",
                "reasoning": {"effort": "medium"},
            }
        ),
        encoding="utf-8",
    )
    return {
        "snapshots": snapshots,
        "metadata": metadata,
        "config": config,
        "results": root / "results.jsonl",
        "errors": root / "errors.jsonl",
    }


def _response(parsed: SchemaValidationResult) -> object:
    """Create a minimal parsed SDK-like response."""
    return SimpleNamespace(
        id="resp_test",
        status="completed",
        output_parsed=parsed,
        output_text=parsed.model_dump_json(),
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _no_gap_response() -> object:
    """Create a successful no-gap response."""
    return _response(
        SchemaValidationResult(
            coverage_assessment="no_critical_gap_found",
            critical_or_possible_gaps=[],
        )
    )


def _candidate(**overrides: object) -> CandidateGap:
    """Create a valid possible-gap candidate with optional overrides."""
    values: dict[str, object] = {
        "gap_id": "gap_1",
        "gap_status": "possible",
        "missing_metadata_concept": "Temporal qualifier",
        "proposed_field_name": "temporal_qualifier",
        "evidence": "The snapshot explicitly labels the period.",
        "why_snapshot_metadata": "It qualifies the represented data.",
        "closest_schema_paths": ["temporal_coverage.period"],
        "why_existing_fields_may_be_insufficient": "Its meaning may be lost.",
        "material_impact": "interpretability",
        "material_consequence": "Readers could misread when the data apply.",
        "uncertainty_note": "Human review is required.",
    }
    values.update(overrides)
    return CandidateGap.model_validate(values)


def _run(
    paths: dict[str, Path],
    responses: FakeResponses,
    **kwargs: object,
) -> object:
    """Run the pipeline with shared paths and a fake client."""
    client = SimpleNamespace(responses=responses)
    return run_validation(
        snapshots_dir=paths["snapshots"],
        metadata_dir=paths["metadata"],
        results_path=paths["results"],
        errors_path=paths["errors"],
        config_path=paths["config"],
        sleep_seconds=0,
        client=client,
        **kwargs,
    )


def test_validation3_uses_frozen_schema_and_parent_context(tmp_path: Path) -> None:
    """Requests contain frozen v1.2 JSON and parent-document context."""
    paths = _write_inputs(tmp_path)
    responses = FakeResponses([_no_gap_response()])

    summary = _run(
        paths,
        responses,
        snapshot_file_names={"document_1_figure_000.png"},
    )

    assert summary.discovered == 1
    assert summary.succeeded == 1
    request = responses.requests[0]
    assert request["model"] == "gpt-5.5"
    assert request["text_format"] is SchemaValidationResult
    user_prompt = request["input"][1]["content"][0]["text"]
    assert '"x-schema-version":"1.2"' in user_prompt
    assert '"title": "Parent report"' in user_prompt
    record = json.loads(paths["results"].read_text(encoding="utf-8"))
    assert record["schema_version"] == "1.2"
    assert record["excluded_schema_fields"] == []


def test_validation3_selects_exact_filenames_and_rejects_unknown(
    tmp_path: Path,
) -> None:
    """Calibration controls can select one exact snapshot without copying it."""
    paths = _write_inputs(tmp_path)
    responses = FakeResponses([_no_gap_response()])
    summary = _run(
        paths,
        responses,
        snapshot_file_names={"document_2_table_001.png"},
    )
    assert summary.discovered == 1
    assert json.loads(paths["results"].read_text())["source"] == "unhcr"

    with pytest.raises(ValueError, match="were not found"):
        _run(
            paths,
            FakeResponses([]),
            snapshot_file_names={"missing_table_000.png"},
        )


def test_validation3_skips_successful_results_on_rerun(tmp_path: Path) -> None:
    """Successful filenames are skipped when the notebook is rerun."""
    paths = _write_inputs(tmp_path)
    selected = {"document_1_figure_000.png"}
    _run(paths, FakeResponses([_no_gap_response()]), snapshot_file_names=selected)

    responses = FakeResponses([])
    summary = _run(paths, responses, snapshot_file_names=selected)

    assert summary.skipped == 1
    assert summary.succeeded == 0
    assert not responses.requests


def test_validation3_rejects_empty_input_directory(tmp_path: Path) -> None:
    """An empty snapshot directory stops execution before API calls."""
    paths = _write_inputs(tmp_path)
    for snapshot in paths["snapshots"].glob("*/*/*.png"):
        snapshot.unlink()

    with pytest.raises(ValueError, match="No snapshots found"):
        _run(paths, FakeResponses([]))


def test_validation3_rejects_unknown_nested_schema_paths(tmp_path: Path) -> None:
    """Candidates using invented v1.2 paths are logged as retryable errors."""
    paths = _write_inputs(tmp_path)
    parsed = SchemaValidationResult(
        coverage_assessment="possible_gap",
        critical_or_possible_gaps=[
            _candidate(closest_schema_paths=["temporal_coverage.invented"])
        ],
    )

    summary = _run(
        paths,
        FakeResponses([_response(parsed)]),
        snapshot_file_names={"document_1_figure_000.png"},
    )

    assert summary.failed == 1
    assert not paths["results"].exists()
    error = json.loads(paths["errors"].read_text(encoding="utf-8"))
    assert error["error_stage"] == "validation"
    assert "Unknown v1.2 schema paths" in error["error"]


def test_validation3_logs_inconsistent_assessments_as_errors(
    tmp_path: Path,
) -> None:
    """Snapshot status and candidate contents must agree."""
    paths = _write_inputs(tmp_path)
    parsed = SchemaValidationResult(
        coverage_assessment="no_critical_gap_found",
        critical_or_possible_gaps=[_candidate()],
    )

    summary = _run(
        paths,
        FakeResponses([_response(parsed)]),
        snapshot_file_names={"document_1_figure_000.png"},
    )

    assert summary.failed == 1
    error = json.loads(paths["errors"].read_text(encoding="utf-8"))
    assert "no-gap result" in error["error"]


@pytest.mark.parametrize("proposed_name", ["Not Snake Case", "unit"])
def test_validation3_rejects_invalid_or_existing_field_proposals(
    tmp_path: Path, proposed_name: str
) -> None:
    """Field proposals must be new snake-case metadata concepts."""
    paths = _write_inputs(tmp_path)
    parsed = SchemaValidationResult(
        coverage_assessment="possible_gap",
        critical_or_possible_gaps=[_candidate(proposed_field_name=proposed_name)],
    )

    summary = _run(
        paths,
        FakeResponses([_response(parsed)]),
        snapshot_file_names={"document_1_figure_000.png"},
    )

    assert summary.failed == 1
    error = json.loads(paths["errors"].read_text(encoding="utf-8"))
    assert "Proposed field" in error["error"]


def test_provenance_ablation_is_local_and_removes_target_paths() -> None:
    """The sensitivity control cannot mutate the cached frozen schema."""
    canonical_before = _frozen_schema_json()
    schema_json, paths = _schema_context(("provenance", "interpretive_notes"))
    schema = json.loads(schema_json)

    assert "provenance" not in schema["properties"]
    assert "interpretive_notes" not in schema["properties"]
    assert "interpretive_notes" not in schema_json
    assert "Provenance" not in schema_json
    assert "`provenance`" not in schema_json
    assert not any(path.startswith("provenance") for path in paths)
    assert "temporal_coverage.period" in paths
    assert _frozen_schema_json() == canonical_before


def test_schema_context_contains_expected_nested_paths() -> None:
    """Generated path inventory covers nested objects and array items."""
    _, paths = _schema_context()

    assert "variables[]" in paths
    assert "variables[].unit" in paths
    assert "variables[].unit.source_text" in paths
    assert "dimensions[].category_groups[].categories" in paths
    assert "dimensions[].category_groups[].categories[]" in paths
    assert "interpretive_notes[]" in paths
    assert "provenance.sources[].name" in paths


def test_validation3_instrument_matches_validation2_except_nested_paths() -> None:
    """The corrected instrument differs only where v1.2 paths require it."""
    validation2_prompt = (
        ROOT / "src/schema_development/validation2/prompts/system.md"
    ).read_text(encoding="utf-8")
    validation3_prompt = (
        ROOT / "src/schema_development/validation3/prompts/system.md"
    ).read_text(encoding="utf-8")
    normalized_prompt = validation3_prompt.replace("v1.2", "v1.1.1")
    normalized_prompt = normalized_prompt.replace(
        "using exact v1.1.1 schema paths. Use dot notation and `[]` for array "
        "items, such as `variables[].unit` or "
        "`dimensions[].category_groups[].categories`.",
        "using exact v1.1.1 field names.",
    ).replace(
        "exact closest v1.1.1 schema paths",
        "exact closest v1.1.1 fields",
    )

    assert normalized_prompt == validation2_prompt
    validation2_fields = set(CandidateGapV2.model_json_schema()["properties"])
    validation3_fields = set(CandidateGap.model_json_schema()["properties"])
    assert validation3_fields == (validation2_fields - {"closest_schema_fields"}) | {
        "closest_schema_paths"
    }
