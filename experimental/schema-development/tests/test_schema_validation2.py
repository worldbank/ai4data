"""Tests for the confirmatory Schema Validation 2 pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from schema_development.paths import ROOT
from schema_development.validation2 import SchemaValidationResult, run_validation
from schema_development.validation2.validation import CandidateGap


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
    """Create a minimal Validation 2 filesystem layout."""
    snapshots = root / "snapshots"
    metadata = root / "metadata"
    snapshot_path = snapshots / "prwp" / "figure" / "document_1_figure_000.png"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"png-data")
    metadata_path = metadata / "prwp" / "document_1.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text('{"title": "Parent report"}', encoding="utf-8")

    schema = root / "schema.md"
    schema.write_text(
        "# Data Snapshot Metadata Schema v1.1.1\n\n"
        "### title\n\nDefinition.\n\n"
        "### time_period\n\nDefinition.\n",
        encoding="utf-8",
    )
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
        "schema": schema,
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


def _run(paths: dict[str, Path], responses: FakeResponses) -> object:
    """Run the pipeline with shared test paths and a fake client."""
    client = SimpleNamespace(responses=responses)
    return run_validation(
        snapshots_dir=paths["snapshots"],
        metadata_dir=paths["metadata"],
        schema_path=paths["schema"],
        results_path=paths["results"],
        errors_path=paths["errors"],
        config_path=paths["config"],
        sleep_seconds=0,
        client=client,
    )


def test_validation2_discovers_snapshots_and_uses_parent_context(
    tmp_path: Path,
) -> None:
    """Discovered snapshots are assessed with full document context."""
    paths = _write_inputs(tmp_path)
    parsed = SchemaValidationResult(
        coverage_assessment="no_critical_gap_found",
        critical_or_possible_gaps=[],
    )
    responses = FakeResponses([_response(parsed)])

    summary = _run(paths, responses)

    assert summary.discovered == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert len(responses.requests) == 1
    request = responses.requests[0]
    assert request["model"] == "gpt-5.5"
    assert request["service_tier"] == "flex"
    assert request["text_format"] is SchemaValidationResult
    system_prompt = request["input"][0]["content"][0]["text"]
    user_prompt = request["input"][1]["content"][0]["text"]
    assert "Validation 1" not in system_prompt
    assert "Validation 2" not in system_prompt
    assert "Validation 1" not in user_prompt
    assert "Validation 2" not in user_prompt
    assert "Data Snapshot Metadata Schema v1.1.1" in user_prompt
    assert '"title": "Parent report"' in user_prompt
    record = json.loads(paths["results"].read_text(encoding="utf-8"))
    assert record["snapshot_file_name"] == "document_1_figure_000.png"
    assert record["parsed_output"]["coverage_assessment"] == ("no_critical_gap_found")


def test_validation2_skips_successful_results_on_rerun(tmp_path: Path) -> None:
    """Successful filenames are skipped when the notebook is rerun."""
    paths = _write_inputs(tmp_path)
    parsed = SchemaValidationResult(
        coverage_assessment="no_critical_gap_found",
        critical_or_possible_gaps=[],
    )
    first_responses = FakeResponses([_response(parsed)])
    _run(paths, first_responses)

    second_responses = FakeResponses([])
    summary = _run(paths, second_responses)

    assert summary.discovered == 1
    assert summary.skipped == 1
    assert summary.succeeded == 0
    assert not second_responses.requests


def test_validation2_rejects_an_empty_snapshot_directory(tmp_path: Path) -> None:
    """An empty snapshot directory stops execution before API calls."""
    paths = _write_inputs(tmp_path)
    paths["snapshots"].joinpath("prwp", "figure", "document_1_figure_000.png").unlink()

    with pytest.raises(ValueError, match="No snapshots found"):
        _run(paths, FakeResponses([]))


def test_validation2_logs_inconsistent_assessments_as_errors(
    tmp_path: Path,
) -> None:
    """Cross-field inconsistencies fail validation and remain retryable."""
    paths = _write_inputs(tmp_path)
    gap = CandidateGap(
        gap_id="gap_1",
        gap_status="possible",
        missing_metadata_concept="A reusable concept",
        proposed_field_name="missing_concept",
        evidence="The snapshot explicitly names the concept.",
        why_snapshot_metadata="The concept describes the represented data.",
        closest_schema_fields=["title"],
        why_existing_fields_may_be_insufficient="Title may conflate the concept.",
        material_impact="interpretability",
        material_consequence="The represented concept could be misunderstood.",
        uncertainty_note="Materiality requires human review.",
    )
    parsed = SchemaValidationResult(
        coverage_assessment="no_critical_gap_found",
        critical_or_possible_gaps=[gap],
    )

    summary = _run(paths, FakeResponses([_response(parsed)]))

    assert summary.failed == 1
    assert summary.succeeded == 0
    assert not paths["results"].exists()
    error = json.loads(paths["errors"].read_text(encoding="utf-8"))
    assert error["error_stage"] == "validation"
    assert "no-gap result" in error["error"]


def test_validation2_output_omits_rationale_and_evidence_source() -> None:
    """The final Structured Output excludes the removed review fields."""
    result_properties = SchemaValidationResult.model_json_schema()["properties"]
    gap_properties = CandidateGap.model_json_schema()["properties"]

    assert "assessment_rationale" not in result_properties
    assert "assessment_status" not in result_properties
    assert "assessment_limitation" not in result_properties
    assert "evidence_source" not in gap_properties


def test_model_facing_schema_preserves_inventory_without_outcome_cues() -> None:
    """The model-facing schema preserves fields while omitting outcome cues."""
    canonical_path = ROOT / "schemas/Data Snapshot Metadata Schema v1.1.1.md"
    model_facing_path = (
        ROOT / "src/schema_development/validation2/schema_v1.1.1_model_facing.md"
    )
    canonical = canonical_path.read_text(encoding="utf-8")
    model_facing = model_facing_path.read_text(encoding="utf-8")
    field_pattern = re.compile(r"^### ([a-z][a-z0-9_]*)$", re.MULTILINE)

    assert field_pattern.findall(model_facing) == field_pattern.findall(canonical)
    assert "Schema Validation Exercise" not in model_facing
    assert "held-out validation results" not in model_facing
    assert "Examples reviewed during validation" not in model_facing
    assert "documentation revision following" not in model_facing


def test_title_ablation_schema_removes_only_title_field() -> None:
    """The counterfactual schema removes only the snapshot title field."""
    model_facing_path = (
        ROOT / "src/schema_development/validation2/schema_v1.1.1_model_facing.md"
    )
    ablation_path = (
        ROOT / "src/schema_development/validation2/schema_v1.1.1_title_ablated.md"
    )
    model_facing = model_facing_path.read_text(encoding="utf-8")
    ablated = ablation_path.read_text(encoding="utf-8")
    field_pattern = re.compile(r"^### ([a-z][a-z0-9_]*)$", re.MULTILINE)
    expected_fields = field_pattern.findall(model_facing)
    expected_fields.remove("title")

    assert field_pattern.findall(ablated) == expected_fields
    assert "### title" not in ablated
    assert "### panel_title" in ablated
    assert "### source_document_title" in ablated


def test_source_note_ablation_schema_removes_only_target_fields() -> None:
    """The Calibration3 schema removes only source and note fields."""
    model_facing_path = (
        ROOT / "src/schema_development/validation2/schema_v1.1.1_model_facing.md"
    )
    ablation_path = (
        ROOT / "src/schema_development/validation2/"
        "schema_v1.1.1_data_source_interpretive_note_ablated.md"
    )
    model_facing = model_facing_path.read_text(encoding="utf-8")
    ablated = ablation_path.read_text(encoding="utf-8")
    field_pattern = re.compile(r"^### ([a-z][a-z0-9_]*)$", re.MULTILINE)
    expected_fields = field_pattern.findall(model_facing)
    expected_fields.remove("data_source")
    expected_fields.remove("interpretive_note")

    assert field_pattern.findall(ablated) == expected_fields
    assert "data_source" not in ablated
    assert "interpretive_note" not in ablated
