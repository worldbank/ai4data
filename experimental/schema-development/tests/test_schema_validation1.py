"""Tests for the Schema Validation 1 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from schema_development.validation1 import SchemaValidationResult, run_validation
from schema_development.validation1.validation import ValidationObservation


class FakeResponses:
    """Record Responses API requests and return configured fake responses."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        """Record one request and return the next fake response."""
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _write_inputs(root: Path) -> dict[str, Path]:
    """Create a minimal Validation 1 input layout."""
    snapshots = root / "snapshots"
    metadata = root / "metadata"
    snapshot = snapshots / "prwp" / "figure" / "report_figure_000.png"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"png-data")
    metadata_path = metadata / "prwp" / "report.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text('{"title": "Parent report"}', encoding="utf-8")

    schema = root / "schema.md"
    schema.write_text("# Schema v1.1\n\n### title\n\nDefinition.\n", encoding="utf-8")
    config = root / "config.json"
    config.write_text(
        json.dumps({"model": "gpt-5.5", "reasoning": {"effort": "medium"}}),
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


def _assessment(field_name: str = "title") -> SchemaValidationResult:
    """Create one parsed assessment using the supplied closest field."""
    return SchemaValidationResult(
        observations=[
            ValidationObservation(
                observation_id="obs_1",
                metadata_concept="Displayed title",
                evidence="The snapshot displays a title.",
                evidence_source="snapshot",
                closest_schema_fields=[field_name],
                fit_status="covered",
                fit_rationale="The title field represents the concept.",
            )
        ],
        candidate_new_fields=[],
    )


def _response(parsed: SchemaValidationResult) -> object:
    """Create a minimal SDK-like structured response."""
    return SimpleNamespace(
        id="resp_test",
        status="completed",
        output_parsed=parsed,
        output_text=parsed.model_dump_json(),
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _run(paths: dict[str, Path], responses: FakeResponses) -> object:
    """Run Validation 1 with a fake API client."""
    return run_validation(
        snapshots_dir=paths["snapshots"],
        metadata_dir=paths["metadata"],
        schema_path=paths["schema"],
        results_path=paths["results"],
        errors_path=paths["errors"],
        config_path=paths["config"],
        sleep_seconds=0,
        client=SimpleNamespace(responses=responses),
    )


def test_validation1_uses_schema_and_parent_context(tmp_path: Path) -> None:
    """A held-out snapshot request includes the frozen schema and metadata."""
    paths = _write_inputs(tmp_path)
    responses = FakeResponses([_response(_assessment())])

    summary = _run(paths, responses)

    assert summary.discovered == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    request = responses.requests[0]
    assert request["model"] == "gpt-5.5"
    assert request["text_format"] is SchemaValidationResult
    user_prompt = request["input"][1]["content"][0]["text"]
    assert "### title" in user_prompt
    assert '"title": "Parent report"' in user_prompt
    record = json.loads(paths["results"].read_text(encoding="utf-8"))
    assert record["snapshot_file_name"] == "report_figure_000.png"
    assert record["parsed_output"]["observations"][0]["fit_status"] == "covered"


def test_validation1_skips_a_successful_result(tmp_path: Path) -> None:
    """A successful snapshot is not submitted again on a resumed run."""
    paths = _write_inputs(tmp_path)
    _run(paths, FakeResponses([_response(_assessment())]))

    responses = FakeResponses([])
    summary = _run(paths, responses)

    assert summary.skipped == 1
    assert summary.succeeded == 0
    assert not responses.requests


def test_validation1_logs_unknown_schema_fields(tmp_path: Path) -> None:
    """A model-invented closest field is recorded as a retryable error."""
    paths = _write_inputs(tmp_path)

    summary = _run(paths, FakeResponses([_response(_assessment("invented_field"))]))

    assert summary.failed == 1
    assert not paths["results"].exists()
    error = json.loads(paths["errors"].read_text(encoding="utf-8"))
    assert error["error_stage"] == "validation"
    assert "Unknown v1.1 fields" in error["error"]
