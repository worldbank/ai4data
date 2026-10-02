"""Tests for deterministic concept-coverage helpers."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from schema_development.concept_coverage import (
    compute_cost,
    make_classification_model,
    process_response,
)


class FakeUsage:
    """Provide the usage fields consumed by coverage response processing."""

    input_tokens = 100
    output_tokens = 50
    total_tokens = 150

    def model_dump(self) -> dict[str, int]:
        """Return JSON-compatible token usage."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def test_classification_model_restricts_canonical_names() -> None:
    """The generated model accepts only concepts in the supplied inventory."""
    model = make_classification_model(("title", "time_period", "not_in_ontology"))

    classification = model(canonical_name="title", confidence="high")

    assert classification.model_dump(mode="json") == {
        "canonical_name": "title",
        "confidence": "high",
    }
    with pytest.raises(ValidationError):
        model(canonical_name="invented_concept", confidence="high")


def test_concept_coverage_processes_structured_output() -> None:
    """A constrained classification is serialized with deterministic cost data."""
    model = make_classification_model(("title", "not_in_ontology"))
    response = SimpleNamespace(
        output_text=" classification ",
        output_parsed=model(canonical_name="title", confidence="medium"),
        usage=FakeUsage(),
        status="completed",
        incomplete_details=None,
    )

    result = process_response(response, model="gpt-5.4-mini")

    assert result["parsed_output"] == {
        "canonical_name": "title",
        "confidence": "medium",
    }
    assert result["cost"] == compute_cost("gpt-5.4-mini", response.usage)
    assert result["error"] is None
