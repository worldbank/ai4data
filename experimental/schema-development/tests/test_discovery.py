"""Tests for deterministic metadata-discovery helpers."""

from types import SimpleNamespace

from schema_development.discovery import (
    CandidateField,
    SchemaDiscoveryResult,
    compute_cost,
    process_response,
)


class FakeUsage:
    """Provide the usage fields consumed by discovery response processing."""

    input_tokens = 1_000_000
    output_tokens = 1_000_000
    total_tokens = 2_000_000

    def model_dump(self) -> dict[str, int]:
        """Return JSON-compatible token usage."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def test_discovery_processes_structured_output() -> None:
    """A parsed response is serialized with its usage and cost."""
    parsed = SchemaDiscoveryResult(
        fields=[
            CandidateField(
                metadata_field="time_period",
                observed_value="2024",
                description="Period represented by the data.",
                source_level="snapshot",
                discovery_value="high",
                reasoning="Supports temporal filtering.",
            )
        ]
    )
    response = SimpleNamespace(
        output_text=" structured output ",
        output_parsed=parsed,
        usage=FakeUsage(),
        status="completed",
        incomplete_details=None,
    )

    result = process_response(response, model="gpt-5.4-mini")

    assert result["raw_output"] == "structured output"
    assert result["parsed_output"]["fields"][0]["metadata_field"] == "time_period"
    assert result["cost"] == compute_cost("gpt-5.4-mini", response.usage)
    assert result["cost"]["total_cost_usd"] == 5.25
    assert result["error"] is None


def test_discovery_records_a_missing_parse() -> None:
    """An incomplete structured response remains inspectable and retryable."""
    response = SimpleNamespace(
        output_text="incomplete",
        output_parsed=None,
        usage=FakeUsage(),
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )

    result = process_response(response, model="gpt-5.5")

    assert result["parsed_output"] is None
    assert result["error"] == (
        "Structured output parse failed or response was incomplete."
    )
