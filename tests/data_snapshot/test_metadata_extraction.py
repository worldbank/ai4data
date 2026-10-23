"""Test Schema v1.4 single-snapshot metadata extraction offline."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai4data.data_snapshot import DataSnapshotMetadata, extract_metadata
from ai4data.data_snapshot.metadata_extraction import load_extraction_config
from ai4data.data_snapshot.metadata_extraction.extraction import (
    _create_api_client,
    _response_format,
    get_extraction_config_path,
)


class _FakeResponses:
    """Record Responses API requests and return a configured result."""

    def __init__(self, response: object | Exception) -> None:
        self.response = response
        self.request: dict[str, object] | None = None

    def create(self, **arguments: object) -> object:
        """Record a request and return or raise the configured result."""
        self.request = arguments
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _write_image(path: Path) -> None:
    """Write a minimal file accepted as a PNG input."""
    path.write_bytes(b"png-data")


def _write_config(path: Path, **overrides: object) -> None:
    """Write a minimal extraction configuration."""
    config: dict[str, object] = {"model": "test-model", "max_output_tokens": 25_000}
    config.update(overrides)
    path.write_text(json.dumps(config), encoding="utf-8")


def test_default_config_uses_openrouter_luna() -> None:
    """Use GPT-5.6 Luna through OpenRouter by default."""
    config = load_extraction_config()

    assert config == {
        "provider": "openrouter",
        "model": "openai/gpt-5.6-luna",
        "max_output_tokens": 25_000,
        "timeout": 900.0,
        "service_tier": "flex",
        "reasoning": {"effort": "medium"},
    }


def test_openai_config_preserves_runtime_settings() -> None:
    """Keep the direct OpenAI Luna request settings available."""
    config = load_extraction_config(get_extraction_config_path("openai"))

    assert config["provider"] == "openai"
    assert config["model"] == "gpt-5.6-luna"
    assert config["max_output_tokens"] == 25_000
    assert config["service_tier"] == "flex"
    assert config["reasoning"] == {"effort": "medium"}


def test_openrouter_client_uses_compatible_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point the existing OpenAI SDK at OpenRouter with its own API key."""
    import openai

    arguments: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> object:
        """Record SDK client configuration."""
        arguments.update(kwargs)
        return object()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", fake_openai)

    _create_api_client("openrouter")

    assert arguments == {
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "max_retries": 0,
    }


@pytest.mark.parametrize("exported_key", [None, "exported-key"])
def test_client_loads_working_directory_env_without_overriding_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exported_key: str | None
) -> None:
    """Read local credentials after installation and preserve shell precedence."""
    import openai

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    if exported_key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", exported_key)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=file-key\n", encoding="utf-8")
    arguments: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> object:
        """Record constructor arguments without making a request."""
        arguments.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    _create_api_client("openrouter")
    assert arguments["api_key"] == (exported_key or "file-key")


def test_extract_metadata_uses_image_and_validates_output(tmp_path: Path) -> None:
    """Send one image with a strict schema and validate the returned metadata."""
    image_path = tmp_path / "secret_source_table_000.png"
    config_path = tmp_path / "config.json"
    _write_image(image_path)
    _write_config(config_path)
    metadata = DataSnapshotMetadata(title="Displacement by year")
    responses = _FakeResponses(
        SimpleNamespace(
            id="resp_test",
            status="completed",
            incomplete_details=None,
            output_text=metadata.model_dump_json(),
            usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
        )
    )

    result = extract_metadata(
        image_path,
        config_path=config_path,
        client=SimpleNamespace(responses=responses),
    )

    assert result.metadata == metadata
    assert result.error is None
    assert responses.request is not None
    assert responses.request["text"]["format"] == _response_format()
    request_input = responses.request["input"]
    prompt_text = " ".join(
        item["text"]
        for message in request_input
        for item in message["content"]
        if item["type"] == "input_text"
    )
    assert image_path.name not in prompt_text
    assert request_input[1]["content"][1]["image_url"].startswith(
        "data:image/png;base64,"
    )


def test_response_format_is_strict_and_openai_compatible() -> None:
    """Generate Structured Outputs from the canonical Pydantic model."""
    response_format = _response_format()
    schema = response_format["schema"]
    serialized = json.dumps(schema)

    assert response_format["type"] == "json_schema"
    assert response_format["strict"] is True
    assert schema["required"] == list(schema["properties"])
    for unsupported in ("allOf", "else", "if", "not", "then"):
        assert f'"{unsupported}"' not in serialized
    assert "Validation rules:" in serialized


def test_incomplete_response_preserves_details_without_parsing(tmp_path: Path) -> None:
    """Report output-budget truncation before attempting Pydantic parsing."""
    image_path = tmp_path / "snapshot.png"
    config_path = tmp_path / "config.json"
    _write_image(image_path)
    _write_config(config_path)
    response = SimpleNamespace(
        id="resp_incomplete",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_text='{"title": "truncated',
        usage=None,
    )

    result = extract_metadata(
        image_path,
        config_path=config_path,
        client=SimpleNamespace(responses=_FakeResponses(response)),
    )

    assert result.metadata is None
    assert result.api_status == "incomplete"
    assert result.incomplete_details == {"reason": "max_output_tokens"}
    assert result.error == "Response was incomplete: max_output_tokens."
    assert result.raw_output == '{"title": "truncated'


def test_failed_response_reports_api_error(tmp_path: Path) -> None:
    """Include the provider's failure code and message in the result."""
    image_path = tmp_path / "snapshot.png"
    config_path = tmp_path / "config.json"
    _write_image(image_path)
    _write_config(config_path)
    response = SimpleNamespace(
        id="resp_failed",
        status="failed",
        incomplete_details=None,
        output_text="",
        usage=None,
        error=SimpleNamespace(
            code="provider_error",
            message="Structured output was rejected.",
        ),
    )

    result = extract_metadata(
        image_path,
        config_path=config_path,
        client=SimpleNamespace(responses=_FakeResponses(response)),
    )

    assert result.metadata is None
    assert result.api_status == "failed"
    assert result.error == (
        "Response ended with status: failed "
        "(provider_error: Structured output was rejected.)."
    )


def test_config_rejects_pipeline_owned_fields(tmp_path: Path) -> None:
    """Prevent config files from replacing prompts, input, or output schema."""
    config_path = tmp_path / "config.json"
    _write_config(config_path, input=[])

    try:
        load_extraction_config(config_path)
    except ValueError as error:
        assert "'input'" in str(error)
    else:
        raise AssertionError("Pipeline-owned input config was accepted.")


def test_config_rejects_unknown_provider(tmp_path: Path) -> None:
    """Reject provider names without a configured client endpoint."""
    config_path = tmp_path / "config.json"
    _write_config(config_path, provider="unknown")

    try:
        load_extraction_config(config_path)
    except ValueError as error:
        assert "provider" in str(error)
    else:
        raise AssertionError("Unknown metadata provider was accepted.")
