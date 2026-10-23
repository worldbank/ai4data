"""Extract validated Schema v1.4 metadata from one snapshot image."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ai4data.data_snapshot.metadata_schema import DataSnapshotMetadata

_PACKAGE_DIR = Path(__file__).parent
_CONFIG_DIR = _PACKAGE_DIR / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "openrouter.json"
_PROMPT_DIR = _PACKAGE_DIR / "prompts"
_COMPLETENESS_PLACEHOLDER = "{{COMPLETENESS_GUIDANCE}}"
_SCHEMA_PLACEHOLDER = "{{MODEL_FACING_SCHEMA_SECTION}}"
_COMPLETENESS_GUIDANCE = (
    "Before returning, re-scan the image against the entire schema and correct any "
    "omissions of fields supported by visible evidence. Leave every unsupported "
    "field null."
)
_PIPELINE_OWNED_CONFIG_FIELDS = {
    "input",
    "instructions",
    "model",
    "provider",
    "stream",
    "text",
    "text_format",
}
_SUPPORTED_PROVIDERS = {"openai", "openrouter"}
_UNSUPPORTED_SCHEMA_KEYWORDS = {
    "$schema",
    "allOf",
    "default",
    "dependentRequired",
    "dependentSchemas",
    "else",
    "examples",
    "if",
    "not",
    "then",
}
_SUPPORTED_STRING_FORMATS = {
    "date",
    "date-time",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "time",
    "uuid",
}


@dataclass(frozen=True)
class ExtractionResult:
    """Represent one metadata extraction outcome.

    Parameters
    ----------
    metadata : DataSnapshotMetadata | None
        Validated Schema v1.4 metadata, or ``None`` after a failure.
    model : str | None
        Requested model, when configuration loading succeeded.
    response_id : str | None
        OpenAI response identifier, when available.
    api_status : str | None
        OpenAI response status, when available.
    incomplete_details : dict[str, Any] | None
        Structured explanation for an incomplete response, when available.
    elapsed_seconds : float | None
        API-call duration, or ``None`` when no call was attempted.
    usage : dict[str, Any] | None
        JSON-compatible token usage returned by the API.
    raw_output : str | None
        Structured output text returned by the API, when available.
    error_type : str | None
        Exception class name after a failure.
    error : str | None
        Error message after a failure.
    """

    metadata: DataSnapshotMetadata | None
    model: str | None
    response_id: str | None
    api_status: str | None
    incomplete_details: dict[str, Any] | None
    elapsed_seconds: float | None
    usage: dict[str, Any] | None
    raw_output: str | None
    error_type: str | None
    error: str | None


def load_extraction_config(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load metadata API settings from a JSON object.

    Parameters
    ----------
    config_path : str | Path, optional
        JSON file containing ``model``, an optional ``provider``, and API
        settings.

    Returns
    -------
    dict[str, Any]
        Validated request settings.

    Raises
    ------
    ValueError
        If the config is not an object, lacks a model, or overrides a field
        owned by the extraction pipeline.
    """
    with Path(config_path).open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Extraction config must be a JSON object.")
    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Extraction config requires a non-empty 'model' string.")
    provider = config.get("provider", "openai")
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError("Extraction config provider must be 'openai' or 'openrouter'.")
    forbidden = (set(config) - {"model", "provider"}) & (_PIPELINE_OWNED_CONFIG_FIELDS)
    if forbidden:
        raise ValueError(
            f"Extraction config cannot set pipeline-owned keys: {sorted(forbidden)}."
        )
    return config


def get_extraction_config_path(provider: str) -> Path:
    """Return the packaged configuration for a metadata provider.

    Parameters
    ----------
    provider : str
        Supported metadata provider name.

    Returns
    -------
    pathlib.Path
        Path to the provider's packaged JSON configuration.

    Raises
    ------
    ValueError
        If the provider is not supported.
    """
    normalized_provider = provider.strip().lower()
    if normalized_provider not in _SUPPORTED_PROVIDERS:
        raise ValueError("Metadata provider must be 'openai' or 'openrouter'.")
    return _CONFIG_DIR / f"{normalized_provider}.json"


def extract_metadata(
    image_path: str | Path,
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    client: Any | None = None,
) -> ExtractionResult:
    """Extract validated Schema v1.4 metadata from one snapshot image.

    Only image-visible evidence is sent to the model. Filenames, directory
    names, and source-document metadata are excluded from the prompt.

    Parameters
    ----------
    image_path : str | Path
        Path to a PNG, JPEG, GIF, or WebP snapshot image.
    config_path : str | Path, optional
        Responses API configuration file.
    client : Any | None, optional
        Injected OpenAI-compatible client for offline tests.

    Returns
    -------
    ExtractionResult
        Validated metadata and call details, or structured failure details.
    """
    model: str | None = None
    started_at: float | None = None
    response: Any | None = None
    try:
        config = load_extraction_config(config_path)
        model = config.pop("model")
        provider = config.pop("provider", "openai")
        system_prompt = _read_prompt(
            "system.md", _COMPLETENESS_PLACEHOLDER, _COMPLETENESS_GUIDANCE
        )
        user_prompt = _read_prompt("user.md", _SCHEMA_PLACEHOLDER, "")
        image_url = _image_data_url(image_path)
        api_client = client if client is not None else _create_api_client(provider)

        started_at = time.perf_counter()
        response = api_client.responses.create(
            model=model,
            text={"format": _response_format()},
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                },
            ],
            **config,
        )
        elapsed_seconds = time.perf_counter() - started_at
        status = getattr(response, "status", None)
        incomplete_details = _serialize_api_value(
            getattr(response, "incomplete_details", None)
        )
        if status == "incomplete":
            reason = (incomplete_details or {}).get("reason", "unknown reason")
            raise ValueError(f"Response was incomplete: {reason}.")
        if status not in {None, "completed"}:
            detail = _api_error_text(getattr(response, "error", None))
            suffix = f" ({detail})" if detail else ""
            raise ValueError(f"Response ended with status: {status}{suffix}.")
        raw_output = getattr(response, "output_text", None)
        if not isinstance(raw_output, str) or not raw_output:
            raise ValueError("Structured output was missing.")
        metadata = DataSnapshotMetadata.model_validate_json(raw_output)
        return ExtractionResult(
            metadata=metadata,
            model=model,
            response_id=getattr(response, "id", None),
            api_status=status,
            incomplete_details=incomplete_details,
            elapsed_seconds=elapsed_seconds,
            usage=_serialize_api_value(getattr(response, "usage", None)),
            raw_output=raw_output,
            error_type=None,
            error=None,
        )
    except Exception as error:
        elapsed_seconds = (
            time.perf_counter() - started_at if started_at is not None else None
        )
        return ExtractionResult(
            metadata=None,
            model=model,
            response_id=getattr(response, "id", None),
            api_status=getattr(response, "status", None),
            incomplete_details=_serialize_api_value(
                getattr(response, "incomplete_details", None)
            ),
            elapsed_seconds=elapsed_seconds,
            usage=_serialize_api_value(getattr(response, "usage", None)),
            raw_output=getattr(response, "output_text", None),
            error_type=type(error).__name__,
            error=str(error),
        )


def _read_prompt(filename: str, placeholder: str, replacement: str) -> str:
    """Read one prompt and replace its required placeholder."""
    prompt = (_PROMPT_DIR / filename).read_text(encoding="utf-8")
    if prompt.count(placeholder) != 1:
        raise ValueError(f"Prompt {filename} must contain {placeholder} exactly once.")
    return prompt.replace(placeholder, replacement).rstrip() + "\n"


def _image_data_url(image_path: str | Path) -> str:
    """Encode a supported local image as a base64 data URL."""
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/gif", "image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"Unsupported image file: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _openai_compatible_schema(value: Any) -> Any:
    """Remove unsupported keywords from canonical Pydantic JSON Schema data."""
    if isinstance(value, list):
        return [_openai_compatible_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    validation_rules = value.get("x-validation-rules")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in _UNSUPPORTED_SCHEMA_KEYWORDS or key.startswith("x-"):
            continue
        if key == "format" and item not in _SUPPORTED_STRING_FORMATS:
            continue
        if key == "anyOf" and _contains_schema_keyword(item, "not"):
            continue
        result[key] = _openai_compatible_schema(item)

    if isinstance(validation_rules, list) and all(
        isinstance(rule, str) for rule in validation_rules
    ):
        guidance = "Validation rules:\n- " + "\n- ".join(validation_rules)
        description = result.get("description")
        result["description"] = (
            f"{description}\n\n{guidance}" if description else guidance
        )
    properties = result.get("properties")
    if result.get("type") == "object" and isinstance(properties, dict):
        result["required"] = list(properties)
        result["additionalProperties"] = False
    return result


def _contains_schema_keyword(value: Any, keyword: str) -> bool:
    """Return whether nested JSON Schema data contain a named keyword."""
    if isinstance(value, dict):
        return keyword in value or any(
            _contains_schema_keyword(item, keyword) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_schema_keyword(item, keyword) for item in value)
    return False


def _inline_ref_siblings(value: Any, root: dict[str, Any]) -> Any:
    """Inline local references that have sibling schema keywords."""
    if isinstance(value, list):
        return [_inline_ref_siblings(item, root) for item in value]
    if not isinstance(value, dict):
        return value

    reference = value.get("$ref")
    if isinstance(reference, str) and len(value) > 1:
        if not reference.startswith("#/"):
            raise ValueError(f"Unsupported JSON Schema reference: {reference}")
        resolved: Any = root
        for part in reference[2:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(resolved, dict) or key not in resolved:
                raise ValueError(f"Unresolved JSON Schema reference: {reference}")
            resolved = resolved[key]
        if not isinstance(resolved, dict):
            raise ValueError(f"JSON Schema reference is not an object: {reference}")
        value = {**resolved, **value}
        value.pop("$ref")
    return {key: _inline_ref_siblings(item, root) for key, item in value.items()}


@cache
def _response_format() -> dict[str, Any]:
    """Build the strict OpenAI response format from the canonical model."""
    canonical_schema = DataSnapshotMetadata.model_json_schema(mode="validation")
    schema = _openai_compatible_schema(canonical_schema)
    schema = _inline_ref_siblings(schema, schema)
    return {
        "type": "json_schema",
        "name": "data_snapshot_metadata_v1_4",
        "strict": True,
        "schema": schema,
    }


def _create_api_client(provider: str) -> Any:
    """Create the configured API client with automatic retries disabled.

    Parameters
    ----------
    provider : str
        Supported API provider name.

    Returns
    -------
    Any
        Configured OpenAI-compatible SDK client.

    Raises
    ------
    ValueError
        If the selected provider's API key is not set.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path.cwd() / ".env")
    if provider == "openrouter":
        api_key_name = "OPENROUTER_API_KEY"
        base_url = "https://openrouter.ai/api/v1"
    else:
        api_key_name = "OPENAI_API_KEY"
        base_url = None
    api_key = os.getenv(api_key_name)
    if not api_key:
        raise ValueError(f"{api_key_name} is not set.")
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=0)


def _serialize_api_value(value: Any) -> dict[str, Any] | None:
    """Convert an OpenAI SDK object to a JSON-compatible dictionary."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", warnings=False)
    if isinstance(value, dict):
        return value
    return {
        key: getattr(value, key)
        for key in (
            "input_tokens",
            "output_tokens",
            "reason",
            "total_tokens",
        )
        if hasattr(value, key)
    }


def _api_error_text(value: Any) -> str | None:
    """Return an API error code and message suitable for CLI output."""
    if value is None:
        return None
    if isinstance(value, dict):
        code = value.get("code")
        message = value.get("message")
    else:
        code = getattr(value, "code", None)
        message = getattr(value, "message", None)
    detail = ": ".join(str(item) for item in (code, message) if item)
    return detail or str(value)
