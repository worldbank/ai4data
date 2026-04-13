"""Extensible LLM explainer registry for anomaly explanation.

Add new providers by registering a callable that parses a batch output row
into the content dict. Build payloads via llm_client.build_payload (provider-agnostic).
"""

import json
import re
from typing import Any, Callable, Dict, List, Union

from ai4data.anomaly.explanation.schemas import AnomalyExplanation

# Type for a row parser: (row: dict) -> content_dict or None
RowParser = Callable[[Dict[str, Any]], dict | None]

_EXPLAINER_REGISTRY: Dict[str, RowParser] = {}


def _parse_openai_row(row: Dict[str, Any]) -> dict | None:
    """Parse OpenAI batch output row to content dict."""
    try:
        body = row["response"].get("body", row["response"])
        choices = body.get("choices", [])
        if choices:
            return json.loads(
                choices[0].get("message", {}).get("content", "{}")
            )
        output = body.get("output", [])
        return json.loads(
            output[0].get("content", [{}])[0].get("text", "{}")
        )
    except (KeyError, IndexError, json.JSONDecodeError):
        return None


_GEMINI_MARKDOWN_FENCE_OPEN = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)


def _strip_markdown_json_fence(text: str) -> str:
    """Remove leading `` ```json `` and trailing `` ``` `` from model output."""
    s = text.strip()
    s = _GEMINI_MARKDOWN_FENCE_OPEN.sub("", s, count=1)
    s = s.rstrip()
    if s.endswith("```"):
        s = s[: -3].rstrip()
    return s.strip()


def _parse_gemini_row(row: Dict[str, Any]) -> dict | None:
    """Parse Gemini batch output row to content dict.

    Gemini often wraps JSON in markdown fences and sometimes returns a top-level
    array instead of ``{\"anomalies\": [...]}``. OpenAI batch output is plain JSON
    matching the schema; we mirror that lenient parsing here so rows are not
    dropped when the model deviates slightly.
    """
    try:
        text = (
            row["response"]
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        text = _strip_markdown_json_fence(text)
        data: Union[dict, List[Any]] = json.loads(text)
        if isinstance(data, list):
            data = {"anomalies": data}
        if not isinstance(data, dict):
            return None
        return data
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


def _message_text_from_anthropic(msg: Any) -> str:
    """Concatenate text blocks from an Anthropic Message."""
    if msg is None:
        return ""
    content = msg.get("content", []) if isinstance(msg, dict) else getattr(
        msg, "content", []
    )
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        elif getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def _parse_anthropic_row(row: Dict[str, Any]) -> dict | None:
    """Parse Anthropic Message Batches output row to content dict."""
    from pydantic import ValidationError

    try:
        res = row.get("result") if isinstance(row, dict) else row["result"]
        if res is None:
            return None
        if hasattr(res, "model_dump"):
            res = res.model_dump(mode="json")
        if isinstance(res, dict) and res.get("type") != "succeeded":
            return None
        msg = res.get("message") if isinstance(res, dict) else res.message
        text = _message_text_from_anthropic(msg)
        if not text.strip():
            return None
        return json.loads(
            AnomalyExplanation.model_validate_json(text).model_dump_json()
        )
    except (ValidationError, json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None


def register_explainer(provider: str, parser: RowParser) -> None:
    """Register a new LLM explainer provider.

    Parameters
    ----------
    provider : str
        Provider name (e.g., "openai", "gemini", "anthropic").
    parser : callable
        Function (row: dict) -> content_dict | None.
        Returns the parsed JSON content or None on failure.
    """
    _EXPLAINER_REGISTRY[provider] = parser


def get_explainer(provider: str) -> RowParser | None:
    """Get the row parser for a provider."""
    return _EXPLAINER_REGISTRY.get(provider)


def list_explainers() -> list[str]:
    """List registered explainer provider names."""
    return list(_EXPLAINER_REGISTRY)


# Register built-in providers
register_explainer("openai", _parse_openai_row)
register_explainer("gemini", _parse_gemini_row)
register_explainer("anthropic", _parse_anthropic_row)
