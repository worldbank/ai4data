"""Build LLM batch request files (JSONL) for OpenAI and Gemini batch APIs.

Creates provider-specific JSONL files from anomaly contexts for upload to
OpenAI Batch API or Gemini Batch API. Each line is one request.
"""

import copy
import json
from hashlib import md5, sha256
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from ai4data.anomaly.explanation.context import extract_anomaly_contexts
from ai4data.anomaly.explanation.llm_client import ENDPOINT_URLS, build_payload
from ai4data.anomaly.explanation.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    get_anomaly_response_format,
)


# Type for a row formatter: (custom_id, payload, provider, **kwargs) -> dict
RowFormatter = Callable[[str, Dict[str, Any], str, dict], Dict[str, Any]]

_BATCH_FORMATTERS: Dict[str, RowFormatter] = {}

# Anthropic Message Batches require custom_id length <= 64; keep all providers aligned.
CUSTOM_ID_MAP_SUFFIX = "_custom_id_map.json"


def _compact_custom_id(ph: str, ind_id: str, geo_id: str, ctx_str: str) -> str:
    """Stable request id ≤64 chars (provider-safe, e.g. Anthropic).

    Full (indicator, geography, context) is hashed; pair with *_custom_id_map.json
    from :func:`build_batch_file` for :func:`parse_batch_output`.
    """
    stable = f"{ph}\0{ind_id}\0{geo_id}\0{ctx_str}"
    return "a1" + sha256(stable.encode()).hexdigest()[:32]


def compact_custom_id(ph: str, ind_id: str, geo_id: str, ctx_str: str) -> str:
    """Same as :func:`_compact_custom_id` — public for migration or tooling."""
    return _compact_custom_id(ph, ind_id, geo_id, ctx_str)


def _format_row_openai(
    custom_id: str,
    payload: Dict[str, Any],
    provider: str,
    *,
    endpoint: str = "responses",
    **_: Any,
) -> Dict[str, Any]:
    """Format a batch row for OpenAI Batch API.

    Each line: {custom_id, method, url, body}.
    """
    url = ENDPOINT_URLS.get(endpoint, "/v1/responses")
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": url,
        "body": payload,
    }


def _format_row_gemini(
    custom_id: str,
    payload: Dict[str, Any],
    provider: str,
    *,
    system_prompt: str = "",
    user_prompt: str = "",
    **_: Any,
) -> Dict[str, Any]:
    """Format a batch row for Gemini Batch API.

    Each line: {key: custom_id, request: {contents, system_instruction?, generation_config}}.
    Gemini expects a GenerateContentRequest.

    Structured JSON is required so outputs match OpenAI (``text.format.json_schema``):
    without ``responseMimeType`` + ``responseJsonSchema``, the model returns prose or
    markdown-wrapped JSON with ad hoc keys—see ``WBG_Scorecard_Anomaly_Explanation``
    (Gemini batch lines built from the same schema as OpenAI).
    """
    from ai4data.anomaly.explanation.schemas import AnomalyExplanation

    contents = [{"role": "user", "parts": [{"text": user_prompt or ""}]}]
    generation_config: Dict[str, Any] = {
        "temperature": 0,
        "topP": 1,
        "maxOutputTokens": 8192,
        # CamelCase keys match Gemini REST / Colab reference notebooks.
        "responseMimeType": "application/json",
        "responseJsonSchema": AnomalyExplanation.model_json_schema(),
    }
    if "max_output_tokens" in payload:
        generation_config["maxOutputTokens"] = payload["max_output_tokens"]

    request: Dict[str, Any] = {
        "contents": contents,
        "generation_config": generation_config,
    }
    if system_prompt:
        request["system_instruction"] = {"parts": [{"text": system_prompt}]}

    return {
        "key": custom_id,
        "request": request,
    }


def _sanitize_anthropic_array_schema(node: Any) -> None:
    """Mutate JSON Schema in place for Anthropic structured output.

    Anthropic constraints on arrays:

    - ``minItems`` must be 0 or 1 (Pydantic ``min_length=2`` becomes ``minItems: 2``,
      which is rejected). We map any ``minItems`` > 1 to ``1``; Pydantic still
      validates exact lengths on parse.
    - ``maxItems`` is not supported and must be omitted.
    """
    if isinstance(node, dict):
        if node.get("type") == "array":
            mi = node.get("minItems")
            if isinstance(mi, int) and mi not in (0, 1):
                node["minItems"] = 1
            node.pop("maxItems", None)
        for v in node.values():
            _sanitize_anthropic_array_schema(v)
    elif isinstance(node, list):
        for x in node:
            _sanitize_anthropic_array_schema(x)


def anthropic_compatible_anomaly_json_schema() -> Dict[str, Any]:
    """JSON Schema for :class:`~ai4data.anomaly.explanation.schemas.AnomalyExplanation` valid for Anthropic batch ``output_config``."""
    from ai4data.anomaly.explanation.schemas import AnomalyExplanation

    schema = copy.deepcopy(AnomalyExplanation.model_json_schema())
    _sanitize_anthropic_array_schema(schema)
    return schema


def _format_row_anthropic(
    custom_id: str,
    payload: Dict[str, Any],
    provider: str,
    *,
    model_id: str = "claude-sonnet-4-6",
    system_prompt: str = "",
    user_prompt: str = "",
    **_: Any,
) -> Dict[str, Any]:
    """Format a batch row for Anthropic Message Batches (JSONL: custom_id + params)."""
    schema = anthropic_compatible_anomaly_json_schema()
    params: Dict[str, Any] = {
        "model": model_id,
        "max_tokens": 8192,
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        },
    }
    return {"custom_id": custom_id, "params": params}


def register_batch_formatter(provider: str, formatter: RowFormatter) -> None:
    """Register a batch row formatter for a provider."""
    _BATCH_FORMATTERS[provider] = formatter


def get_batch_formatter(provider: str) -> Optional[RowFormatter]:
    """Get the batch formatter for a provider."""
    return _BATCH_FORMATTERS.get(provider)


def list_batch_providers() -> List[str]:
    """List providers with registered batch formatters."""
    return list(_BATCH_FORMATTERS)


# Register built-in formatters
register_batch_formatter("openai", _format_row_openai)
register_batch_formatter("gemini", _format_row_gemini)
register_batch_formatter("anthropic", _format_row_anthropic)


def _iter_batch_rows(
    shortlist: pd.DataFrame,
    source_df: pd.DataFrame,
    geography_name_map: Dict[str, str],
    indicator_name_map: Dict[str, str],
    *,
    provider: str = "openai",
    model_id: str = "gpt-4.1-mini",
    endpoint: str = "responses",
    period_window: int = 3,
    min_outlier_count: int = 3,
    prompt_hash: Optional[str] = None,
    id_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Iterator[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Yield (custom_id, payload, formatter_kwargs) for each context."""
    formatter = get_batch_formatter(provider)
    if formatter is None:
        raise ValueError(
            f"Unknown batch provider '{provider}'. "
            f"Available: {list_batch_providers()}. "
            "Use register_batch_formatter() to add custom providers."
        )

    response_format = get_anomaly_response_format()
    from jinja2 import Template

    user_template = Template(USER_PROMPT_TEMPLATE)

    ph = (
        prompt_hash
        or md5((SYSTEM_PROMPT + "\n" + USER_PROMPT_TEMPLATE).encode()).hexdigest()[:8]
    )

    for _, row in shortlist.iterrows():
        ind_id = row["indicator_id"]
        geo_id = row["geography_id"]
        # Use list of tuple to always get DataFrame (single row would otherwise return Series)
        series_df = source_df.loc[[(ind_id, geo_id)]].reset_index()
        contexts = extract_anomaly_contexts(
            series_df,
            geography_name_map=geography_name_map,
            indicator_name_map=indicator_name_map,
            period_window=period_window,
            min_outlier_count=min_outlier_count,
        )
        for ctx in contexts:
            ctx_str = json.dumps(ctx, indent=2)
            user_prompt = user_template.render(INPUT_SERIES_INFO=ctx_str)
            custom_id = _compact_custom_id(ph, ind_id, geo_id, ctx_str)
            if id_map is not None:
                id_map[custom_id] = {
                    "indicator_id": ind_id,
                    "geography_id": geo_id,
                }

            if provider == "anthropic":
                payload: Dict[str, Any] = {}
            else:
                payload = build_payload(
                    endpoint=endpoint,
                    model_id=model_id,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    response_format=response_format,
                    with_search=False,
                )

            formatter_kwargs = {
                "endpoint": endpoint,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "model_id": model_id,
            }
            yield custom_id, payload, formatter_kwargs


def build_batch_file(
    output_path: str | Path,
    shortlist: pd.DataFrame,
    source_df: pd.DataFrame,
    geography_name_map: Dict[str, str],
    indicator_name_map: Dict[str, str],
    *,
    provider: str = "openai",
    model_id: Optional[str] = None,
    endpoint: str = "responses",
    period_window: int = 3,
    min_outlier_count: int = 3,
) -> Path:
    """Build a JSONL batch file for LLM inference.

    Parameters
    ----------
    output_path : str or Path
        Path to write the JSONL file.
    shortlist : pd.DataFrame
        DataFrame with indicator_id and geography_id columns (output of anomaly ranking).
    source_df : pd.DataFrame
        Canonical DataFrame indexed by (indicator_id, geography_id).
    geography_name_map : dict
        Maps geography_id -> geography_name.
    indicator_name_map : dict
        Maps indicator_id -> indicator_name.
    provider : str
        One of "openai", "gemini", or "anthropic". Determines the batch row format.
    model_id : str, optional
        Model identifier. Defaults: ``gpt-4.1-mini`` (OpenAI), ``gemini-2.5-flash``
        (Gemini), ``claude-sonnet-4-6`` (Anthropic).
    endpoint : str
        API endpoint (OpenAI: "responses" or "completions").
    period_window : int
        Context window size for extract_anomaly_contexts.
    min_outlier_count : int
        Minimum outlier count for extract_anomaly_contexts.

    Returns
    -------
    tuple of (Path, int)
        Path to the written JSONL file and number of requests written.
        A sidecar ``{stem}_custom_id_map.json`` is always written next to it so
        :func:`parse_batch_output` can resolve compact ``custom_id`` strings
        (required e.g. for Anthropic's 64-character limit).
    """
    if model_id is None:
        _defaults = {
            "openai": "gpt-4.1-mini",
            "gemini": "gemini-2.5-flash",
            "anthropic": "claude-sonnet-4-6",
        }
        model_id = _defaults.get(provider, "gpt-4.1-mini")

    formatter = get_batch_formatter(provider)
    if formatter is None:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {list_batch_providers()}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    id_map: Dict[str, Dict[str, str]] = {}
    count = 0
    with open(output_path, "w") as f:
        for custom_id, payload, formatter_kwargs in _iter_batch_rows(
            shortlist,
            source_df,
            geography_name_map,
            indicator_name_map,
            provider=provider,
            model_id=model_id,
            endpoint=endpoint,
            period_window=period_window,
            min_outlier_count=min_outlier_count,
            id_map=id_map,
        ):
            row = formatter(custom_id, payload, provider, **formatter_kwargs)
            f.write(json.dumps(row) + "\n")
            count += 1

    map_path = output_path.with_name(output_path.stem + CUSTOM_ID_MAP_SUFFIX)
    map_path.write_text(json.dumps(id_map, indent=2), encoding="utf-8")

    return output_path, count
