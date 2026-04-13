"""Heuristic and LLM-assisted suggestions for canonical column mappings."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Sequence

from ai4data.anomaly.explanation.adapters import REQUIRED_CANONICAL_COLUMNS

# Lowercase forms (no punctuation) -> canonical key; first match wins per canonical key order.
_HEURISTIC_ALIASES: Dict[str, str] = {
    # indicator
    "indicator": "indicator_id",
    "indicatorcode": "indicator_id",
    "indicator_code": "indicator_id",
    "ind": "indicator_id",
    "indicatorid": "indicator_id",
    # indicator name
    "indicatorname": "indicator_name",
    "indicator_name": "indicator_name",
    "indicatorlabel": "indicator_name",
    "indicator_label": "indicator_name",
    # geography
    "countrycode": "geography_id",
    "country_code": "geography_id",
    "refarea": "geography_id",
    "ref_area": "geography_id",
    "geographyid": "geography_id",
    "geography_id": "geography_id",
    "geo_id": "geography_id",
    "iso": "geography_id",
    "iso3": "geography_id",
    # geography name
    "countryname": "geography_name",
    "country_name": "geography_name",
    "refarealabel": "geography_name",
    "ref_area_label": "geography_name",
    "geographyname": "geography_name",
    "geography_name": "geography_name",
    # period
    "year": "period",
    "period": "period",
    "time": "period",
    "date": "period",
    # value
    "value": "value",
    # imputed
    "imputed": "is_imputed",
    "is_imputed": "is_imputed",
    "isimputed": "is_imputed",
    # anomaly score
    "abszscore": "anomaly_score",
    "abszscore_zscore": "anomaly_score",
    "abszscorezscore": "anomaly_score",
    "zscore": "anomaly_score",
    "anomalyscore": "anomaly_score",
    "anomaly_score": "anomaly_score",
    "score": "anomaly_score",
    # outlier count
    "outlierindicatortotal": "outlier_count",
    "outlier_indicator_total": "outlier_count",
    "outliercount": "outlier_count",
    "outlier_count": "outlier_count",
    # optional freq
    "freq": "freq",
    "frequency": "freq",
}


def _normalize_header(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def suggest_column_mapping(headers: Sequence[str]) -> Dict[str, str]:
    """Suggest canonical→source column mapping from CSV/header names (heuristic only).

    Matches common World Bank / Scorecard / WDI-style names. Review and edit
    the result before calling :class:`ConfigurableAdapter`.

    Parameters
    ----------
    headers : sequence of str
        Column names from the file (e.g. from ``pd.read_csv(path, nrows=0).columns``).

    Returns
    -------
    dict
        Keys are canonical names present in ``REQUIRED_CANONICAL_COLUMNS`` (and ``freq`` if
        matched). Values are **source** column names from the file.
    """
    # Map normalized -> original header (first occurrence wins)
    norm_to_orig: Dict[str, str] = {}
    for h in headers:
        n = _normalize_header(h)
        if n not in norm_to_orig:
            norm_to_orig[n] = h

    # Exact alias hits on normalized names
    canonical_to_source: Dict[str, str] = {}
    for n, orig in norm_to_orig.items():
        canon = _HEURISTIC_ALIASES.get(n)
        if canon and canon in REQUIRED_CANONICAL_COLUMNS + ["freq"]:
            if canon not in canonical_to_source:
                canonical_to_source[canon] = orig

    # Secondary: substring contains for indicator / country (if still missing)
    headers_lower = {h.lower(): h for h in headers}
    if "indicator_id" not in canonical_to_source:
        for key in ("indicator.code", "indicator_code"):
            if key in headers_lower:
                canonical_to_source["indicator_id"] = headers_lower[key]
                break
    if "geography_id" not in canonical_to_source:
        for key in ("country.code", "country_code"):
            if key in headers_lower:
                canonical_to_source["geography_id"] = headers_lower[key]
                break

    return canonical_to_source


def suggest_column_mapping_with_llm(
    headers: Sequence[str],
    *,
    provider: str = "openai",
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """Use an LLM once to propose canonical→source column names (review before use).

    Parameters
    ----------
    headers : sequence of str
        Column names from the CSV.
    provider : str
        ``"openai"`` or ``"gemini"``.
    model_id : str, optional
        Defaults: ``gpt-4.1-mini`` (OpenAI) or ``gemini-2.5-flash`` (Gemini).
    api_key : str, optional
        API key; falls back to environment variables.

    Returns
    -------
    dict
        Mapping from canonical column name to **source** column string, only for keys
        the model could assign. Omitted keys must be filled manually.
    """
    provider = provider.lower().strip()
    if provider not in ("openai", "gemini"):
        raise ValueError("provider must be 'openai' or 'gemini'")

    required = list(REQUIRED_CANONICAL_COLUMNS)
    schema_desc = ", ".join(required)
    header_list = list(headers)
    prompt = f"""You map CSV columns to a fixed canonical schema for anomaly timeseries data.

Canonical keys (each must map to exactly one column from the file, or be null if impossible):
{schema_desc}

CSV columns (use these strings exactly when they match):
{json.dumps(header_list, indent=2)}

Respond with a single JSON object whose keys are ONLY those canonical names and whose values are either the exact source column name from the list above or null."""

    if provider == "openai":
        return _llm_mapping_openai(prompt, model_id=model_id or "gpt-4.1-mini", api_key=api_key)
    return _llm_mapping_gemini(prompt, model_id=model_id or "gemini-2.5-flash", api_key=api_key)


def _llm_mapping_openai(prompt: str, *, model_id: str, api_key: Optional[str]) -> Dict[str, str]:
    import os

    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model_id,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You output only valid JSON objects. No markdown.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content or "{}"
    raw = json.loads(text)
    return _coerce_mapping(raw)


def _llm_mapping_gemini(prompt: str, *, model_id: str, api_key: Optional[str]) -> Dict[str, str]:
    import os

    from google import genai
    from google.genai import types

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Gemini API key required (GEMINI_API_KEY or GOOGLE_API_KEY).")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    text = (resp.text or "").strip()
    raw = json.loads(text)
    return _coerce_mapping(raw)


def _coerce_mapping(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if k not in REQUIRED_CANONICAL_COLUMNS and k != "freq":
            continue
        if v is None or v == "":
            continue
        out[str(k)] = str(v)
    return out
