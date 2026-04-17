"""Parse LLM batch output into structured anomaly DataFrames."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ai4data.anomaly.explanation.batch_builder import CUSTOM_ID_MAP_SUFFIX
from ai4data.anomaly.explanation.explainers import (
    get_explainer,
    list_explainers,
    register_explainer,
)


def _resolve_custom_id_map_path(
    output_path: Path,
    explicit: Optional[Path] = None,
) -> Optional[Path]:
    """Locate sidecar JSON written by :func:`build_batch_file` next to batch input.

    Downloaded batch results are often named ``{stem}_out.jsonl``; the map is
    ``{stem}_custom_id_map.json`` (without ``_out``).
    """
    if explicit is not None:
        p = Path(explicit)
        return p if p.exists() else None
    stem = output_path.stem
    if stem.endswith("_out"):
        base = stem[:-4]
        candidate = output_path.with_name(f"{base}{CUSTOM_ID_MAP_SUFFIX}")
        if candidate.exists():
            return candidate
    candidate = output_path.with_name(f"{stem}{CUSTOM_ID_MAP_SUFFIX}")
    if candidate.exists():
        return candidate
    return None


def parse_batch_output(
    output_path: str | Path,
    provider: str,
    indicator_name_map: Dict[str, str],
    geography_name_map: Dict[str, str],
    custom_id_separator: str = "-",
    custom_id_parts: tuple = (0, 2, 3),  # (prefix, indicator_idx, geography_idx)
    custom_id_map_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Parse JSONL batch output into an anomaly explanation DataFrame.

    Parameters
    ----------
    output_path : str or Path
        Path to the JSONL file (OpenAI or Gemini batch output format).
    provider : str
        One of "openai", "gemini", or "anthropic".
    indicator_name_map : dict
        Maps indicator_id -> indicator_name.
    geography_name_map : dict
        Maps geography_id -> geography_name.
    custom_id_separator : str
        Separator in custom_id (e.g., "nosearch-c660ac92-INDICATOR-GEO-hash").
    custom_id_parts : tuple
        Indices for (prefix, indicator_code, country_code) in split(custom_id_separator).
        Used only for **legacy** ``nosearch-...`` ids when no map is available.
    custom_id_map_path : Path, optional
        JSON dict ``custom_id -> {indicator_id, geography_id}`` from
        :func:`build_batch_file`. If omitted, looks next to ``output_path`` for
        ``{stem}_custom_id_map.json``, or ``{stem_without_out}_custom_id_map.json``
        when the file name ends with ``_out``.

    Returns
    -------
    pd.DataFrame
        One row per anomaly with columns: custom_id, indicator_code, indicator,
        country_code, country, window, is_anomaly, classification, confidence,
        explanation, evidence_strength, evidence_source, source, window_str.
    """
    parser = get_explainer(provider)
    if parser is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available: {list_explainers()}. "
            "Use register_explainer() to add custom providers."
        )

    out_path = Path(output_path)
    map_path = _resolve_custom_id_map_path(out_path, custom_id_map_path)
    id_map: Dict[str, Any] = {}
    if map_path is not None:
        id_map = json.loads(map_path.read_text(encoding="utf-8"))

    out_df = pd.read_json(output_path, lines=True)
    anomalies: List[Dict[str, Any]] = []

    for _, row in out_df.iterrows():
        row = row.copy()
        if provider == "gemini":
            row["custom_id"] = row.get("key", row.get("custom_id", ""))

        custom_id = str(row["custom_id"])
        if custom_id in id_map:
            indicator_code = id_map[custom_id]["indicator_id"]
            country_code = id_map[custom_id]["geography_id"]
        else:
            parts = custom_id.split(custom_id_separator)
            if len(parts) < max(custom_id_parts) + 1:
                continue
            indicator_code = parts[custom_id_parts[1]]
            country_code = parts[custom_id_parts[2]]

        content = parser(row)
        if content is None:
            continue

        for anomaly in content.get("anomalies", []):
            anomalies.append({
                "custom_id": custom_id,
                "indicator_code": indicator_code,
                "indicator": indicator_name_map.get(indicator_code, indicator_code),
                "country_code": country_code,
                "country": geography_name_map.get(country_code, country_code),
                **anomaly,
            })

    if not anomalies:
        return pd.DataFrame()

    df = pd.DataFrame(anomalies)

    def _window_to_str(x: Any) -> str:
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            return f"{x[0]}-{x[1]}"
        return ""

    if "window" in df.columns:
        df["window_str"] = df["window"].apply(_window_to_str)
    else:
        df["window_str"] = ""

    return df
