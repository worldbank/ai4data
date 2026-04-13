"""Helpers for legacy ``nosearch-...`` batch ``custom_id`` strings (pre–compact-id format).

New batches use :func:`compact_custom_id` plus a sidecar map. Old batches used::

    nosearch-{prompt_hash_8}-{indicator_id}-{geography_id}-{md5(context_json)}

You cannot recover ``context_json`` from the trailing MD5, so **new-style compact ids**
that match a fresh :func:`build_batch_file` run require re-extracting contexts from
your canonical data (or rebuilding the batch file). This module supports:

- Parsing **indicator** and **geography** from legacy ids (for joins and maps).
- Writing a **sidecar map file** in the new JSON shape so tools expecting
  ``*_custom_id_map.json`` can load legacy outputs (optional; :func:`parse_batch_output`
  already parses legacy ids without a map).
- Computing a **new compact id** if you still have the exact ``ctx_str`` string used
  when the batch was built.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from ai4data.anomaly.explanation.batch_builder import (
    CUSTOM_ID_MAP_SUFFIX,
    compact_custom_id,
)

_HEX32 = re.compile(r"^[0-9a-fA-F]{32}$")


def parse_legacy_nosearch_custom_id(custom_id: str) -> Optional[Dict[str, str]]:
    """Parse legacy ``custom_id`` of the form ``nosearch-{ph}-{ind}-{geo}-{md5_32}``.

    ``indicator_id`` may contain hyphens; they are reconstructed by joining all
    segments between the prompt hash and the geography code.

    Parameters
    ----------
    custom_id : str
        Legacy id from an older batch JSONL.

    Returns
    -------
    dict or None
        Keys: ``prompt_hash``, ``indicator_id``, ``geography_id``, ``context_md5_hex``,
        or ``None`` if the string does not match the legacy pattern.
    """
    if not custom_id or not custom_id.startswith("nosearch-"):
        return None
    parts = custom_id.split("-")
    if len(parts) < 5:
        return None
    if parts[0] != "nosearch":
        return None
    md5_tail = parts[-1]
    if not _HEX32.match(md5_tail):
        return None
    geo_id = parts[-2]
    ph = parts[1]
    indicator_id = "-".join(parts[2:-2])
    return {
        "prompt_hash": ph,
        "indicator_id": indicator_id,
        "geography_id": geo_id,
        "context_md5_hex": md5_tail.lower(),
    }


def new_compact_id_from_legacy_parts(
    prompt_hash: str,
    indicator_id: str,
    geography_id: str,
    context_json_str: str,
) -> str:
    """Compute the **new** compact id for the same logical request as a legacy row.

    ``context_json_str`` must be **byte-for-byte** the same string passed when the
    original batch line was built (same JSON serialization as
    ``json.dumps(ctx, indent=2)`` in :func:`build_batch_file`). If you only have a
    legacy ``custom_id``, you cannot recover this from the MD5 suffix; re-run context
    extraction from your canonical dataframe instead.

    Parameters
    ----------
    prompt_hash : str
        Eight-character prompt hash (from legacy id or current pipeline).
    indicator_id, geography_id : str
        Series identifiers.
    context_json_str : str
        Exact context JSON string used in the user prompt for that request.
    """
    return compact_custom_id(
        prompt_hash, indicator_id, geography_id, context_json_str
    )


def custom_id_map_from_legacy_batch_output_lines(
    lines: list[dict[str, Any]],
    *,
    custom_id_key: str = "custom_id",
) -> Dict[str, Dict[str, str]]:
    """Build a ``custom_id -> {indicator_id, geography_id}`` map from parsed JSONL rows.

    Reads ``custom_id`` (or Gemini ``key``) from each line, parses legacy ``nosearch-``
    ids, and skips rows that do not match (e.g. already compact ``a1...`` ids).
    """
    out: Dict[str, Dict[str, str]] = {}
    for row in lines:
        cid = row.get(custom_id_key) or row.get("key", "")
        if not cid:
            continue
        parsed = parse_legacy_nosearch_custom_id(str(cid))
        if parsed is None:
            continue
        out[str(cid)] = {
            "indicator_id": parsed["indicator_id"],
            "geography_id": parsed["geography_id"],
        }
    return out


def write_custom_id_map_from_legacy_batch_output(
    output_jsonl_path: str | Path,
    *,
    destination: Optional[str | Path] = None,
) -> Path:
    """Write ``*_custom_id_map.json`` next to a **legacy** provider output JSONL file.

    Useful when downstream tooling expects the sidecar map file. :func:`parse_batch_output`
    does not require this for legacy ``nosearch-`` ids.

    Parameters
    ----------
    output_jsonl_path : path
        Path to downloaded batch output (e.g. ``*_out.jsonl``).
    destination : path, optional
        Defaults to ``{stem}{CUSTOM_ID_MAP_SUFFIX}`` beside ``output_jsonl_path``.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    output_jsonl_path = Path(output_jsonl_path)
    rows: list[dict[str, Any]] = []
    with open(output_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    id_map = custom_id_map_from_legacy_batch_output_lines(rows)
    dest = (
        Path(destination)
        if destination is not None
        else output_jsonl_path.with_name(output_jsonl_path.stem + CUSTOM_ID_MAP_SUFFIX)
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    return dest
