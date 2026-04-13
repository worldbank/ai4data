"""Feedback system for anomaly explanation reviewers.

Schema and storage for reviewer feedback on LLM-generated explanations.
Uses (indicator_code, geography_code, window_str) as stable key for lookups.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Facet keys: orthogonal quality dimensions
FACET_KEYS: tuple[str, ...] = (
    "anomaly_validity",
    "classification",
    "explanation",
    "evidence",
)

# Allowed rating labels per facet cell (per explainer)
RATING_VALUES: tuple[str, ...] = (
    "correct",
    "partially_correct",
    "incorrect",
    "not_applicable",
    "unsure",
)

OVERALL_BASIS_VALUES: tuple[str, ...] = ("explicit", "derived")

# Feedback schema for app integration (see docs for semantics)
FEEDBACK_SCHEMA: Dict[str, Any] = {
    "item_id": "int (index into review items)",
    "indicator_code": "str",
    "geography_code": "str",
    "window_str": "str",
    "verdict": "approved | rejected | needs_review (QA gate on the item, not 'all facets perfect')",
    "comment": "str (optional)",
    "suggested_classification": "str (optional, if classification facet wrong)",
    "facets": "dict[str, dict[str, str]] — facet_name -> explainer_name -> rating",
    "reference_explainer": "str (optional) — primary model for audit / training",
    "best_explainer": "str (optional) — closest overall when models disagree",
    "overall_basis": "explicit | derived (optional)",
    "timestamp": "ISO8601",
}

DEFAULT_FEEDBACK_FILENAME = "anomaly_feedback.json"

# In-memory store (persist to JSON file for production)
_feedback_store: List[Dict[str, Any]] = []
_feedback_path: Optional[Path] = None


def _feedback_key(entry: Dict[str, Any]) -> tuple:
    """Stable key for matching feedback to an anomaly item."""
    return (
        entry.get("indicator_code", ""),
        entry.get("geography_code", ""),
        entry.get("window_str", ""),
    )


def _normalize_loaded_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure newer fields exist for backward compatibility with old JSON files."""
    out = dict(entry)
    if not isinstance(out.get("facets"), dict):
        out["facets"] = {}
    # Normalize nested facet dicts to str -> str
    facets: Dict[str, Any] = {}
    for fk, inner in out["facets"].items():
        if fk not in FACET_KEYS:
            continue
        if not isinstance(inner, dict):
            continue
        facets[fk] = {str(k): str(v) for k, v in inner.items() if v is not None}
    out["facets"] = facets
    for opt in ("reference_explainer", "best_explainer", "overall_basis"):
        if opt not in out:
            out[opt] = ""
        elif out[opt] is None:
            out[opt] = ""
    return out


def validate_facets(
    facets: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Validate and normalize facets: only known facet keys and rating values."""
    if not facets:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for facet_key, per_explainer in facets.items():
        if facet_key not in FACET_KEYS:
            continue
        if not isinstance(per_explainer, dict):
            continue
        inner: Dict[str, str] = {}
        for explainer_name, rating in per_explainer.items():
            name = str(explainer_name).strip()
            if not name:
                continue
            r = str(rating).strip()
            if r not in RATING_VALUES:
                raise ValueError(
                    f"Invalid rating for {facet_key}/{name!r}: {rating!r}; "
                    f"expected one of {RATING_VALUES}",
                )
            inner[name] = r
        if inner:
            out[facet_key] = inner
    return out


def validate_overall_basis(value: Optional[str]) -> str:
    if not value or not str(value).strip():
        return "explicit"
    v = str(value).strip()
    if v not in OVERALL_BASIS_VALUES:
        raise ValueError(f"overall_basis must be one of {OVERALL_BASIS_VALUES}, got {v!r}")
    return v


def _persist_store() -> None:
    """Write feedback store to file if path is set."""
    global _feedback_path, _feedback_store
    if _feedback_path:
        _feedback_path.parent.mkdir(parents=True, exist_ok=True)
        _feedback_path.write_text(
            json.dumps(_feedback_store, indent=2, default=str),
            encoding="utf-8",
        )


def init_feedback_store(path: str | Path | None = None) -> None:
    """Initialize feedback storage, loading from file if it exists.

    If path is None, uses DEFAULT_FEEDBACK_FILENAME in current working directory,
    so feedback is always persisted by default.
    """
    global _feedback_store, _feedback_path
    _feedback_path = Path(path) if path is not None else Path.cwd() / DEFAULT_FEEDBACK_FILENAME
    _feedback_store = []
    if _feedback_path.exists():
        try:
            raw = json.loads(_feedback_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                _feedback_store = []
            else:
                _feedback_store = [_normalize_loaded_entry(e) for e in raw if isinstance(e, dict)]
        except Exception:
            _feedback_store = []


def submit_feedback(
    item_id: int,
    indicator_code: str,
    geography_code: str,
    window_str: str,
    verdict: str,
    comment: Optional[str] = None,
    suggested_classification: Optional[str] = None,
    facets: Optional[Dict[str, Any]] = None,
    reference_explainer: Optional[str] = None,
    best_explainer: Optional[str] = None,
    overall_basis: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit reviewer feedback for an anomaly item.

    Uses upsert by (indicator_code, geography_code, window_str): if feedback
    already exists for this anomaly, it is updated; otherwise a new entry is appended.
    """
    if verdict not in ("approved", "rejected", "needs_review"):
        raise ValueError("verdict must be approved, rejected, or needs_review")

    facets_clean = validate_facets(facets)
    basis_clean = validate_overall_basis(overall_basis)

    key = (indicator_code, geography_code, window_str)
    entry = {
        "item_id": item_id,
        "indicator_code": indicator_code,
        "geography_code": geography_code,
        "window_str": window_str,
        "verdict": verdict,
        "comment": comment or "",
        "suggested_classification": suggested_classification or "",
        "facets": facets_clean,
        "reference_explainer": (reference_explainer or "").strip(),
        "best_explainer": (best_explainer or "").strip(),
        "overall_basis": basis_clean,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for i, existing in enumerate(_feedback_store):
        if _feedback_key(existing) == key:
            _feedback_store[i] = entry
            _persist_store()
            return entry

    _feedback_store.append(entry)
    _persist_store()
    return entry


def get_feedback(
    item_id: Optional[int] = None,
    indicator_code: Optional[str] = None,
    geography_code: Optional[str] = None,
    window_str: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve feedback, optionally filtered."""
    out = _feedback_store
    if item_id is not None:
        out = [e for e in out if e["item_id"] == item_id]
    if indicator_code is not None:
        out = [e for e in out if e["indicator_code"] == indicator_code]
    if geography_code is not None:
        out = [e for e in out if e["geography_code"] == geography_code]
    if window_str is not None:
        out = [e for e in out if e["window_str"] == window_str]
    return out


def get_feedback_for_item(
    indicator_code: str,
    geography_code: str,
    window_str: str,
) -> Optional[Dict[str, Any]]:
    """Get the most recent feedback for an anomaly item by stable key."""
    matches = get_feedback(
        indicator_code=indicator_code,
        geography_code=geography_code,
        window_str=window_str,
    )
    return matches[-1] if matches else None


def _row_for_csv(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested facets for CSV consumers."""
    row = {k: v for k, v in entry.items() if k != "facets"}
    facets = entry.get("facets")
    row["facets_json"] = json.dumps(facets if isinstance(facets, dict) else {}, sort_keys=True)
    row["stable_key"] = "|".join(
        [
            str(entry.get("indicator_code", "")),
            str(entry.get("geography_code", "")),
            str(entry.get("window_str", "")),
        ],
    )
    return row


def export_feedback_csv(path: str | Path) -> Path:
    """Export all feedback to CSV for downstream use."""
    import pandas as pd

    path = Path(path)
    rows = [_row_for_csv(e) for e in _feedback_store]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path
