"""
wds_schema_validator.py
=======================
Validator to ensure all information from the original WDS JSON is preserved
in the mapped document schema output.

Usage:
  from wds_schema_validator import validate_wds_to_schema

  result = validate_wds_to_schema(wds_doc, mapped_schema)
  if not result["valid"]:
      for m in result["missing"]:
          print(f"Missing: {m['field']} = {m['value']}")
"""

from __future__ import annotations

import json
import re
from typing import Any


def _normalize(value: Any) -> str:
    """Convert value to normalized string for comparison."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_normalize(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_normalize(v) for v in value.values())
    return str(value)


def _extract_leaf_values(obj: Any, out: set[str]) -> None:
    """Recursively extract all string/numeric leaf values into out."""
    if obj is None:
        return
    if isinstance(obj, (str, int, float, bool)):
        s = _normalize(obj)
        if s:
            out.add(s)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _extract_leaf_values(item, out)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _extract_leaf_values(v, out)
        return


def _extract_wds_values(value: Any) -> set[str]:
    """Extract comparable string values from a WDS field value."""
    out: set[str] = set()
    if value is None:
        return out
    if isinstance(value, str):
        # Split comma-separated values
        for part in value.split(","):
            s = _normalize(part)
            if s:
                out.add(s)
        if not out and value.strip():
            out.add(_normalize(value))
        return out
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, dict):
                for sub in v.values():
                    out.update(_extract_wds_values(sub))
            else:
                out.update(_extract_wds_values(v))
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            out.update(_extract_wds_values(item))
        return out
    out.add(_normalize(value))
    return out


def _is_empty(value: Any) -> bool:
    """Return True if value is considered empty."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


# WDS fields that are stored in additional (explicit or catch-all)
_ADDITIONAL_FIELDS = {
    "action", "projectid_sort", "docm_id", "ml_repnme", "origu", "owner",
    "projectid", "projn", "lndinstr_exact", "prdln_exact",
    "entityids", "historic_topic", "totvolnb",
}

# Known value transformations: (wds_field, wds_value) -> expected schema values
# Used to verify transformed values are correctly ported (e.g. countrycode "1W" -> "WLD")
_VALUE_TRANSFORMATIONS: dict[tuple[str, str], set[str]] = {
    ("countrycode", "1W"): {"WLD", "World"},
    ("countrycode", "1w"): {"WLD", "World"},
}
# count/ref_country: region names map to codes
for _region, _code in [
    ("World", "WLD"),
    ("Africa", "AFR"),
    ("East Asia and Pacific", "EAP"),
    ("Europe and Central Asia", "ECA"),
    ("Latin America and Caribbean", "LAC"),
    ("Middle East and North Africa", "MNA"),
    ("South Asia", "SAR"),
    ("North America", "NAR"),
]:
    _VALUE_TRANSFORMATIONS[("count", _region)] = {_region, _code}

# Language name -> code (common)
for _name, _code in [
    ("english", "en"), ("arabic", "ar"), ("chinese", "zh"), ("french", "fr"),
    ("spanish", "es"), ("russian", "ru"), ("portuguese", "pt"), ("japanese", "ja"),
]:
    _VALUE_TRANSFORMATIONS[("lang", _name)] = {_name.capitalize(), _code}
    _VALUE_TRANSFORMATIONS[("available_in", _name)] = {_name.capitalize(), _code}

# Mapping: wds_key -> additional key(s) where it may appear
_ADDITIONAL_KEY_MAP = {
    "action": ["additional.wds_action"],
    "projectid_sort": ["additional.wds_projectid_sort"],
    "docm_id": ["additional.wds_docm_id"],
    "ml_repnme": ["additional.wds_ml_repnme"],
    "origu": ["additional.wds_origin_unit"],
    "owner": ["additional.wds_owner"],
    "projectid": ["additional.project_id"],
    "projn": ["additional.project_name"],
    "lndinstr_exact": ["additional.wds_lndinstr_exact"],
    "prdln_exact": ["additional.wds_prdln_exact"],
    "entityids": ["additional.wds_entityids"],
    "historic_topic": ["additional.wds_historic_topic"],
    "totvolnb": ["additional.wds_totvolnb"],
}


def validate_wds_to_schema(
    wds_doc: dict[str, Any],
    mapped_schema: dict[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Validate that all non-empty WDS fields are preserved in the mapped schema.

    Args:
        wds_doc: Original document from WDS API (documents.<key>).
        mapped_schema: Output from wds_to_schema().
        strict: If True, also verify values match; if False, only check presence.

    Returns:
        Dict with:
        - valid: bool
        - missing: list of {field, value, reason}
        - in_additional: list of fields verified in additional block
        - in_schema: list of fields verified in schema elements
        - messages: list of info/error strings
    """
    result: dict[str, Any] = {
        "valid": True,
        "missing": [],
        "in_additional": [],
        "in_schema": [],
        "messages": [],
    }

    additional = mapped_schema.get("additional", {})
    schema_values: set[str] = set()
    _extract_leaf_values(mapped_schema, schema_values)

    for wds_key, wds_val in wds_doc.items():
        if _is_empty(wds_val):
            continue

        wds_values = _extract_wds_values(wds_val)

        # 1. Check additional block
        if wds_key in _ADDITIONAL_FIELDS:
            keys_to_check = _ADDITIONAL_KEY_MAP.get(wds_key, [f"additional.wds_{wds_key}"])
            found = False
            for add_key in keys_to_check:
                if add_key in additional:
                    add_val = additional[add_key]
                    if strict:
                        add_vals = _extract_wds_values(add_val)
                        if wds_values <= add_vals or wds_values <= schema_values:
                            found = True
                            break
                    else:
                        found = True
                        break
            if found:
                result["in_additional"].append(wds_key)
            else:
                result["valid"] = False
                result["missing"].append({
                    "field": wds_key,
                    "value": wds_val,
                    "reason": f"Expected in additional block (keys: {keys_to_check})",
                })
            continue

        # 2. Catch-all: check additional.wds_{key}
        catch_all_key = f"additional.wds_{wds_key.replace('.', '_')}"
        if catch_all_key in additional:
            result["in_additional"].append(wds_key)
            continue

        # 3. Check schema elements (value presence, including transformations)
        if wds_values and schema_values:
            found = False
            missing_vals: list[str] = []
            for wv in wds_values:
                # Literal match
                if wv in schema_values:
                    found = True
                    break
                # Check known transformations (e.g. countrycode "1W" -> "WLD")
                trans_key = (wds_key, wv)
                trans_key_lower = (wds_key, wv.lower())
                expected = _VALUE_TRANSFORMATIONS.get(trans_key) or _VALUE_TRANSFORMATIONS.get(trans_key_lower)
                if expected and (expected & schema_values):
                    found = True
                    break
                # Substring match for long values (e.g. abstract text)
                if len(wv) >= 8:
                    for sv in schema_values:
                        if wv in sv or sv in wv:
                            found = True
                            break
                    if found:
                        break
                missing_vals.append(wv)
            if found:
                result["in_schema"].append(wds_key)
            else:
                result["valid"] = False
                reason = f"Value(s) not found in schema: {missing_vals[:5]}{'...' if len(missing_vals) > 5 else ''}"
                if any((wds_key, v) in _VALUE_TRANSFORMATIONS or (wds_key, v.lower()) in _VALUE_TRANSFORMATIONS for v in missing_vals):
                    reason += " (expected transformed values, e.g. 1W->WLD)"
                result["missing"].append({
                    "field": wds_key,
                    "value": wds_val,
                    "reason": reason,
                })
        else:
            result["in_schema"].append(wds_key)

    if result["valid"]:
        result["messages"].append(
            f"All {len(result['in_additional']) + len(result['in_schema'])} non-empty WDS fields verified."
        )
    else:
        result["messages"].append(
            f"{len(result['missing'])} field(s) may be missing or not properly preserved."
        )

    return result


def validate_and_raise(
    wds_doc: dict[str, Any],
    mapped_schema: dict[str, Any],
    *,
    strict: bool = False,
) -> None:
    """
    Validate and raise ValueError if any WDS data is missing.
    """
    result = validate_wds_to_schema(wds_doc, mapped_schema, strict=strict)
    if not result["valid"]:
        msg = "WDS-to-schema validation failed:\n"
        for m in result["missing"]:
            msg += f"  - {m['field']}: {m['reason']}\n"
        raise ValueError(msg)


def main() -> None:
    """CLI: validate a WDS doc and its mapped schema from JSON files or stdin."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate that all WDS metadata is preserved in mapped schema."
    )
    parser.add_argument("--wds", help="Path to WDS document JSON (or API response)")
    parser.add_argument("--mapped", help="Path to mapped schema JSON")
    parser.add_argument("--strict", action="store_true", help="Strict value matching")
    args = parser.parse_args()

    if args.wds and args.mapped:
        with open(args.wds, encoding="utf-8") as f:
            data = json.load(f)
        if "documents" in data:
            doc = next(v for v in data["documents"].values() if isinstance(v, dict))
        else:
            doc = data

        with open(args.mapped, encoding="utf-8") as f:
            mapped = json.load(f)

        result = validate_wds_to_schema(doc, mapped, strict=args.strict)
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
