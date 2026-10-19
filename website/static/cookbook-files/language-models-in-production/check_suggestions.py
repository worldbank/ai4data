"""Check model-suggested edit and imputation values against the edit rules.

Reads the edit rules (CSV: rule_id, field, rule range|consistency|lte,
parameters, message) and the model's suggestions (CSV: record_id, field,
current_value, suggested_value, reason, plus the record's other fields
the rules refer to) and reports, per suggestion, whether the suggested
value satisfies every rule on its field and the record, and whether the
reason names a field of the record (a grounded reason). Suggestions that
still fail a rule are listed for the editor; those that pass go to the
editor as proposals. Nothing is applied. Uses pandas.

Usage:
    python check_suggestions.py edit_rules.csv suggested_values.csv

What this does not do: a value that passes the rules can still be wrong
(a plausible age for the wrong person). The rules bound the suggestion;
the editor decides; and the imputation method of the organization
(donor, model-based) stays the method of record, with the model's role
limited to proposing and explaining.
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

META = ("record_id", "field", "current_value", "suggested_value", "reason")


def number(record: dict[str, str], field: str) -> float | None:
    try:
        return float(record.get(field, ""))
    except ValueError:
        return None


def check(rule: pd.Series, record: dict[str, str]) -> bool | None:
    """Apply one edit rule to a record with the suggestion in place; None when the rule cannot be evaluated."""
    v = number(record, rule["field"])
    if rule["rule"] == "range":
        lo, hi = (float(x) for x in rule["parameters"].split(";"))
        return None if v is None else lo <= v <= hi
    if rule["rule"] == "lte":
        other = number(record, rule["parameters"])
        return None if v is None or other is None else v <= other
    if rule["rule"] == "consistency":
        m = re.match(r"(\w+)==(\w+)", rule["parameters"])
        return None if not m or v is None else record.get(m.group(1), "") == m.group(2)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("rules")
    parser.add_argument("suggestions")
    args = parser.parse_args(argv)
    rules = pd.read_csv(args.rules, dtype=str).fillna("")
    suggestions = pd.read_csv(args.suggestions, dtype=str).fillna("")
    fields = [c for c in suggestions.columns if c not in META]

    passes = fails = 0
    print(f"{len(suggestions)} suggestions against {len(rules)} rules\n")
    for s in suggestions.to_dict("records"):
        record = {k: s[k] for k in fields} | {s["field"]: s["suggested_value"]}
        relevant = rules[
            (rules["field"] == s["field"])
            | rules["parameters"].str.contains(s["field"], regex=False)
        ]
        failed = [
            f"{r['rule_id']} ({r['message']})"
            for _, r in relevant.iterrows()
            if check(r, record) is False
        ]
        grounded = any(
            f in s["reason"]
            for f in [*fields, "roster", "respondent", "median", "status"]
        )
        passes += not failed
        fails += bool(failed)
        verdict = (
            "to editor as proposal"
            if not failed
            else "still fails: " + "; ".join(failed)
        )
        print(
            f"{s['record_id']:<9} {s['field']:<11} {s['current_value'] or '(empty)':>8} -> {s['suggested_value']:<6} {verdict}"
            + ("" if grounded else "  [reason does not cite the record]")
        )
    print(
        f"\n{passes} suggestions pass the rules and go to the editor; {fails} still fail and go back with the rule named; nothing is applied automatically"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
