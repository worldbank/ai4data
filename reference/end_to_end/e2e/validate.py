"""Step 2: validate every record in three layers before it is published.

1. Structure: the record validates against the World Bank schema for its
   type. The schemas require very little, so this layer catches malformed
   records, not incomplete ones.
2. Completeness: the fields the organization requires at the chosen
   maturity level, from a profile of dotted paths. The profile decides what
   complete means; the schema does not.
3. Validity: rules named in the profile: dates parse, periods are ordered,
   URLs look like URLs, values come from a vocabulary, a text has a minimum
   length, placeholders such as "n/a" are absent.

A record that fails any layer is withheld from publication, and the reason
is in the report. The pinned schema files are the snapshot the cookbook
tests use, so the run needs no network.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .records import Record, first, record_id, values_at

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PERIOD = re.compile(r"^\d{4}(-\d{2}(-\d{2})?|-Q[1-4]|-W\d{2}|-S[12])?$")
URL = re.compile(r"^https?://\S+$")
PLACEHOLDERS = {"n/a", "na", "none", "-", "tbd", "unknown", "?"}


class SchemaStore:
    """The pinned schema files, loaded by file name; references resolve by basename."""

    def __init__(self, schema_dir: Path) -> None:
        self.dir = schema_dir
        self._cache: dict[str, dict] = {}

    def get(self, name: str) -> dict:
        if name not in self._cache:
            with (self.dir / name).open(encoding="utf-8") as fh:
                self._cache[name] = json.load(fh)
        return self._cache[name]


def _strip(schema: dict) -> dict:
    # The files name a metaschema that jsonschema does not know; validate as 2020-12.
    return {k: v for k, v in schema.items() if k != "$schema"}


def structure_errors(record: Record, schema_name: str, store: SchemaStore) -> list[str]:
    import jsonschema
    from referencing import Registry, Resource
    from referencing.exceptions import NoSuchResource
    from referencing.jsonschema import DRAFT202012

    def retrieve(uri: str) -> Resource:
        name = uri.rsplit("/", 1)[-1]
        try:
            contents = store.get(name)
        except FileNotFoundError as exc:
            raise NoSuchResource(ref=uri) from exc
        return Resource.from_contents(
            _strip(contents), default_specification=DRAFT202012
        )

    validator = jsonschema.Draft202012Validator(
        _strip(store.get(schema_name)), registry=Registry(retrieve=retrieve)
    )
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '(root)'}: {err.message[:100]}"
        for err in validator.iter_errors(record)
    ]


def missing_fields(record: Record, profile: dict, level: str) -> list[str]:
    levels = profile["levels"]
    order = list(levels)
    wanted = [p for lvl in order[: order.index(level) + 1] for p in levels[lvl]]
    return [p for p in wanted if first(record, p) is None]


def _strings(record: Record, path: str) -> list[str]:
    return [
        v
        for v in values_at(record, path.split("."))
        if isinstance(v, str) and v.strip()
    ]


def validity_issues(record: Record, profile: dict) -> list[str]:
    rules = profile.get("rules", {})
    issues: list[str] = []
    for path in rules.get("dates", []):
        issues += [
            f"{path} {v!r} is not YYYY-MM-DD"
            for v in _strings(record, path)
            if not DATE.match(v)
        ]
    for path in rules.get("periods", []):
        issues += [
            f"{path} {v!r} is not an ISO 8601 or SDMX period"
            for v in _strings(record, path)
            if not PERIOD.match(v)
        ]
    for start_path, end_path in rules.get("period_order", []):
        starts, ends = _strings(record, start_path), _strings(record, end_path)
        if starts and ends and starts[0] > ends[0]:
            issues.append(f"{start_path} {starts[0]!r} is after {end_path} {ends[0]!r}")
    for path in rules.get("urls", []):
        issues += [
            f"{path} {v!r} is not a URL"
            for v in _strings(record, path)
            if not URL.match(v)
        ]
    for path, allowed in rules.get("vocabularies", {}).items():
        issues += [
            f"{path} {v!r} is not in the vocabulary"
            for v in _strings(record, path)
            if v not in allowed
        ]
    for path, n in rules.get("min_words", {}).items():
        issues += [
            f"{path} has {len(v.split())} words, fewer than {n}"
            for v in _strings(record, path)
            if len(v.split()) < n
        ]
    for path in rules.get("placeholders", []):
        issues += [
            f"{path} is the placeholder {v!r}"
            for v in _strings(record, path)
            if v.strip().lower() in PLACEHOLDERS
        ]
    return issues


@dataclass
class RecordResult:
    kind: str
    idno: str
    structure: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.structure or self.missing or self.issues)

    @property
    def reasons(self) -> list[str]:
        out = [f"structure: {s}" for s in self.structure]
        out += [f"missing at this level: {m}" for m in self.missing]
        out += [f"invalid: {i}" for i in self.issues]
        return out


def validate(
    records: list[tuple[str, Record]],
    profiles: dict[str, dict],
    store: SchemaStore,
    level: str,
) -> list[RecordResult]:
    results = []
    for kind, record in records:
        profile = profiles[kind]
        result = RecordResult(kind=kind, idno=record_id(kind, record))
        result.structure = structure_errors(record, profile["schema"], store)
        result.missing = missing_fields(record, profile, level)
        result.issues = validity_issues(record, profile)
        results.append(result)
    return results


def load_profiles(fixtures: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for kind, name in (
        ("indicator", "profile_indicator.json"),
        ("microdata", "profile_microdata.json"),
    ):
        with (fixtures / name).open(encoding="utf-8") as fh:
            out[kind] = json.load(fh)
    return out


def as_dict(result: RecordResult) -> dict[str, Any]:
    return {
        "kind": result.kind,
        "idno": result.idno,
        "published": result.ok,
        "structure_errors": result.structure,
        "missing_fields": result.missing,
        "validity_issues": result.issues,
    }
