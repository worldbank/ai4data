"""Check catalog metadata in three layers, for any World Bank metadata type.

1. Structure: each record is validated against the World Bank metadata
   schema for its type (worldbank/metadata-schemas): indicator, indicators
   database, microdata (DDI Codebook), geospatial (ISO 19115/19139),
   document (Dublin Core), table, script, image, or video. The schemas
   require very little (an identifier and a title), so passing this layer
   means the record is well-formed, and nothing more.
2. Completeness: each record is checked against a profile, a JSON file in
   which the office lists the fields it requires at each maturity level,
   as dotted paths into the record, with a reason per field. The schema
   does not decide what is complete; the office does.
3. Validity: rules named in the profile: dates parse, periods are ordered,
   URLs look like URLs, values come from a vocabulary, a text field has a
   minimum length, two fields are not identical, placeholders such as
   "n/a" are absent.

What this does not check: whether the text is correct, whether a title and
an abstract describe the same thing, whether a unit matches the values, or
whether the text is specific enough to be useful. Those are semantic
checks; see the program's Generative AI for Metadata Quality work.

Input: a JSON file with one record or a list of records in the nested form
of the schema (what NADA and the Metadata Editor export), or, for flat
types such as indicators, a CSV with one row per record. CSV columns are
dotted paths (or bare names placed under the profile's "csv_root"); list
fields take ";" between values and each value fills the item's main
property, which is read from the schema.

Usage:
    python check_metadata.py example_catalog.csv profile_indicator.json
    python check_metadata.py example_microdata.json profile_microdata.json
    python check_metadata.py records.json profile_geospatial.json --level ai-ready
    python check_metadata.py records.json profile.json --schema-dir ./wb-schemas

Exit status is 0 when every record passes all three layers, 1 otherwise,
so the script can run in a scheduled job or a CI step.

Needs the `jsonschema` package (4.18 or later, which brings `referencing`)
for layer 1. Without it, layer 1 is skipped and the report says so. Schema
files are downloaded on first use into a folder next to this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RAW = "https://raw.githubusercontent.com/worldbank/metadata-schemas/main/schemas/"
TYPES = {
    "indicator": "timeseries-schema.json",
    "indicators-db": "timeseries-db-schema.json",
    "microdata": "microdata-schema.json",
    "geospatial": "geospatial-schema.json",
    "document": "document-schema.json",
    "table": "table-schema.json",
    "script": "script-schema.json",
    "image": "image-schema.json",
    "video": "video-schema.json",
}

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PERIOD = re.compile(r"^\d{4}(-\d{2}(-\d{2})?|-Q[1-4]|-W\d{2}|-S[12])?$")
URL = re.compile(r"^https?://\S+$")
IDNO = re.compile(r"^[A-Za-z0-9_.\-]+$")
PLACEHOLDERS = {"n/a", "na", "none", "-", "tbd", "unknown", "?"}

Record = dict[str, Any]
Schema = dict[str, Any]


def warn(message: str) -> None:
    print(message, file=sys.stderr)


# --- schemas ---------------------------------------------------------------


class SchemaStore:
    """Loads World Bank schema files by name, downloading them on first use.

    The type schemas reference the others by relative file name
    ("ddi-schema.json"), so every reference is resolved by basename.
    """

    def __init__(self, schema_dir: Path) -> None:
        self.dir = schema_dir
        self._cache: dict[str, Schema | None] = {}

    def get(self, name: str) -> Schema | None:
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def _load(self, name: str) -> Schema | None:
        path = self.dir / name
        if not path.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
            try:
                with urllib.request.urlopen(RAW + name, timeout=20) as resp:
                    path.write_bytes(resp.read())
            except OSError as exc:
                warn(f"could not download {name}: {exc}")
                return None
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    # Navigation through $ref (across files) and allOf.

    def deref(self, name: str, node: Schema) -> tuple[str, Schema]:
        """Follow $ref chains; returns the file name and the resolved node."""
        while isinstance(node, dict) and "$ref" in node:
            file, _, fragment = node["$ref"].partition("#")
            if file:
                name = file.rsplit("/", 1)[-1]
            target = self.get(name)
            if target is None:
                return name, {}
            node = target
            for part in fragment.split("/"):
                if part:
                    node = node[part]
        return name, node

    def child(self, name: str, node: Schema, key: str) -> tuple[str, Schema] | None:
        """The schema of property `key`, looking through allOf."""
        name, node = self.deref(name, node)
        if key in node.get("properties", {}):
            return self.deref(name, node["properties"][key])
        for sub in node.get("allOf", []):
            found = self.child(name, sub, key)
            if found:
                return found
        return None

    def node_at(self, root: str, path: str) -> tuple[str, Schema] | None:
        """The schema node for a dotted path, stepping into array items."""
        schema = self.get(root)
        if schema is None:
            return None
        name, node = self.deref(root, schema)
        for key in path.split("."):
            if node.get("type") == "array":
                name, node = self.deref(name, node.get("items", {}))
            found = self.child(name, node, key)
            if not found:
                return None
            name, node = found
        return name, node

    def is_array(self, root: str, path: str) -> bool:
        found = self.node_at(root, path)
        return bool(found) and found[1].get("type") == "array"

    def item_key(self, root: str, path: str) -> str | None:
        """For an array field: the property a bare value fills, "" for plain items, None if not an array."""
        found = self.node_at(root, path)
        if not found or found[1].get("type") != "array":
            return None
        _, item = self.deref(found[0], found[1].get("items", {}))
        if item.get("type") == "object" or "properties" in item:
            required = item.get("required") or []
            props = list(item.get("properties", {}))
            return (required or props or [""])[0]
        return ""


# --- records ---------------------------------------------------------------


@dataclass
class Column:
    """How one CSV column maps into the nested record. Computed once per file."""

    path: str
    item_key: str | None  # None: scalar; "": list of strings; "x": list of {x: value}
    array_prefixes: set[str] = field(
        default_factory=set
    )  # path prefixes that are arrays


def plan_columns(
    header: list[str], profile: dict, store: SchemaStore, root: str
) -> list[Column]:
    """Resolve each CSV column against the profile and the schema, once."""
    csv_root = profile.get("csv_root", "")
    renames = profile.get("csv_columns", {})
    # The schema's first (required) property is the default item key for a
    # list column; the profile can name another, for example "uri" for
    # reference lists whose first property is "source".
    item_keys = profile.get("csv_item_keys", {})
    plan = []
    for col in header:
        path = renames.get(col, col)
        if csv_root and not path.startswith(csv_root + "."):
            path = f"{csv_root}.{path}"
        parts = path.split(".")
        prefixes = {".".join(parts[: i + 1]) for i in range(len(parts) - 1)}
        plan.append(
            Column(
                path=path,
                item_key=item_keys.get(path, store.item_key(root, path)),
                array_prefixes={p for p in prefixes if store.is_array(root, p)},
            )
        )
    return plan


def set_path(record: Record, column: Column, value: Any) -> None:
    """Set a value at the column's path, creating the first item of any array on the way."""
    parts = column.path.split(".")
    node = record
    for i, key in enumerate(parts[:-1]):
        if ".".join(parts[: i + 1]) in column.array_prefixes:
            node = node.setdefault(key, [{}])[0]
        else:
            node = node.setdefault(key, {})
    node[parts[-1]] = value


def csv_value(column: Column, raw: str) -> Any:
    if column.item_key is None:
        return raw
    items = [v.strip() for v in raw.split(";") if v.strip()]
    if column.item_key == "":
        return items
    return [{column.item_key: v} for v in items]


def load_records(
    path: Path, profile: dict, store: SchemaStore, root: str
) -> list[Record]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else [data]
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        plan = plan_columns(reader.fieldnames or [], profile, store, root)
        records = []
        for row in reader:
            record: Record = {}
            for column, col in zip(plan, reader.fieldnames or []):
                raw = (row.get(col) or "").strip()
                if raw:
                    set_path(record, column, csv_value(column, raw))
            records.append(record)
    return records


def values_at(obj: Any, parts: list[str]) -> list[Any]:
    """All values at a dotted path, flattening through lists."""
    if not parts:
        return [obj]
    if isinstance(obj, list):
        return [v for item in obj for v in values_at(item, parts)]
    if isinstance(obj, dict) and parts[0] in obj:
        return values_at(obj[parts[0]], parts[1:])
    return []


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def present(record: Record, path: str) -> bool:
    return any(nonempty(v) for v in values_at(record, path.split(".")))


def first(record: Record, path: str) -> Any:
    return next((v for v in values_at(record, path.split(".")) if nonempty(v)), None)


def strings_at(record: Record, path: str) -> list[str]:
    return [
        v
        for v in values_at(record, path.split("."))
        if isinstance(v, str) and v.strip()
    ]


def record_id(record: Record, id_path: str, index: int) -> str:
    return str(first(record, id_path) or f"record {index + 1}")


# --- layers ----------------------------------------------------------------


@dataclass
class StructureResult:
    valid: int | None  # None when the layer could not run
    problems: list[str]


@dataclass
class CompletenessResult:
    wanted: list[str]
    missing_by_record: dict[str, list[str]]
    missing_count: Counter


def layer_structure(
    records: list[Record], store: SchemaStore, root: str, id_path: str
) -> StructureResult:
    try:
        import jsonschema
        from referencing import Registry, Resource
        from referencing.exceptions import NoSuchResource, Unresolvable
        from referencing.jsonschema import DRAFT202012
    except ImportError:
        return StructureResult(
            None, ["jsonschema 4.18+ not installed; layer 1 skipped"]
        )
    root_schema = store.get(root)
    if root_schema is None:
        return StructureResult(None, ["schema not available; layer 1 skipped"])

    def strip(schema: Schema) -> Schema:
        # The files name a metaschema that jsonschema does not know; validate as 2020-12.
        return {k: v for k, v in schema.items() if k != "$schema"}

    def retrieve(uri: str) -> Resource:
        contents = store.get(uri.rsplit("/", 1)[-1])
        if contents is None:
            raise NoSuchResource(ref=uri)
        return Resource.from_contents(
            strip(contents), default_specification=DRAFT202012
        )

    validator = jsonschema.Draft202012Validator(
        strip(root_schema), registry=Registry(retrieve=retrieve)
    )
    problems: list[str] = []
    invalid: set[str] = set()
    for i, record in enumerate(records):
        rid = record_id(record, id_path, i)
        try:
            for err in validator.iter_errors(record):
                where = "/".join(str(p) for p in err.absolute_path)
                problems.append(f"{rid}: {where}: {err.message[:90]}")
                invalid.add(rid)
        except (Unresolvable, jsonschema.exceptions.SchemaError) as exc:
            problems.append(f"{rid}: validation failed: {exc}")
            invalid.add(rid)
    return StructureResult(len(records) - len(invalid), problems)


def layer_completeness(
    records: list[Record], profile: dict, level: str, id_path: str
) -> CompletenessResult:
    levels = profile["levels"]
    order = list(levels)
    wanted = [path for lvl in order[: order.index(level) + 1] for path in levels[lvl]]
    missing_by_record: dict[str, list[str]] = {}
    missing_count: Counter = Counter()
    for i, record in enumerate(records):
        missing = [path for path in wanted if not present(record, path)]
        if missing:
            missing_by_record[record_id(record, id_path, i)] = missing
            missing_count.update(missing)
    return CompletenessResult(wanted, missing_by_record, missing_count)


def layer_validity(records: list[Record], profile: dict, id_path: str) -> list[str]:
    rules = profile.get("rules", {})
    issues: list[str] = []
    seen: Counter = Counter()
    for i, record in enumerate(records):
        rid = record_id(record, id_path, i)
        seen[rid] += 1
        if first(record, id_path) and not IDNO.match(rid):
            issues.append(f"{rid}: identifier has characters outside A-Z, 0-9, _ . -")
        for path in rules.get("dates", []):
            issues += [
                f"{rid}: {path} {v!r} is not YYYY-MM-DD"
                for v in strings_at(record, path)
                if not DATE.match(v)
            ]
        for path in rules.get("periods", []):
            issues += [
                f"{rid}: {path} {v!r} is not an ISO 8601 or SDMX period"
                for v in strings_at(record, path)
                if not PERIOD.match(v)
            ]
        for start_path, end_path in rules.get("period_order", []):
            start, end = first(record, start_path), first(record, end_path)
            ordered_pair = (
                isinstance(start, str)
                and isinstance(end, str)
                and PERIOD.match(start)
                and PERIOD.match(end)
            )
            if ordered_pair and end < start:
                issues.append(
                    f"{rid}: {end_path} ({end}) is before {start_path} ({start})"
                )
        for path in rules.get("urls", []):
            issues += [
                f"{rid}: {path} {v!r} is not a URL"
                for v in strings_at(record, path)
                if not URL.match(v)
            ]
        for path, allowed in rules.get("vocabularies", {}).items():
            issues += [
                f"{rid}: {path} {v!r} is not in the profile vocabulary"
                for v in strings_at(record, path)
                if v not in allowed
            ]
        for path, n in rules.get("min_words", {}).items():
            issues += [
                f"{rid}: {path} has fewer than {n} words"
                for v in strings_at(record, path)
                if len(v.split()) < n
            ]
        for a, b in rules.get("not_equal", []):
            va, vb = first(record, a), first(record, b)
            if (
                isinstance(va, str)
                and isinstance(vb, str)
                and va.strip().lower() == vb.strip().lower()
            ):
                issues.append(f"{rid}: {b} repeats {a}")
        for path in rules.get("placeholders", []):
            issues += [
                f"{rid}: {path} is a placeholder ({v!r})"
                for v in strings_at(record, path)
                if v.strip().lower() in PLACEHOLDERS
            ]
    issues += [
        f"{rid}: identifier appears {n} times" for rid, n in seen.items() if n > 1
    ]
    return issues


# --- report ----------------------------------------------------------------


def report(
    *,
    dtype: str,
    root: str,
    n: int,
    level: str,
    profile: dict,
    structure: StructureResult,
    completeness: CompletenessResult,
    issues: list[str],
) -> None:
    print(f"Layer 1  Structure (World Bank {root})")
    if structure.valid is None:
        print(f"         {structure.problems[0]}")
    else:
        print(
            f"         {structure.valid} of {n} records are valid instances of the schema"
        )
        for problem in structure.problems[:20]:
            print(f"         {problem}")
    print(
        "         Note: the schemas require little more than an identifier and a title. Validity is not completeness.\n"
    )

    complete = n - len(completeness.missing_by_record)
    print(
        f"Layer 2  Completeness against profile {profile.get('name', dtype)!r}, "
        f"level {level!r} ({len(completeness.wanted)} fields)"
    )
    print(f"         {complete} of {n} records complete")
    reasons = profile.get("fields", {})
    for path, count in completeness.missing_count.most_common():
        print(f"         missing {path:<48} in {count} record(s)")
        if path in reasons:
            print(f"                 why: {reasons[path]}")
    for rid, missing in completeness.missing_by_record.items():
        print(f"         {rid}: {', '.join(missing)}")
    print()

    print(f"Layer 3  Validity: {len(issues)} issue(s)")
    for issue in issues:
        print(f"         {issue}")
    print()
    print(
        "Not checked here: whether the text is correct, whether title and abstract agree, whether a unit"
    )
    print(
        "matches the values, or whether the text is specific. Those need review or a semantic check"
    )
    print("(see Generative AI for Metadata Quality).")


# --- main ------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "records", type=Path, help="JSON records (nested) or CSV (flat)"
    )
    parser.add_argument(
        "profile", type=Path, help="completeness profile JSON for the data type"
    )
    parser.add_argument(
        "--type",
        choices=sorted(TYPES),
        help='data type; default: the profile\'s "type"',
    )
    parser.add_argument(
        "--level", help="profile level to check up to; default: the first level"
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path(__file__).with_name("wb-schemas"),
        help="folder holding the World Bank schema files (downloaded on first use)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with args.profile.open(encoding="utf-8") as fh:
        profile = json.load(fh)
    dtype = args.type or profile.get("type")
    if dtype not in TYPES:
        warn(f"unknown data type {dtype!r}; choose from {sorted(TYPES)}")
        return 2
    level = args.level or next(iter(profile["levels"]))
    if level not in profile["levels"]:
        warn(f"unknown level {level!r}; profile has {list(profile['levels'])}")
        return 2
    root = TYPES[dtype]
    id_path = profile.get("rules", {}).get("id", "idno")

    store = SchemaStore(args.schema_dir)
    records = load_records(args.records, profile, store, root)
    print(f"{len(records)} {dtype} record(s) from {args.records}\n")

    structure = layer_structure(records, store, root, id_path)
    completeness = layer_completeness(records, profile, level, id_path)
    issues = layer_validity(records, profile, id_path)
    report(
        dtype=dtype,
        root=root,
        n=len(records),
        level=level,
        profile=profile,
        structure=structure,
        completeness=completeness,
        issues=issues,
    )

    failed = (
        bool(completeness.missing_by_record)
        or bool(issues)
        or (structure.valid is not None and structure.valid < len(records))
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
