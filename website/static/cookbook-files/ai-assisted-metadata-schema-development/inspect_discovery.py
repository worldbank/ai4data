"""Inspect selected recorded discovery responses against a source register.

Usage: python inspect_discovery.py discovery_excerpt.jsonl source_images.csv
This standard-library script checks identities, status, and selected field
structure. It writes nothing and never calls a model. Exit status is 0 for
valid selected responses, 1 for logged failed responses, or 2 for invalid input.
What this does not check: semantic accuracy, complete historical prompts,
omitted fields, raw responses, token accounting, or unseen execution records.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def inspect_records(
    records: list[dict], register: list[dict[str, str]]
) -> tuple[int, int, int, int]:
    """Validate selected records and count retained and original fields.

    Parameters
    ----------
    records : list[dict]
        Recorded response excerpts, one per selected snapshot.
    register : list[dict[str, str]]
        Source register containing identities, corpora, and split membership.

    Returns
    -------
    tuple[int, int, int, int]
        Valid responses, failed responses, retained fields, original fields.

    Raises
    ------
    TypeError
        If a selected field is not an object.
    ValueError
        If records are empty, identities conflict, provenance is inconsistent,
        or a completed response violates the demonstrated field structure.
    """
    if not records:
        raise ValueError("No response records.")
    sources = {row["snapshot_id"]: row for row in register}
    if len(sources) != len(register):
        raise ValueError("Duplicate register identities.")
    seen: set[str] = set()
    valid = failed = retained = original = 0
    for row in records:
        name = row.get("snapshot_id")
        if not isinstance(name, str) or name not in sources or name in seen:
            raise ValueError("Unknown or duplicate response identity.")
        seen.add(name)
        source = sources[name]
        if source["split"] != "development" or row.get("source") != source["corpus"]:
            raise ValueError(f"Discovery provenance mismatch: {name}")
        if not isinstance(row.get("model"), str) or not row["model"].strip():
            raise ValueError(f"Missing model identity: {name}")
        if (
            row.get("status") != "completed"
            or row.get("error") is not None
            or row.get("incomplete_details") is not None
        ):
            failed += 1
            continue
        parsed = row.get("parsed_output")
        fields = parsed.get("fields") if isinstance(parsed, dict) else None
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"No parsed fields: {name}")
        for field in fields:
            if not isinstance(field, dict):
                raise TypeError(f"Invalid field object: {name}")
            for key in [
                "metadata_field",
                "observed_value",
                "description",
                "source_level",
                "discovery_value",
                "reasoning",
            ]:
                if not isinstance(field.get(key), str) or not field[key].strip():
                    raise ValueError(f"Missing field value: {name}/{key}")
            if field["source_level"] not in {"snapshot", "document", "both"}:
                raise ValueError(f"Unknown source level: {name}")
            if field["discovery_value"] not in {"high", "medium", "low"}:
                raise ValueError(f"Unknown discovery value: {name}")
        count = row.get("original_field_count")
        if type(count) is not int or count < len(fields):
            raise ValueError(f"Invalid original field count: {name}")
        valid += 1
        retained += len(fields)
        original += count
    return valid, failed, retained, original


def main(argv: list[str] | None = None) -> int:
    """Inspect the response excerpt and print a bounded integrity summary.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments; None reads the process arguments.

    Returns
    -------
    int
        Zero for valid records or one for logged failures; invalid input exits 2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument("register", type=Path)
    args = parser.parse_args(argv)
    try:
        records = [
            json.loads(line)
            for line in args.responses.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all(isinstance(row, dict) for row in records):
            raise ValueError("Each response must be an object.")
        with args.register.open(encoding="utf-8", newline="") as handle:
            register = list(csv.DictReader(handle))
        required = {"snapshot_id", "corpus", "split"}
        if not register or any(not required <= row.keys() for row in register):
            raise ValueError("The source register lacks required columns.")
        valid, failed, retained, original = inspect_records(records, register)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"{len(records)} selected response records: {valid} valid, {failed} failed")
    print(
        f"{retained} retained fields; {original} fields in the original valid responses"
    )
    print("These excerpts omit original fields and raw responses.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
