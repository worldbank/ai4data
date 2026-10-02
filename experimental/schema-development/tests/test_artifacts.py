"""Verify the paper's checked-in evidence at stable workflow boundaries."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from schema_development.paths import ROOT


def _csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV records while tolerating a UTF-8 byte-order mark."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _jsonl_count(path: Path) -> int:
    """Count and parse every non-empty JSONL record."""
    count = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def test_study_manifests_freeze_disjoint_samples() -> None:
    """The development and held-out manifests retain the paper's split."""
    development = _csv_rows(ROOT / "data/manifests/development_snapshots.csv")
    heldout = _csv_rows(ROOT / "data/manifests/heldout_snapshots.csv")
    development_names = {row["snapshot_filename"] for row in development}
    heldout_names = {row["snapshot_filename"] for row in heldout}

    assert len(development) == len(development_names) == 210
    assert len(heldout) == len(heldout_names) == 202
    assert development_names.isdisjoint(heldout_names)


def test_discovery_artifacts_retain_reported_counts() -> None:
    """Discovery outputs preserve the paper's main aggregation boundaries."""
    artifacts = ROOT / "artifacts/discovery"
    discovery = _csv_rows(artifacts / "2.0-discovery_results.csv")
    profiles = _csv_rows(artifacts / "3.0-field_profiles.csv")
    refined = _csv_rows(artifacts / "3.1-ontology_v1.csv")
    ontology_lines = (
        (artifacts / "3.1-ontology_v0.md").read_text(encoding="utf-8").splitlines()
    )

    assert len(discovery) == 3_041
    assert len(profiles) == 833
    assert sum(line.startswith("|") for line in ontology_lines) - 2 == 70
    assert sum(row["status"] == "keep" for row in refined) == 41


def test_validation_artifacts_and_frozen_schema_are_complete() -> None:
    """All held-out runs and the exact Appendix H schema remain available."""
    for directory in ("validation1", "validation2", "appendix_h"):
        assert _jsonl_count(ROOT / f"artifacts/{directory}/results.jsonl") == 202

    schema = ROOT / "schemas/Data Snapshot Metadata Schema v1.1.1.md"
    fields = re.findall(
        r"^### ([a-z][a-z0-9_]*)$", schema.read_text(encoding="utf-8"), re.MULTILINE
    )
    assert len(fields) == len(set(fields)) == 36

    frozen = ROOT / "artifacts/appendix_h/evaluated_schema_v1.2.schema.json"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == (
        "9403afa54339b5d71d7860c64211b387050c05d72524c45cf3bd4fedb36d3032"
    )
