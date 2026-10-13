"""Step 1: load the documented records and the published values.

The series records are in the nested form of the World Bank indicator
schema (timeseries-schema.json), the survey record in the microdata schema
(DDI Codebook based), and the values in a CSV whose columns are SDMX
cross-domain concept names. The same files are what the Metadata Editor
exports and what NADA stores, so an organization replaces the fixtures
with its own exports and the rest of the pipeline runs unchanged.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Record = dict[str, Any]


@dataclass
class Documented:
    series: list[Record]
    survey: Record
    values: list[dict[str, str]]
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def records(self) -> list[tuple[str, Record]]:
        """(type, record) for every record, series first."""
        return [("indicator", r) for r in self.series] + [("microdata", self.survey)]


def values_at(obj: Any, parts: list[str]) -> list[Any]:
    """All values at a dotted path, flattening through lists."""
    if not parts:
        return [obj]
    if isinstance(obj, list):
        return [v for item in obj for v in values_at(item, parts)]
    if isinstance(obj, dict) and parts[0] in obj:
        return values_at(obj[parts[0]], parts[1:])
    return []


def first(record: Record, path: str) -> Any:
    for v in values_at(record, path.split(".")):
        if v not in (None, "", [], {}):
            return v
    return None


def record_id(kind: str, record: Record) -> str:
    path = (
        "series_description.idno"
        if kind == "indicator"
        else "study_desc.title_statement.idno"
    )
    return str(first(record, path) or "?")


def load(fixtures: Path) -> Documented:
    with (fixtures / "series.json").open(encoding="utf-8") as fh:
        series = json.load(fh)
    with (fixtures / "survey.json").open(encoding="utf-8") as fh:
        survey = json.load(fh)
    with (fixtures / "values.csv").open(newline="", encoding="utf-8") as fh:
        values = list(csv.DictReader(fh))
    return Documented(
        series=series,
        survey=survey,
        values=values,
        sources={
            "series": "series.json",
            "survey": "survey.json",
            "values": "values.csv",
        },
    )
