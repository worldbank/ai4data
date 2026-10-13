"""Step 4: an MCP server over the published catalog.

Exposes three read-only tools to any MCP client: search the catalog by
words in the name or description, read a record, and fetch the values of a
series. Every observation carries a claim identifier, so that a renderer
can verify the numbers a model takes from it (see e2e/verify.py). The
response fields follow the SDMX cross-domain concepts, and the value rows
carry the keys the Proof-Carrying Numbers Data360 preset reads:
claim_id, OBS_VALUE, REF_AREA, TIME_PERIOD.

Usage:
    python server.py <catalog_dir>          # stdio transport, for a client
    E2E_CATALOG=<catalog_dir> mcp dev server.py

The runner starts this server as a subprocess and talks to it over stdio,
so the run exercises the protocol end to end. Works with mcp 1.x and 2.x.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

try:  # mcp 2.x
    from mcp.server import MCPServer as Server
except ImportError:  # mcp 1.x, where the class was called FastMCP
    from mcp.server.fastmcp import FastMCP as Server

CATALOG = Path(
    sys.argv[1]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else os.environ.get("E2E_CATALOG", "build/catalog")
)

mcp = Server("example-catalog")


def _index() -> list[dict]:
    with (CATALOG / "index.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _record(idno: str) -> dict | None:
    path = CATALOG / "records" / f"{idno}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _text(entry: dict) -> str:
    record = _record(entry["idno"]) or {}
    if entry["type"] == "indicator":
        sd = record.get("series_description", {})
        parts = [sd.get("name", ""), sd.get("definition_long", "")] + [
            k.get("name", "") for k in sd.get("keywords", [])
        ]
    else:
        info = record.get("study_desc", {}).get("study_info", {})
        parts = [entry.get("name", ""), info.get("abstract", "")] + [
            k.get("keyword", "") for k in info.get("keywords", [])
        ]
    return " ".join(parts).lower()


@mcp.tool()
def search_catalog(query: str, limit: int = 5) -> list[dict]:
    """Search published series and surveys by words in the name, description, or keywords.

    Returns idno, type, and name for each match, best first. Only records
    that passed validation are in the catalog, so a product withheld at
    publication is absent here too.
    """
    words = [w for w in query.lower().split() if len(w) > 2]
    hits = []
    for entry in _index():
        text = _text(entry)
        score = sum(1 for w in words if w in text)
        if score:
            hits.append((score, entry))
    hits.sort(key=lambda h: (-h[0], h[1]["idno"]))
    return [
        {"idno": e["idno"], "type": e["type"], "name": e["name"]}
        for _, e in hits[:limit]
    ]


@mcp.tool()
def get_record(idno: str) -> dict:
    """Return the full metadata record of a series or a survey in the World Bank schema."""
    record = _record(idno)
    if record is None:
        return {"error": f"no record {idno} in the catalog"}
    return record


@mcp.tool()
def get_values(series: str, ref_area: str = "NAT", period: str | None = None) -> dict:
    """Return the published values of a series with unit, status, release, and source.

    Each observation carries a claim_id that names it, so that an
    application can verify a number quoted from it.
    """
    path = CATALOG / "values" / f"{series}.csv"
    if not path.exists():
        return {"error": f"no values for {series} in the catalog"}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r["REF_AREA"] == ref_area
            and (period is None or r["TIME_PERIOD"] == period)
        ]
    if not rows:
        return {
            "error": f"no values for {series} / {ref_area} / {period or 'all periods'}"
        }
    return {
        "SERIES": series,
        "REF_AREA": ref_area,
        "UNIT_MEASURE": rows[0]["UNIT_MEASURE"],
        "RELEASE": rows[0]["RELEASE"],
        "SOURCE_URL": rows[0]["SOURCE_URL"],
        "observations": [
            {
                "claim_id": f"{series}:{ref_area}:{r['TIME_PERIOD']}",
                "TIME_PERIOD": r["TIME_PERIOD"],
                "OBS_VALUE": float(r["OBS_VALUE"]),
                "OBS_STATUS": r["OBS_STATUS"],
                "REF_AREA": ref_area,
                "UNIT_MEASURE": r["UNIT_MEASURE"],
            }
            for r in rows
        ],
    }


if __name__ == "__main__":
    mcp.run()
