"""A minimal MCP server over a statistics catalog.

Exposes two read-only tools to any MCP-compatible AI client: one to search
the catalog and one to fetch values for a series. The data come from the two
example CSV files next to this script, whose columns follow the World Bank
indicator metadata schema (catalog) and SDMX cross-domain concepts (values).
In production the tools would call the office's existing SDMX or REST API.

The World Bank's Data360 MCP server (github.com/worldbank/data360-mcp) is a
production example of the same pattern, with search, metadata, data, and
analysis tools.

Requires the official Python SDK:  pip install "mcp[cli]"
Run locally:                       python mcp_server_example.py
Try it with the inspector:         mcp dev mcp_server_example.py
"""

import csv
from pathlib import Path

try:  # mcp 2.x
    from mcp.server import MCPServer as Server
except ImportError:  # mcp 1.x, where the class was called FastMCP
    from mcp.server.fastmcp import FastMCP as Server

HERE = Path(__file__).parent
mcp = Server("statistics-example")


def _read(name):
    with open(HERE / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


CATALOG = _read("example_catalog.csv")
VALUES = _read("example_values.csv")


@mcp.tool()
def search_series(query: str, limit: int = 5) -> list[dict]:
    """Search published series by words in the name or definition.

    Returns idno, name, measurement_unit, periodicity, and coverage for each
    match.
    """
    words = query.lower().split()
    hits = []
    for rec in CATALOG:
        text = (rec["name"] + " " + rec["definition_long"]).lower()
        score = sum(1 for w in words if w in text)
        if score:
            hits.append((score, rec))
    hits.sort(key=lambda h: h[0], reverse=True)
    keys = (
        "idno",
        "name",
        "measurement_unit",
        "periodicity",
        "time_period_start",
        "time_period_end",
    )
    return [{k: rec[k] for k in keys} for _, rec in hits[:limit]]


@mcp.tool()
def get_values(series: str, ref_area: str = "NAT") -> dict:
    """Return the published values for a series.

    The response carries the unit, the observation status, the release date,
    and the source URL so that the client can cite them.
    """
    rows = [v for v in VALUES if v["SERIES"] == series and v["REF_AREA"] == ref_area]
    if not rows:
        return {"error": f"no values for {series} / {ref_area}"}
    return {
        "SERIES": series,
        "REF_AREA": ref_area,
        "UNIT_MEASURE": rows[0]["UNIT_MEASURE"],
        "RELEASE": rows[0]["RELEASE"],
        "SOURCE_URL": rows[0]["SOURCE_URL"],
        "observations": [
            {
                "TIME_PERIOD": r["TIME_PERIOD"],
                "OBS_VALUE": float(r["OBS_VALUE"]),
                "OBS_STATUS": r["OBS_STATUS"],
            }
            for r in rows
        ],
    }


if __name__ == "__main__":
    mcp.run()
