"""The indicator tool as an MCP server, with FastMCP.

    python indicator_server.py            # stdio transport, for a local client

Notebook 04 starts it as a subprocess through the FastMCP client and calls its
tools over the protocol. The same shape, exposed over HTTP, is what the Data360
MCP server offers to any MCP client.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workshop as w  # noqa: E402

mcp = FastMCP("indicator-server")


@mcp.tool
def list_indicators() -> list[dict]:
    """The indicators this server holds, with their ids and names."""
    return w.list_indicators()


@mcp.tool
def get_indicator(indicator: str, country: str, start: int | None = None, end: int | None = None) -> dict:
    """Observations of a World Development Indicators series for one country. `indicator`
    is an id such as IT.NET.USER.ZS; `country` is a name or ISO3 code; `start` and `end`
    bound the years. Every observation carries a claim_id for verification."""
    return w.indicator_lookup(indicator, country, start, end)


if __name__ == "__main__":
    mcp.run(show_banner=False)
