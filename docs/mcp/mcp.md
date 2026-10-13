# Model Context Protocol (MCP) for AI-centric Data Dissemination

## Overview

The Model Context Protocol (MCP) is an **open standard** introduced by Anthropic in November 2024 to simplify how large language models (LLMs) connect to external data sources and tools. Think of MCP as the "USB-C of AI integrations"—a universal interface that enables AI systems, like Claude or ChatGPT, to tap into datasets, APIs, documents, or productivity tools without building bespoke integrations for each one.

For development data, MCP is a transformative opportunity. Today, integrating World Bank data into an AI assistant requires custom retrieval logic, data transformation code, and API authentication—separately for each AI system. MCP provides a single server implementation that any MCP-compatible AI client can use. Build it once; use it everywhere.

---

## The Problem MCP Solves

Before MCP, connecting an AI assistant to development data looked like this:

```
AI System A → custom plugin → WDI API → parse response → inject into prompt
AI System B → different custom plugin → same WDI API → different parsing → different prompt
AI System C → yet another custom integration → ...
```

Each integration is bespoke, fragile, and duplicates effort. MCP replaces this with:

```
AI System A ──┐
AI System B ──┼──► MCP Client → MCP Protocol → MCP Server → WDI API / Catalog
AI System C ──┘
```

A single MCP server implementation serves all compatible AI clients. The AI system calls standardized tools and receives standardized responses. The data provider maintains a single server.

---

## MCP Architecture

The MCP architecture has three components:

**MCP Protocol** — A JSON-RPC based specification that defines the interface between client and server. It standardizes how tools are discovered, called, and how results are returned. The protocol is transport-agnostic (HTTP, stdin/stdout, WebSocket) and LLM-agnostic.

**MCP Server** — A server that implements the MCP protocol and exposes data as **tools** and **resources**. For development data, a server might expose tools like:
- `get_indicator(indicator_code, country_code, start_year, end_year)` — Retrieve WDI timeseries
- `search_indicators(query)` — Semantic search over the indicator catalog
- `get_dataset_metadata(dataset_id)` — Retrieve metadata for a microdata dataset
- `list_countries(region)` — List countries by region

**MCP Client** — A client that uses the MCP protocol to connect to servers. This is built into MCP-compatible AI systems (Claude Desktop, Cursor, Zed, and many others) or can be implemented in custom AI applications.

```
┌─────────────────────────────────────────────────────────┐
│                    AI Application                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │            LLM (Claude, GPT-4, etc.)             │   │
│  └───────────────────────┬──────────────────────────┘   │
│                          │ tool calls                    │
│  ┌───────────────────────▼──────────────────────────┐   │
│  │                  MCP Client                       │   │
│  └───────────────────────┬──────────────────────────┘   │
└──────────────────────────┼──────────────────────────────┘
                           │ MCP Protocol (JSON-RPC)
┌──────────────────────────▼──────────────────────────────┐
│                    MCP Server                            │
│  Tools: get_indicator, search_indicators, ...            │
│  Resources: catalog metadata, documentation              │
│                          │                               │
│               ┌──────────▼──────────┐                   │
│               │   Development Data   │                   │
│               │  (WDI, Microdata,   │                   │
│               │   Documents, ...)   │                   │
│               └─────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

---

## Example: A Development Data MCP Tool

Here is an example tool definition that an MCP server for development data might expose:

```python
@mcp_server.tool()
def get_indicator(
    indicator_code: str,
    country_code: str,
    start_year: int = 2000,
    end_year: int = 2024,
) -> dict:
    """Retrieve a World Development Indicator timeseries.

    Args:
        indicator_code: WDI indicator code (e.g., 'NY.GDP.MKTP.KD.ZG')
        country_code: ISO 3166-1 alpha-3 country code (e.g., 'NGA')
        start_year: Start year (default 2000)
        end_year: End year (default 2024)

    Returns:
        Dictionary with indicator metadata and annual values.
    """
    # Fetch from WDI API and return structured response
    ...
```

When a user asks an MCP-enabled AI assistant "What was Nigeria's GDP growth rate in 2020?", the assistant calls `get_indicator("NY.GDP.MKTP.KD.ZG", "NGA", 2020, 2020)`, receives the value, and incorporates it directly into its response—with provenance from the official WDI source.

---

## Why MCP Matters for Official Statistics

For national statistical offices (NSOs) and international data organizations, MCP offers:

1. **Discoverability at zero marginal cost.** Once a data catalog is published as an MCP server, it becomes queryable by any AI assistant that supports MCP—without additional integration work.
2. **Provenance and trust.** MCP tools return structured data with metadata, so AI assistants can cite the specific indicator, source, and vintage in their responses—maintaining the traceability expected of official statistics.
3. **Scalable AI integration.** A single MCP implementation serves every platform that speaks the protocol, so the office negotiates no integration agreements per platform.
4. **Standardized access control.** The MCP protocol supports authentication and authorization, allowing controlled access to restricted or embargoed datasets.

See [MCP for Official Statistics](mcp-for-official-statistics.md) for use cases specific to NSOs and official data producers.

---

## Getting Started

**Servers and applications from the program**

- [Data360 MCP server](https://github.com/worldbank/data360-mcp): in operation over the World Bank Data360 platform, with a public endpoint, versioned releases on PyPI, and setup guides for three MCP clients. Its status and known limits are on the [tools catalogue](/tools#data360-mcp).
- [Data AI Chatbot](https://github.com/worldbank/data-ai-chatbot): a self-hostable chat application over any MCP server, with Proof-Carrying Numbers marks on the numbers in its answers.
- [End-to-end reference implementation](../reference-implementation/index.md): a server over a published catalog, called through the protocol, with the answer verified against the tool results.
- [Minimal example server](/cookbook/ai-ready-dissemination/retrieve) in the dissemination cookbook, and the full design in the [agents cookbook](/cookbook/serving-statistics-to-agents/).

**Protocol resources**

- [Model Context Protocol specification](https://modelcontextprotocol.io/): the protocol documentation, including its security guidance for authorization.
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk): for building servers and clients in Python.

---

## Limitations

- MCP standardizes how tools are discovered and called. It does not make a catalog complete or an API reliable; the server passes through what they hold, which the completeness and retrieval chapters of the [dissemination cookbook](/cookbook/ai-ready-dissemination/) address first.
- The protocol carries authorization and the specification gives security guidance, but access control, quotas, and terms are decisions the office makes and enforces at the server. The [access chapter of the agents cookbook](/cookbook/serving-statistics-to-agents/access) gives the design.
- A server returns data; the quality of an answer depends on the client's model and prompt. The [evaluation chapter](/cookbook/serving-statistics-to-agents/evaluate) measures it.
- The specification is young and its transports and authorization flows have changed between versions. Servers pin an SDK version and state the protocol version they implement.
