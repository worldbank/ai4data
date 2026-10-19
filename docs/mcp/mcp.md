# Model Context Protocol (MCP) for data dissemination

## Overview

The Model Context Protocol (MCP) is an open standard introduced by Anthropic in November 2024 that defines how a large language model (LLM) application connects to external data sources and tools. An MCP server exposes a data source as a fixed set of tools and resources; any MCP client, such as a desktop assistant or a custom agent, discovers and calls them through the same protocol.

For development data, the protocol replaces one integration per AI system with one server per data provider. Before it, connecting an assistant to World Bank data meant custom retrieval logic, data transformation code, and API authentication for each AI system. With it, the provider maintains one server and each client that implements the protocol can use it.

---

## The integration problem the protocol addresses

Without a shared protocol, each AI system connects to a data source through its own plugin:

```
AI System A → custom plugin → WDI API → parse response → inject into prompt
AI System B → different custom plugin → same WDI API → different parsing → different prompt
AI System C → yet another custom integration → ...
```

Each integration is separate and duplicates the work of the others. With the protocol, the clients share one server:

```
AI System A ──┐
AI System B ──┼──► MCP Client → MCP Protocol → MCP Server → WDI API / Catalog
AI System C ──┘
```

The AI system calls the server's tools and receives structured responses. The data provider maintains one server.

---

## MCP architecture

The architecture has three components.

**MCP protocol.** A JSON-RPC based specification that defines the interface between client and server: how tools are discovered, how they are called, and how results are returned. The specification defines two standard transports, stdio for a client-launched subprocess and Streamable HTTP for a server behind an HTTP endpoint (earlier revisions used HTTP with server-sent events), and it allows custom transports over any bidirectional channel. The protocol does not depend on a particular model.

**MCP server.** A server that implements the protocol and exposes data as tools and resources. For development data, a server might expose tools such as:

- `get_indicator(indicator_code, country_code, start_year, end_year)`: a WDI time series
- `search_indicators(query)`: search over the indicator catalog by meaning
- `get_dataset_metadata(dataset_id)`: the metadata of a microdata dataset
- `list_countries(region)`: the countries in a region

**MCP client.** A client that uses the protocol to connect to servers. It is built into MCP-compatible AI applications (Claude Desktop, Cursor, Zed, and others) and can be implemented in a custom application.

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

## Example: a development data MCP tool

An example tool definition that an MCP server for development data might expose:

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

When a user asks an MCP-enabled assistant for Nigeria's GDP growth rate in 2020, the assistant calls `get_indicator("NY.GDP.MKTP.KD.ZG", "NGA", 2020, 2020)`, receives the value, and uses it in its answer with the WDI record as the source.

---

## What the protocol provides for official statistics

For national statistical offices and international data organizations, a server over a catalog has four properties:

1. **One server for every client.** Once a catalog is published as an MCP server, each client that implements the protocol can query it without further integration work on the provider's side.
2. **Provenance in the response.** The tools return structured data with metadata, so a client can cite the indicator, the source, and the vintage of each value it uses.
3. **One implementation per provider.** The office maintains one server and does not write an integration for each AI platform.
4. **Authorization in the protocol.** The specification carries authentication and authorization, which the server uses to control access to restricted or embargoed series.

[MCP for official statistics](mcp-for-official-statistics.md) gives the use cases for statistical offices and the servers in operation.

---

## Getting started

**Servers and applications from the program**

- [Data360 MCP server](https://github.com/worldbank/data360-mcp): in operation over the World Bank Data360 platform, with a public endpoint, versioned releases on PyPI, and setup guides for three MCP clients. Its status and known limits are on the [tools catalogue](/tools#data360-mcp).
- [Data AI Chatbot](https://github.com/worldbank/data-ai-chatbot): a self-hostable chat application over any MCP server, with Proof-Carrying Numbers marks on the numbers in its answers.
- [End-to-end reference implementation](../reference-implementation/index.md): a server over a published catalog, called through the protocol, with the answer verified against the tool results.
- [Minimal example server](/cookbook/ai-ready-dissemination/retrieve) in the dissemination cookbook, and the full design in the [agents cookbook](/cookbook/serving-statistics-to-agents/).

**Protocol resources**

- [Model Context Protocol specification](https://modelcontextprotocol.io/): the protocol documentation, including the transport bindings and the security guidance for authorization.
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk): for building servers and clients in Python.

---

## Limitations

- MCP standardizes how tools are discovered and called. It does not make a catalog complete or an API reliable; the server passes through what they hold, which the completeness and retrieval chapters of the [dissemination cookbook](/cookbook/ai-ready-dissemination/) address first.
- The protocol carries authorization and the specification gives security guidance, but access control, quotas, and terms are decisions the office makes and enforces at the server. The [access chapter of the agents cookbook](/cookbook/serving-statistics-to-agents/access) gives the design.
- A server returns data; the quality of an answer depends on the client's model and prompt. The [evaluation chapter](/cookbook/serving-statistics-to-agents/evaluate) measures it.
- The specification is young, and its transports and authorization flows have changed between revisions. Servers pin an SDK version and state the protocol version they implement.
