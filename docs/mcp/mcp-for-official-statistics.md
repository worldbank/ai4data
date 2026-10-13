---
title: MCP for official statistics
description: What a Model Context Protocol server over a statistical catalog returns, the use cases it serves for a statistical office, the servers in operation, and where the design is documented in full.
---

# MCP for official statistics

A Model Context Protocol (MCP) server over official statistics gives every MCP-compatible AI client the same tools: search the catalog, read a record's metadata, fetch values for a series with their unit, period, status, release, and source. The office maintains one server; the clients, from desktop assistants to custom agents, need no integration of their own.

## Use cases for a statistical office

| Use case | What the server does | What the office gains |
|---|---|---|
| Answers from official figures | A chatbot calls the data tool and quotes the returned value with its period and source | Its figures reach users through AI systems, with provenance |
| Catalog discovery | An agent searches by meaning and reads the metadata before choosing a series | Fewer wrong series chosen by users and by models |
| Verified numbers | Each value carries a claim identifier that a renderer can check with [Proof-Carrying Numbers](../pcn/index.md) | Numbers in generated text that match the record or are flagged |
| Internal assistants | Staff assistants call the same tools over internal series under access control | One interface for public and internal use, with the tier rule enforced at the server |
| Monitoring of use | The server logs tool calls by client | Evidence of which series AI systems use, and which questions they fail on |

## What the tools return

A server over a catalog needs few tools. The cookbook on serving statistics to agents designs them in full; the shape is:

- `search_series(query, limit)`: identifier, name, unit, periodicity, and coverage for each match.
- `get_metadata(series)`: the record in the organization's schema, with definition, method, breaks, and caveats.
- `get_values(series, ref_area, period)`: observations with `TIME_PERIOD`, `OBS_VALUE`, `OBS_STATUS`, the unit, the release date, the source URL, and a claim identifier per observation.

The fields follow the SDMX cross-domain concepts so that a client built for one office works for another.

## Servers in operation and reference code

- The [Data360 MCP server](https://github.com/worldbank/data360-mcp) runs over the World Bank Data360 platform with a public endpoint, search with country-coverage checks, metadata, disaggregation, codelist lookup, data, and chart tools. Its [client setup guides](https://github.com/worldbank/data360-mcp/blob/main/docs/mcp_clients_setup.md) cover three MCP clients.
- The [Data AI Chatbot](https://github.com/worldbank/data-ai-chatbot) is a self-hostable application over any MCP server, with PCN marks on the numbers in its answers.
- The [dissemination cookbook](/cookbook/ai-ready-dissemination/retrieve) ships a minimal server over two CSV files, and the [end-to-end reference implementation](../reference-implementation/index.md) runs a server over a published catalog and verifies an answer against it.
- MCP access to NADA catalogs is in development; the released NADA software has a REST API and no MCP interface.

## Design in full

The [Practical Guide to Serving Official Statistics to AI Agents](/cookbook/serving-statistics-to-agents/) covers the purpose and scope of agent access, the design of the tools, context beyond the tools, provenance in every response, access control, quotas, and terms, safety of the interface, evaluation, distribution to clients and developers, and operation. The [MCP architecture page](mcp.md) gives the protocol background.

## Limitations

- An MCP server exposes what the catalog and the API already hold. Incomplete records and unstable identifiers pass through to every client.
- Access control, quotas, and terms are the office's to add; the protocol carries authorization but does not decide it.
- A server answers tool calls. Whether a client's answer is right depends on the client's model and prompt, which the evaluation chapter of the agents cookbook measures.
