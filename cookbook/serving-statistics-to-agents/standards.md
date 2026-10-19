---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The published standards, specifications, and reference implementations each chapter builds on, with the ones the World Bank has adopted marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards wherever one exists. Where the World
Bank's Development Data Group operates an implementation, the guide uses
it as the reference.

## Protocol and implementations

| Standard or resource | Use | World Bank use | In this guide |
|---|---|---|---|
| [Model Context Protocol](https://modelcontextprotocol.io/specification/latest) | Open protocol for connecting AI applications to tools, resources, and prompts over JSON-RPC 2.0; the current specification is dated 2026-07-28 | The [Data360 MCP server](https://github.com/worldbank/data360-mcp) and [NADA AI](/tools#nada-ai), the MCP server over NADA catalogs | Every chapter |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | Reference implementation (`MCPServer` in version 2, `FastMCP` in version 1) | The dissemination cookbook's server example | Chapters [2](./tools.mdx) and [9](./operate.mdx) |
| [Data360 MCP server](https://github.com/worldbank/data360-mcp) | A server in operation over the Data360 API with discovery, metadata, data, analysis, and visualization tools and guidance resources; MIT licence with the World Bank IGO rider | | Chapters 1, 2, 4, 6, and 9 |
| [NADA](https://nada.ihsn.org/) and [NADA AI](/tools#nada-ai) | Catalog with a REST API and schema.org markup; NADA AI, being integrated into NADA, serves it over MCP with semantic search | Runs the [Microdata Library](https://microdata.worldbank.org/) | Chapter 2 |
| [Model Context Protocol for statistics](/docs/mcp/) | The program's description of a statistics server's architecture and tools | | Chapters 1 and 2 |

## Data and metadata

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [SDMX](https://sdmx.org/) cross-domain concepts | Field names for values (`SERIES`, `REF_AREA`, `TIME_PERIOD`, `OBS_VALUE`, `OBS_STATUS`, `UNIT_MEASURE`) and the observation status code list | The [WDI SDMX API](https://datahelpdesk.worldbank.org/knowledgebase/articles/1886701-sdmx-api-queries) | Chapters 2 and [4](./provenance.mdx) |
| [World Bank metadata schemas](https://github.com/worldbank/metadata-schemas) | Field names for series records (`idno`, `name`, `definition_long`, `measurement_unit`, `series_break`, `citation_requirement`) | The Development Data Group's catalogs | Chapters 2 and 4 |
| [JSON Schema](https://json-schema.org/) | Input and output schemas of tools (2020-12 by default in MCP) | | Chapter 2 |
| [OpenAPI](https://www.openapis.org/) | Description of the API the server wraps | The Data360 API | Chapters 1 and 2 |
| [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) | Licence returned with data | The default for [World Bank datasets](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets) | Chapter 4 |

## Safety and operation

| Standard | Use | In this guide |
|---|---|---|
| [MCP authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) (OAuth 2.1, RFC 9728, RFC 8707) | Authorization for HTTP transports where user-level access is required | | Chapter [5](./access.mdx) |
| [MCP resources and prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/resources) | Context beyond the tools: guidance, code lists, calendars, prompt templates | | Chapter [3](./context.mdx) |
| [MCP registry server.json](https://registry.modelcontextprotocol.io/) | Listing format for servers | | Chapter [8](./distribute.mdx) |
| MCP security requirements | Servers validate inputs, implement access controls, rate-limit, and sanitize outputs; clients keep a human in the loop and treat annotations as untrusted | Chapter [6](./safety.mdx) |
| [COUNTER Code of Practice for Research Data](https://coprd.countermetrics.org/) | Counting machine access separately from regular access | Chapter [9](./operate.mdx) |
| [Semantic Versioning](https://semver.org/) | Versioning the server and its tools | Chapter 9 |
| [Proof-Carrying Numbers](https://arxiv.org/abs/2509.06902) | Verification of numbers in answers against the record | Chapters 4 and [7](./evaluate.mdx) |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
