---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: Schema and file conventions for the policy development experiment.
---

# Standards used in this guide

## Metadata and records

| Standard or convention | Use | World Bank use | In this guide |
|---|---|---|---|
| [Data Snapshot Metadata Schema v1.4](/docs/data-snapshots/metadata-schema/) | Describes the analytical content of a data snapshot. | The Data Snapshot project implements the schema in Pydantic. | The same schema governs the reference records and extraction payloads. |
| [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12) | Expresses structural constraints. | The Data Snapshot project exports a validation schema in this dialect. | The gold release includes a frozen schema. The Pydantic implementation also checks constraints in Python. |
| [JSON Lines](https://jsonlines.org/) | Stores one JSON value per line. | Used here as a proposed round file convention. | Extraction and judge rows can be associated by round and sample ID. |

## Corrections

Report a version or reference correction through the repository [issue tracker](https://github.com/worldbank/ai4data/issues).
