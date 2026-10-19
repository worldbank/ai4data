---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The published standards, program methods, and tools each chapter builds on, with the ones the World Bank has adopted marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards wherever one exists. Where the World
Bank's Development Data Group has adopted a standard or built a tool for
its own catalogs, the guide uses that one.

## Records and vocabularies

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [World Bank metadata schemas](https://github.com/worldbank/metadata-schemas) | The fields a model drafts or flags (`definition_long`, `measurement_unit`, `keywords`, `topics` with vocabulary and URI) | The Development Data Group's catalogs | Chapters [3](./draft.mdx), [5](./review.mdx), and [6](./standardize.mdx) |
| [SKOS](https://www.w3.org/TR/skos-reference/) | Concept schemes with preferred and alternate labels per language and URIs | | Chapter 6 |
| [XKOS](https://ddialliance.org/xkos) | Statistical classifications and correspondences | | Chapter 6 |
| [Metadata Editor](https://github.com/worldbank/metadata-editor) | Applying accepted changes with validation; publishing to NADA | Used by the Development Data Group's curators | Chapters 5 and 6 |
| [NADA](https://nada.ihsn.org/) | The catalog the records are published to | Runs the [Microdata Library](https://microdata.worldbank.org/) | Chapter 5 |
| [ISO 639](https://www.iso.org/iso-639-language-code) | Language codes for multilingual labels and fields | | Chapters 6 and 7 |
| [schema.org `inLanguage`](https://schema.org/inLanguage) | Language of a catalog page for crawlers and assistants | NADA pages carry schema.org | Chapter 7 |
| [Frictionless Table Schema](https://specs.frictionlessdata.io/table-schema/) | Description of a legacy spreadsheet's columns before mapping | | Chapter 2 |

## Program methods and tools

| Resource | Use | In this guide |
|---|---|---|
| [Generative AI for Metadata Quality](/docs/metadata-quality/generative-ai-for-metadata-quality) and its [notebook](/docs/notebooks/metadata-quality-assessment-with-llm) | Scoring records on completeness, semantic alignment, specificity, and consistency with structured output; the task families and risks | Chapters [1](./scope.mdx), 3, and [4](./assess.mdx) |
| [Metadata Reviewer](/docs/metadata-reviewer/overview) | Agentic detection of issues with category, severity, and a proposed fix; the client; the review board | Chapter 5 |
| [Metadata Augmentation](/docs/metadata-augmentation/) | Model curation of variable groups with a self-consistency check | Chapter 3 |
| [Efficient and Inclusive AI](/docs/inclusive-ai/) | Model size by task, batch processing, provider-agnostic configuration with local models | Chapter [8](./improve.mdx) |
| [litellm](https://github.com/BerriAI/litellm) | The provider-agnostic client the program's tools use | Chapters 1 and 8 |

## Governance and measurement

| Standard | Use | In this guide |
|---|---|---|
| [UN Fundamental Principles of Official Statistics](https://unstats.un.org/fpos/) | Professional responsibility for published metadata | Chapter 1 |
| [ISO/IEC 42001](https://www.iso.org/standard/42001) and [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) | Management and risk frameworks the AI-use policy and register fit into | Chapter 1 |
| Inter-rater agreement (Cohen's kappa) | Agreement between curators as the reference for calibration | Chapter 4 |
| [JSON Schema](https://json-schema.org/) | Structured output for drafts and scores | Chapters 3 and 4 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
