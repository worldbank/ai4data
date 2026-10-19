---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The process models, classifications, quality frameworks, and program tools each chapter builds on, with the ones the World Bank uses marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards wherever one exists. Where the
program's workstreams serve a production phase, the guide uses them.

## Process and quality frameworks

| Standard | Use | In this guide |
|---|---|---|
| [GSBPM 5.1](https://unece.org/statistics/modernstats/gsbpm) | The phases and sub-processes the map and every chapter refer to | Chapter [1](./map.mdx) and all others |
| [UN Fundamental Principles of Official Statistics](https://unstats.un.org/fpos/) | Professional responsibility and transparency of methods, which keep decisions with people and require the statement of model use | Chapters 1 and [7](./assurance.mdx) |
| [UN National Quality Assurance Frameworks manual](https://unstats.un.org/unsd/methodology/dataquality/) | The quality reporting the statement of model use fits into | Chapter 7 |
| [European Statistics Code of Practice](https://ec.europa.eu/eurostat/web/quality/european-statistics-code-of-practice) | Sound methodology, quality commitment, and transparency principles | Chapter 7 |
| [ISO/IEC 42001](https://www.iso.org/standard/42001) | A management system standard for AI use | Chapter 7 |
| [UNECE Statistical Data Editing](https://unece.org/statistics/statistical-data-editing) | Editing and imputation methods, the method of record | Chapter [4](./editing.mdx) |

## Classifications

| Standard | Use | In this guide |
|---|---|---|
| [ISCO-08](https://ilostat.ilo.org/methods/concepts-and-definitions/classification-occupation/) | Occupation coding: structure, definitions, index | Chapter [3](./coding.mdx) |
| [ISIC Rev. 4](https://unstats.un.org/unsd/classifications/Econ/isic) and the [UN classifications registry](https://unstats.un.org/unsd/classifications/) | Activity coding and correspondence tables between versions | Chapter 3 |
| [ISCED](https://uis.unesco.org/en/topic/international-standard-classification-education-isced) | Education coding | Chapter 3 |
| [DDI Lifecycle 3.3](https://ddialliance.org/Specification/DDI-Lifecycle/3.3/) | Reusable questions and concepts, the form of a question bank | Chapter [2](./design.mdx) |

## Program methods and tools

| Resource | Use | In this guide |
|---|---|---|
| [Mapping to the GSBPM](/docs/gsbpm-mapping) | The program's workstreams placed in the production phases | Chapter 1 |
| [Anomaly Detection and Explanation](/docs/anomaly-detection/) | Detection and structured LLM explanation of unusual values, with a feedback system | Chapter 4 |
| Statistical classification and coding workstream | AI-assisted coding against standard classifications with human review | Chapter 3 |
| Global Question Bank workstream | Multilingual semantic mappings of survey questions across surveys | Chapter 2 |
| [Proof-Carrying Numbers](https://arxiv.org/abs/2509.06902) | Verification of stated numbers against the source | Chapter [5](./commentary.mdx) |
| [Efficient and Inclusive AI Applications](/docs/inclusive-ai/) | Local open-weight models, batch processing, provider-agnostic configuration | Chapter [6](./running.mdx) |
| [Metadata Curation with Language Models cookbook](/cookbook/metadata-curation-with-llms/) and [Evaluation Suites cookbook](/cookbook/evaluation-suites/) | Drafting, review, and translation workflows; evaluation suites and report cards | Chapters 2, 5, and 7 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
