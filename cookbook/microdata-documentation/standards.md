---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The published standards each chapter builds on, with the ones the World Bank has adopted marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards wherever one exists. Where the World
Bank's Development Data Group has adopted a standard for its own microdata
catalog, the guide uses that one.

## Documentation of microdata

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [DDI Codebook](https://ddialliance.org/ddi-codebook) | Documentation of a study, its files, and its variables: labels, categories, universes, questions, derivations, concepts, summary statistics, variable groups | The Microdata Library, NADA, and the Metadata Editor; the World Bank microdata schema is DDI Codebook 2.5 as JSON Schema | Every chapter |
| [World Bank metadata schemas](https://github.com/worldbank/metadata-schemas) | JSON Schema for microdata (DDI-based) and the other data types, with a Python library and Excel templates (MIT licence) | The Development Data Group's catalogs and tools | Chapters [1](./find.mdx), [3](./variables.mdx), [7](./fitness.mdx), [8](./access.mdx) |
| [DDI Lifecycle](https://ddialliance.org/ddi-lifecycle) | Reusable questions, concepts, and variables across studies | | Chapters [4](./concepts.mdx), recipe 4.3, and [6](./rounds.mdx), recipe 6.3 |
| [DDI-CDI](https://ddialliance.org/ddi-cdi) | Data structures and provenance across data types | | Chapter 7: provenance of derived files |
| [Metadata Editor](https://github.com/worldbank/metadata-editor) | Open-source application for documenting microdata and other types, with templates, validation, and publishing to NADA | Used by the Development Data Group's curators | Chapters 1 and 3 |
| [NADA](https://nada.ihsn.org/) | Open-source catalog: DDI records, schema.org markup, variable-level search, access types and request workflows, REST API, MCP interface | Runs the [Microdata Library](https://microdata.worldbank.org/) | Chapters 1, 5, and 8 |

## Concepts and classifications

| Standard | Use | In this guide |
|---|---|---|
| [ISCED](https://uis.unesco.org/en/topic/international-standard-classification-education-isced) | Education levels (ISCED 2011) | Chapter 4 |
| [ISCO](https://www.ilo.org/public/english/bureau/stat/isco/) | Occupations (ISCO-08) | Chapter 4 |
| [ISIC](https://unstats.un.org/unsd/classifications/Econ/isic) | Economic activities (ISIC Rev.4) | Chapter 4 |
| [COICOP](https://unstats.un.org/unsd/classifications/Econ/coicop) | Consumption expenditure | Chapter 4 |
| [ICLS resolutions](https://www.ilo.org/resource/19th-icls-resolution-i) | Statistics of work, employment, and labour underutilization (19th ICLS) | Chapter 4: labour force status |
| [SKOS](https://www.w3.org/TR/skos-reference/) | Concept schemes with multilingual labels | Chapter 4 |
| [XKOS](https://ddialliance.org/xkos) | Statistical classifications and correspondences | Chapters 4 and 6: crosswalks and correspondence tables |

## Identifiers, discovery, and licences

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [DataCite](https://schema.datacite.org/) | DOIs and citation metadata for datasets | The Microdata Library assigns DataCite DOIs (prefix 10.48529) | Chapters 1, 7, and 8 |
| [schema.org Dataset](https://schema.org/Dataset) | Dataset markup on study pages | NADA writes it on every study page | Chapter 1 |
| [DCAT 3](https://www.w3.org/TR/vocab-dcat-3/) | Catalog records for harvesting | | Chapter 1 |
| [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) | Open licence with attribution | The default for [World Bank datasets](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets) | Chapter 8 |

## Disclosure control and quality

| Standard or tool | Use | In this guide |
|---|---|---|
| [sdcMicro](https://github.com/sdcTools/sdcMicro) | Open-source statistical disclosure control for microdata: risk measures, suppression, top-coding | Chapter 8 |
| [IHSN](https://www.ihsn.org/) guidance | Anonymization and dissemination policy for microdata | Chapter 8 |
| [UN Fundamental Principles of Official Statistics](https://unstats.un.org/fpos/) | Confidentiality and quality principles | Chapters 7 and 8 |
| [W3C PROV](https://www.w3.org/TR/prov-overview/) | Provenance of derived data | Chapter 7 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
