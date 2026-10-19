---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The published standards, codes of practice, and open resources each chapter builds on, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards and codes of practice wherever one
exists. Where the World Bank's Development Data Group has adopted one for
its own catalogs, the guide uses that one.

## Usage metrics and citations

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [Scholix](https://scholexplorer.openaire.eu/) | Exchange of dataset-literature links between services | | Chapter 3 |
| [COUNTER Code of Practice for Research Data](https://coprd.countermetrics.org/) | Rules for counting dataset views and downloads, with machine access separated from regular access; developed with [Make Data Count](https://makedatacount.org/) | | Chapter [8](./report.mdx) |
| [DataCite](https://schema.datacite.org/) | DOIs and citation metadata for datasets; citation events for datasets with DOIs | The Microdata Library assigns DataCite DOIs | Chapters [1](./define.mdx), [3](./citations.mdx), [4](./collect.mdx), and 8 |
| [Crossref](https://www.crossref.org/) | DOIs and reference lists of publications | | Chapters 3 and 4 |
| [OpenAlex](https://openalex.org/) | Open index of scholarly works with an API | | Chapters 3 and 4 |

## Collection

| Standard | Use | In this guide |
|---|---|---|
| [RFC 9309 (robots.txt)](https://www.rfc-editor.org/rfc/rfc9309) | The rules a crawler reads before collecting from a site | Chapter 5 |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | Text extraction from PDFs, as in the program's document parser | Chapters 4 and 5 |

## Extraction and harmonization

| Resource | Use | In this guide |
|---|---|---|
| [Monitoring of Data Use](/docs/data_use/) | The program's extraction, deduplication, and harmonization pipeline (`ai4data.data_use`) | Chapters [6](./detect.mdx) and [7](./harmonize.mdx) |
| [GLiNER](https://arxiv.org/abs/2311.08526) | Zero-shot named-entity recognition, the model family the extractor uses, with the program's adapter for dataset mentions | Chapter 6 |
| [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) | Fuzzy string matching used in the program's harmonization | Chapter 7 |
| [sentence-transformers](https://www.sbert.net/) | Sentence embeddings for semantic matching in the program's harmonization | Chapter 7 |

## Records and links

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [World Bank metadata schemas](https://github.com/worldbank/metadata-schemas) | `data_sources` in document records; `aliases` and `citation_requirement` in series records | The Development Data Group's catalogs | Chapters 1 and [9](./act.mdx) |
| [NADA](https://nada.ihsn.org/) | Catalog with citations lists per study | Runs the [Microdata Library](https://microdata.worldbank.org/) | Chapters 8 and 9 |
| [schema.org Dataset](https://schema.org/Dataset) | `citation` on dataset pages | NADA writes schema.org on catalog pages | Chapter 9 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
