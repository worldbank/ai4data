---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The published standards and datasets each chapter builds on, with the ones the World Bank has adopted marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards wherever one exists. Where the World
Bank's Development Data Group has adopted a standard for its own catalogs,
the guide uses that one.

## Documents and tables

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [World Bank document schema](https://github.com/worldbank/metadata-schemas) | Record for a document (Dublin Core based) with `data_sources`, `toc_structured`, identifiers, and citation | The Development Data Group's catalogs; the Metadata Editor; NADA | Chapter [1](./inventory.mdx) |
| [World Bank table schema](https://github.com/worldbank/metadata-schemas) | Record for a statistical table: columns, rows, data sources, definitions, licence, citation, relations | The same catalogs and tools | Chapter [7](./publish.mdx) |
| [Dublin Core](https://www.dublincore.org/specifications/dublin-core/) | Core metadata elements for documents | The basis of the document schema | Chapter 1 |
| [Metadata Editor](https://github.com/worldbank/metadata-editor) | Documentation of documents and tables with templates and validation; publishing to NADA | Used by the Development Data Group's curators | Chapters 1 and 7 |
| [NADA](https://nada.ihsn.org/) | Catalog for documents, tables, microdata, and indicators, with schema.org markup, search, and a REST API; semantic search and an MCP server through [NADA AI](/tools#nada-ai), which is being integrated into NADA | Runs the [Microdata Library](https://microdata.worldbank.org/) | Chapters 1, 7, and 8 |

## Layout detection and extraction

| Standard or resource | Use | In this guide |
|---|---|---|
| [Data Snapshots](https://arxiv.org/abs/2606.06242) (Dy and Solatorio) | Benchmark of open-source layout detection models on institutional documents | Chapter [3](./locate.mdx) |
| [data-snapshot dataset](https://huggingface.co/datasets/ai4data/data-snapshot) | 3,908 annotated regions (Figure, Table) across 476 PDFs with normalized XYXY bounding boxes, from UNHCR reports, World Bank policy research working papers, and refugee documents; MIT licence | Chapter 3: annotation format and test set |
| [COCO format](https://cocodataset.org/#format-data) | Interchange format for object detection annotations and tools | Chapter 3 |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF text, page rendering, and table extraction; used by the program's document parser | Chapters 1 and 4 |
| [PDF/A (ISO 19005)](https://www.pdfa.org/resource/iso-19005-pdfa/) | Archival PDF with a text layer, the output of the OCR stage | | Chapter 2 |
| [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) | OCR to searchable PDF/A with Tesseract | | Chapter 2 |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Open-source OCR for scanned documents | Chapters 1 and 2 |

## Files, provenance, and citation

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [CSV on the Web](https://www.w3.org/TR/tabular-data-primer/) | Column types and annotations for tidy files | | Chapters 4 and 7 |
| [W3C PROV](https://www.w3.org/TR/prov-overview/) | Provenance of derived data (`wasDerivedFrom`) | | Chapter 7 |
| [schema.org Dataset](https://schema.org/Dataset) | Markup on table pages, with `isBasedOn` and `distribution` | NADA writes schema.org on catalog pages | Chapter 8 |
| [DataCite](https://schema.datacite.org/) | DOIs for documents and extracted tables | The Microdata Library assigns DataCite DOIs to datasets, and the Open Knowledge Repository assigns DOIs to publications (prefix 10.1596) | Chapter 8 |
| [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) | Licence inherited by extracted data | The default for [World Bank datasets](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets); World Bank publications carry CC BY 3.0 IGO | Chapter 7 |
| SDMX `OBS_STATUS` | The model for a status per value (verified, flagged, estimated) | | Chapter 5 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
