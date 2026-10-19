---
id: data-types
title: Application by data type
sidebar_position: 11
hide_table_of_contents: true
description: How the nine questions and the recipes apply to indicators, survey microdata, geospatial data, documents, and tables, and which standard describes each type.
---

# Application by data type

The nine questions apply to every kind of data an organization publishes. What
changes by type is the record schema, the access format, what "meaning"
consists of, and the risks. This page gives the mapping. The running
example in the chapters uses indicators; the files list on the
[overview](./index.mdx) has a record for each of the other types, and the
metadata check in [recipe 1.1](./find.mdx) runs on all of them.

## Record schema and standards by type

The World Bank's [metadata schemas](https://github.com/worldbank/metadata-schemas)
cover each type with a JSON Schema that follows the international standard
for that type. Using them means a record can move between the organization's
catalog, NADA, the Metadata Editor, and the World Bank's own catalogs
without translation. Two open-source tools from the World Bank and IHSN
work with all of these types: the
[Metadata Editor](https://github.com/worldbank/metadata-editor) for
documenting (templates, validation, export to SDMX metadata structures,
schema.org, Croissant, and DCAT, publishing to NADA), and
[NADA](https://nada.ihsn.org/) for cataloguing (schema.org markup on each
page, keyword and semantic search, a REST API, an MCP interface, and
managed access to microdata).

| Type | World Bank schema | Based on | Access format | Interface | What "meaning" consists of | Identifier |
|---|---|---|---|---|---|---|
| Indicators and time series | `timeseries-schema.json`, `timeseries-db-schema.json` | SDMX concepts | SDMX-CSV, SDMX-JSON, tidy CSV | SDMX REST API, REST with OpenAPI, MCP | Definition, unit, periodicity, code lists, series breaks | `idno`; DOI through DataCite |
| Survey and administrative microdata | `microdata-schema.json` | [DDI Codebook](https://ddialliance.org/ddi-codebook) | CSV, Stata, SPSS files with a DDI data dictionary | Catalog API (NADA), access-conditioned download | Universe, sampling, weights, variables with value labels, questionnaire | `idno`; DOI through DataCite |
| Geospatial data | `geospatial-schema.json` | [ISO 19115/19139](https://www.iso.org/standard/53798.html) | GeoPackage, GeoJSON, Shapefile, Cloud Optimized GeoTIFF | [OGC API Features](https://ogcapi.ogc.org/features/), WMS/WFS, [STAC](https://stacspec.org/) for imagery | Coordinate reference system, extent, resolution, feature catalogue, lineage | `idno`; DOI through DataCite |
| Documents and publications | `document-schema.json` | [Dublin Core](https://www.dublincore.org/specifications/dublin-core/) | PDF with extracted text; tables and figures extracted as data | Catalog API, full-text search | Abstract, type, data sources cited, structured table of contents | `idno`; DOI |
| Statistical tables | `table-schema.json` | World Bank table schema | CSV with [CSV on the Web](https://www.w3.org/TR/tabular-data-primer/) metadata | Catalog API | Rows, columns, unit of observation, footnotes, definitions | `idno` |
| Scripts and reproducibility packages | `script-schema.json` | World Bank script schema | Code and data in a versioned archive | Catalog API, code hosting | Inputs, outputs, software, steps | `idno`; DOI |
| Images and video | `image-schema.json`, `video-schema.json` | [IPTC](https://iptc.org/standards/photo-metadata/), Dublin Core | Image and video files | Catalog API | Caption, location, date, rights | `idno` |

## The nine questions by data type

| Question | Indicators | Microdata | Geospatial | Documents |
|---|---|---|---|---|
| **1. Find** | Name, definition, keywords, aliases in the record; schema.org Dataset on the page | Title, abstract, keywords, and the variable labels, which are what users search for | Title, abstract, keywords, bounding box and topic category; DCAT and schema.org with `spatialCoverage` | Title, abstract, keywords, and the full text; schema.org `Report` or `ScholarlyArticle` |
| **2. Retrieve** | SDMX or REST API by series, area, period | Download under the stated access conditions; a data dictionary API for variables | OGC API Features or WFS for vectors, tiles or COG for rasters; a download in an open format | The PDF, the extracted text, and the tables and figures as data ([Data Snapshots](https://arxiv.org/abs/2606.06242)) |
| **3. Understand** | Definition, unit, periodicity, code lists, series breaks | Universe, sampling, weights, variable definitions and value labels, questionnaire | CRS, extent, resolution, feature catalogue (attribute definitions), lineage | Document type, data sources cited, structured table of contents |
| **4. Ask** | Search by meaning over records; answers from the API | Search over variable labels across surveys; answers that name the survey, the variable, and the universe | Search over titles and keywords with a map filter; answers that state the CRS and the edition | Full-text search; answers that cite the page |
| **5. Trust** | Source, release, `OBS_STATUS`; number check against the API | Citation requirement, version, DOI; access conditions stated with the data | Edition and gazette date, lineage, DOI | DOI, publication date, bibliographic citation |
| **6. Evaluate** | Known-item questions per language | Known-variable questions ("age of household head") across surveys | Known-layer questions with a place name | Known-document questions and page-level citation checks |
| **7. Monitor use** | Mentions of series names and acronyms | Mentions of survey names and acronyms (the main case for mention detection) | Mentions of layer names; service request logs | Citations and downloads |
| **8. Govern** | Public data; low risk | Disclosure control before any exposure; access tiers; no microdata to external services | Sensitive locations; resolution limits | Embargoes; confidential annexes |
| **9. Sustain** | Same for all types: open standards, separable AI layer, versioned profiles and question sets | | | |

## Metadata check by data type

The checker selects the schema from the profile's `type` and validates the
record against it, so the same command works for each type. The profiles
are starting points for the organization to edit.

```bash
python check_metadata.py example_catalog.csv      profile_indicator.json
python check_metadata.py example_microdata.json   profile_microdata.json
python check_metadata.py example_geospatial.json  profile_geospatial.json
python check_metadata.py example_document.json    profile_document.json
```

Nested records in JSON are what NADA and the Metadata Editor export. CSV
works for flat types such as indicators; for microdata, geospatial data,
and documents, use JSON.

## Principal differences between data types

- **Microdata** needs chapter 8 before chapter 2. Disclosure control and
  access tiers come before any API or agent interface, and microdata never
  go to an external AI service. The variables are also the main search
  target: users look for "age of household head" more often than for a survey title,
  which is what the program's [Metadata Augmentation](/docs/metadata-augmentation/)
  work addresses by grouping variables into themes.
- **Geospatial data** carry their meaning in the coordinate reference
  system, the extent, and the feature catalogue. A layer without a stated
  CRS is unusable by software, whatever the description says. OGC services
  are the equivalent of the SDMX API for this type.
- **Documents** are where most official statistics are still locked. The
  data inside them (tables, figures) become retrievable only when extracted
  and published next to the document, and the `data_sources` field links a
  document to the datasets it used, which is what [chapter 7](./monitor-use.mdx)
  needs.
- **Tables** published as files need column definitions and the unit of
  observation in the record; CSV on the Web carries the column types.
