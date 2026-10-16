---
id: standards
title: Standards and tools
sidebar_label: Standards and tools
sidebar_position: 10
description: The standards, formats, and tools the recipes of this guide rely on, with the chapter in which each applies.
---

# Standards and tools

The recipes rely on the following standards, formats, and tools. Each
row names where it applies in the guide.

| Standard or tool | What it is | Chapters |
|---|---|---|
| [Croissant 1.0](https://docs.mlcommons.org/croissant/docs/croissant-spec.html) | MLCommons format for describing machine-learning datasets in JSON-LD on schema.org: files with checksums, record sets and typed fields, splits. The scripts write 1.0 records, which the validator checks; version 1.1 adds provenance and usage policies. | 3, 6 |
| [mlcroissant](https://github.com/mlcommons/croissant) | The Python implementation: a validator against the specification and a loader that reads a dataset from its record (Apache 2.0). | 3 |
| [schema.org Dataset](https://schema.org/Dataset) | The vocabulary Croissant extends; the markup on a dataset page that search engines read. | 3, 7 |
| [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards) | The card format repositories render, with a YAML header of licence, language, task, and size. | 4 |
| [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) | Gebru and others (2018): the questions on motivation, composition, collection, preprocessing, uses, distribution, and maintenance that a card answers. | 1, 4, 5 |
| [Data Cards](https://arxiv.org/abs/2204.01075) | Pushkarna and others (2022): structured summaries of a dataset's essential facts for responsible use. | 4 |
| [Apache Parquet](https://parquet.apache.org/) | Columnar file format for large tabular datasets, read by every machine-learning library. | 2 |
| [Data Package](https://datapackage.org/) | Frictionless container format with a table schema per resource, an alternative to the dictionary CSV. | 2 |
| [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) | The default open licence, with a URL that tools resolve. | 7 |
| [DataCite metadata schema](https://schema.datacite.org/) | The fields for a DOI per dataset version, including relations between versions. | 8 |
| [sdcMicro practice guide](https://sdcpractice.readthedocs.io/) | The disclosure control workflow the unit applies to a new product. | 7 |
| [ISCO-08](https://ilostat.ilo.org/methods/concepts-and-definitions/classification-occupation/) | The occupation classification of the running example's label. | 1 to 8 |
| [World Bank metadata schemas](https://github.com/worldbank/metadata-schemas) | The catalog record the dataset's card and licence feed, with the Metadata Editor's Croissant export. | 3, 7 |
