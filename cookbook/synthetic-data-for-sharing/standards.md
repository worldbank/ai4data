---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The methods, measures, tools, and program resources each chapter builds on, with the ones the World Bank uses marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published methods and measures wherever one exists. Where
the program's workstreams provide a tool, the guide uses it.

## Methods and tools

| Resource | Use | World Bank use | In this guide |
|---|---|---|---|
| [REaLTabFormer](https://github.com/worldbank/REaLTabFormer) ([paper](https://arxiv.org/abs/2302.02041)) | Transformer-based synthesis of tabular and relational data with a memorization control (MIT) | The program's synthetic data workstream | Chapters [2](./method.mdx), [4](./risk.mdx), and [5](./relational.mdx) |
| [synthpop](https://cran.r-project.org/package=synthpop) ([paper](https://doi.org/10.18637/jss.v074.i11)) | Sequential synthesis with CART and parametric methods; utility and disclosure measures | | Chapters 2, [3](./utility.mdx), and 4 |
| [SDV](https://github.com/sdv-dev/SDV) and [SDMetrics](https://github.com/sdv-dev/SDMetrics) | Copula, network, and multi-table synthesizers (Business Source License); quality and privacy metrics (MIT) | | Chapters 2 to 5 |
| [sdcMicro practice guide](https://sdcpractice.readthedocs.io/) | Disclosure control practice and risk measures | Written at the World Bank | Chapter 4 |

## Measures

| Measure | Use | In this guide |
|---|---|---|
| Total variation distance; Cramér's V; Pearson correlation | Marginal and association comparisons | Chapter 3 |
| Propensity-score utility (pMSE) and confidence interval overlap ([Snoke et al., 2018](https://doi.org/10.1111/rssa.12358)) | General and analysis-specific utility | Chapter 3 |
| Distance to closest record; attribute inference; membership inference ([Shokri et al., 2017](https://arxiv.org/abs/1610.05820)) | Disclosure risk of synthetic data | Chapter 4 |

## Documentation and governance

| Standard | Use | In this guide |
|---|---|---|
| [World Bank microdata schema](https://github.com/worldbank/metadata-schemas) | The catalog record of the synthetic file, with links to the real survey | Chapter [6](./release.mdx) |
| [Metadata Editor](https://worldbank.github.io/metadata-editor-docs) and [NADA](https://nada.ihsn.org/) | Creating the record and publishing the file with its access type | Chapter 6 |
| [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/) | The licence for open synthetic files | Chapter 6 |
| [UNECE, Synthetic Data for Official Statistics: A Starter Guide](https://unece.org/statistics/publications/synthetic-data-official-statistics-starter-guide) | The reference introduction and use cases for statistical organizations | Chapters [1](./purpose.mdx) and [7](./workflow.mdx) |
| [Access chapter of the microdata cookbook](/cookbook/microdata-documentation/access) | Tiers, disclosure control records, and the access workflow | Chapters 1, 4, 6, and 7 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
