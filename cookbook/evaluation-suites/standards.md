---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The measures, methods, and program tools each chapter builds on, with the ones the World Bank uses marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published measures and methods wherever one exists. Where
the program's workstreams have an evaluation practice, the guide uses it.

## Measures

| Measure or method | Use | In this guide |
|---|---|---|
| Recall@k, MRR, nDCG ([Manning, Raghavan, and Schütze, *Introduction to Information Retrieval*, chapter 8](https://nlp.stanford.edu/IR-book/)) | Retrieval quality with known relevant items, and with graded relevance | Chapter [3](./retrieval.mdx) |
| Precision, recall, F1 | Extraction and classification quality; exact and lenient span matching as in the [CoNLL](https://www.clips.uantwerpen.be/conll2003/ner/) and [SemEval](https://semeval.github.io/) traditions | Chapter [5](./extraction.mdx) |
| Cohen's kappa ([Cohen, 1960](https://doi.org/10.1177/001316446002000104)) | Agreement between labellers, and between a grader and humans, corrected for chance | Chapters [2](./questions.mdx) and [6](./judges.mdx) |
| Bootstrap confidence intervals ([Efron and Tibshirani, 1993](https://doi.org/10.1201/9780429246593)) | Intervals on pass rates and on paired differences between runs | Chapters [7](./gates.mdx) and [9](./reporting.mdx) |
| Paired comparison | The same questions through two versions, so that the difference is between versions and not between question sets | Chapter 7 |
| Stratified sampling | Review samples that cover every language and intent, with oversampling of failures reported separately | Chapter [8](./production.mdx) |

## Program methods and tools

| Resource | Use | In this guide |
|---|---|---|
| [PI-FT toolkit](/pift-toolkit/pipeline) and [Data Discoverability](/docs/data-discoverability/) | Retrieval evaluation with Recall@k, MRR, and nDCG, and graded evaluation with a model as judge | Chapters 3 and 6 |
| [Proof-Carrying Numbers](https://arxiv.org/abs/2509.06902) | Verification of stated numbers against the source, the basis of numeric accuracy | Chapter [4](./answers.mdx) |
| [Monitoring of Data Use](/docs/data_use/) | Mention extraction with labelled samples and precision and recall | Chapter 5 |
| [Metadata Reviewer](/docs/metadata-reviewer/overview) | Human review of model suggestions with recorded decisions, the source of curation evaluation sets | Chapter [2](./questions.mdx) |
| The evaluation chapters of the other cookbooks ([dissemination](/cookbook/ai-ready-dissemination/evaluate), [agents](/cookbook/serving-statistics-to-agents/evaluate), [monitoring](/cookbook/monitoring-data-use/detect), [curation](/cookbook/metadata-curation-with-llms/improve)) | The component-specific suites this guide generalizes | All chapters |

## Frameworks and formats

| Standard | Use | In this guide |
|---|---|---|
| [Inspect](https://inspect.ai-safety-institute.org.uk/) | An open-source evaluation framework for language model systems with tool use and graders | Chapters 4, 6, and 7 |
| [ranx](https://github.com/AmenRa/ranx) and [BEIR](https://github.com/beir-cellar/beir) | Retrieval evaluation libraries and benchmark formats | Chapter 3 |
| [Model cards](https://arxiv.org/abs/1810.03993) and [datasheets for datasets](https://arxiv.org/abs/1803.09010) | Documentation formats for the component and for the question set | Chapter [9](./reporting.mdx) |
| [ISO/IEC 42001](https://www.iso.org/standard/42001) and [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) | Management and risk frameworks that require measured performance and monitoring | Chapters [1](./inventory.mdx) and 8 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
