---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The management, risk, quality, and licensing standards each chapter builds on, with the ones the World Bank uses marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards wherever one exists. Where the World
Bank's Development Data Group has adopted a standard, the guide uses that
one.

## Principles and management

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [UN Fundamental Principles of Official Statistics](https://unstats.un.org/fpos/) | Professional independence, confidentiality, and transparency of methods, which the policy restates for AI use and the statement of model use serves | The basis of the Development Data Group's statistical work | Chapters [1](./position.mdx) and [7](./statement.mdx) |
| [OECD AI Principles](https://oecd.ai/en/ai-principles) | Principles many governments have adopted, which the policy can cite | | Chapter 1 |
| [ISO/IEC 42001](https://www.iso.org/standard/42001) | An AI management system: policy, records, management review, corrective action | | Chapters 1, [2](./register.mdx), [4](./decisions.mdx), [8](./review.mdx) |
| [Committee for the Coordination of Statistical Activities](https://unstats.un.org/unsd/ccsa/) | The recommendations for national statistical organizations on AI-ready official statistics, which the assessment framework's questions map to | The framework's questions map to its recommendations | Chapter 8 |

## Risk and quality

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) | Govern, map, measure, manage; the structure the chapters follow | | Chapters 1 and [3](./risk.mdx) |
| [ISO/IEC 23894](https://www.iso.org/standard/77304.html) | Guidance on AI risk management, compatible with the three ratings and the tiers | | Chapter 3 |
| [UN National Quality Assurance Frameworks manual](https://unstats.un.org/unsd/methodology/dataquality/) | The quality reporting the statement of model use is part of | | Chapter 7 |
| [European Statistics Code of Practice](https://ec.europa.eu/eurostat/web/quality/european-statistics-code-of-practice) | Sound methodology and transparency principles | | Chapter 7 |
| [OWASP Top 10 for LLM applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | The risks, including prompt injection, that the controls of the High tier address | | Chapter 3 |

## Licences and terms

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [Creative Commons licences](https://creativecommons.org/licenses/) and [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | Open licences for data and publications | [CC BY 4.0](https://datacatalog.worldbank.org/public-licenses) for World Bank open data | Chapter [6](./licensing.mdx) |
| [Open Data Commons licences](https://opendatacommons.org/licenses/) | Open licences for databases | | Chapter 6 |
| [SPDX licence list](https://spdx.org/licenses/) | Short identifiers for licences in records and fields | | Chapter 6 |
| [schema.org Dataset](https://schema.org/Dataset), [DCAT 3](https://www.w3.org/TR/vocab-dcat-3/), and [Croissant](https://mlcommons.org/croissant/) | The structured fields that carry the licence to AI systems | The Metadata Editor and NADA export schema.org and DCAT | Chapter 6 |
| [Open Source AI Definition](https://opensource.org/ai/open-source-ai-definition) and [OpenRAIL licences](https://www.licenses.ai/) | What open and responsible-AI model licences allow | | Chapters [5](./providers.mdx) and 6 |
| [Model cards](https://arxiv.org/abs/1810.03993) | The documentation convention a register row can link to | | Chapter 2 |

## Program methods and tools

| Resource | Use | In this guide |
|---|---|---|
| [AI-readiness assessment framework](/ai-readiness-assessment) | The dimensions and questions the review maps the artefacts to | Chapter 8 |
| [Operational companion](/ai-readiness-in-practice) | Program resources and evidence per dimension | Chapter 8 |
| [Proof-Carrying Numbers](https://arxiv.org/abs/2509.06902) | Verified numbers in generated text, the control on the public assistant | Chapter 3 |
| [Efficient and Inclusive AI](/docs/inclusive-ai/) | Provider-agnostic configuration and local open-weight models | Chapter 5 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
