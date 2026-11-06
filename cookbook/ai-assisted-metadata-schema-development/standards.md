---
id: standards
title: Standards reviewed in this guide
sidebar_position: 90
description: The standards and reference vocabularies compared in the research, their chapter roles, and the limits of their adoption.
---

# Standards reviewed in this guide

These references were compared during the Data Snapshot study. They provide
terminology, semantic distinctions, and representation patterns for
[chapter 7](./standards-refinement.mdx). A reviewed relationship does not
establish formal adoption, conformance, or measured interoperability.
The World Bank use column records the research comparison only.

## Resource description and provenance

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [DCMI Metadata Terms](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/) | General resource meanings such as title, source, and coverage inform field comparisons. | The study compared relevant terms. | Chapter 7 examines local field scope. |
| [Schema.org](https://schema.org/Dataset) | Cross-domain resource and measurement properties provide candidate mappings. | The study compared relevant terms. | Chapter 7 distinguishes exact and bounded relationships. |
| [DCAT 3](https://www.w3.org/TR/vocab-dcat-3/) | Catalog, spatial, and temporal description supply reference patterns. | The study compared relevant terms. | Chapter 7 considers application fit. |
| [PROV-O](https://www.w3.org/TR/prov-o/) | Derivation and attribution distinguish data origins from credited agents. | The study used this distinction in refinement. | Chapter 7 explains the provenance-field split. |
| [JATS 1.3](https://jats.nlm.nih.gov/publishing/1.3/) | Publishing conventions provide figure-label and grouping patterns. | The study compared relevant patterns. | Chapter 7 distinguishes terminology from deferred structure. |
| [SKOS](https://www.w3.org/TR/skos-reference/) | Concept schemes and mappings provide vocabulary reference patterns. | The study compared relevant patterns. | Chapters 4 and 7 keep the concept inventory distinct from formal ontology engineering. |

## Statistical and operational concepts

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [SDMX 3.1](https://sdmx.org/standards-2/) | Dimensions, frequency, and units provide statistical reference concepts. | The study compared relevant terms. | Chapter 7 preserves local scope when code lists do not fit exactly. |
| [DDI Lifecycle 3.3](https://docs.ddialliance.org/DDI-Lifecycle/3.3/) | Variables, universe, categories, geography, and collection supply descriptive references. | The study compared relevant terms. | Chapters 5 and 7 separate conceptual meaning from operational representation. |
| [IATI 2.03](https://reference.iatistandard.org/en/iati-standard/203/) | Project, finance, organization, and location terms provide domain-specific references. | The study compared relevant terms. | Chapter 7 examines mappings that cover only part of a broader local meaning. |
| [RDF Data Cube](https://www.w3.org/TR/vocab-data-cube/) and [OWL-Time](https://www.w3.org/TR/owl-time/) | Statistical dimensions and time relations provide additional representation patterns. | The study compared relevant patterns. | Chapter 7 identifies richer structures whose implementation was deferred. |

## Values, formats, and visualization references

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) | Date and period formats inform value guidance. | The study reviewed applicable guidance. | Chapter 7 preserves source precision and avoids unsupported normalization. |
| [ISO 3166](https://www.iso.org/iso-3166-country-codes.html) and [UN M49](https://unstats.un.org/unsd/methodology/m49/) | Country and area identifiers inform optional geographic mappings. | The study reviewed applicable guidance. | Chapter 7 requires an exact and contextually supported mapping. |
| [ISO 4217](https://www.iso.org/iso-4217-currency-codes.html) | Currency identifiers inform value guidance. | The study reviewed applicable guidance. | Chapter 7 distinguishes explicit denominations from ambiguous symbols. |
| [BCP 47](https://www.rfc-editor.org/info/bcp47) | Language tags provide reference identifiers. | The study reviewed applicable guidance. | Chapter 7 avoids inferring an unexpressed region or script. |
| [UN/CEFACT Recommendation 20](https://unece.org/code-list-recommendations) | Unit codes provide optional normalized values. | The study reviewed applicable guidance. | Chapter 7 preserves units that lack an exact supported mapping. |
| [Vega-Lite](https://vega.github.io/vega-lite/docs/) | Visualization terminology provides a partial reference vocabulary. | The study reviewed relevant terminology. | Chapter 7 retains unfamiliar source-grounded visualization forms. |

## Research basis and authoring status

Dy & Solatorio (2026). *Artifact-Driven Metadata Schema Development: A Human-in-the-Loop Methodology with Multimodal LLMs*. Working paper.

Appendix F documents the standards review and its sources. Chapter 7 now provides
worked crosswalk, adjudication, and inventory-reconciliation exercises. The
historical review was interactive and human-adjudicated; the guide does not
supply a fixed historical prompt or automated standards pipeline.

## Worked-term verification

The following authoritative definitions were rechecked on **2026-11-07** for
chapter 7. These checks are separate from the historical study's source dates.
The table records the terms used in the worked comparisons; it does not imply
that every term in the broader research crosswalk was reverified during chapter
authoring.

| Worked terms | Authoritative documentation and source status |
|---|---|
| DCMI title, subject, and source | [Title](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/terms/title/), [Subject](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/terms/subject/), and [Source](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/terms/source/). The DCMI Metadata Terms recommendation is dated 2020-01-20. |
| PROV-O derivation and attribution | [wasDerivedFrom](https://www.w3.org/TR/prov-o/#wasDerivedFrom) and [wasAttributedTo](https://www.w3.org/TR/prov-o/#wasAttributedTo), in the W3C Recommendation dated 2013-04-30. |
| Schema.org measured variables and projects | [variableMeasured](https://schema.org/variableMeasured) and [Project](https://schema.org/Project). These canonical term pages are mutable and identify themselves as the development version. |

The manuscript records Schema.org access on 2026-09-23 without a pinned release.
The new check date must not be substituted for that historical access date.
Current definitions support the chapter's teaching comparison; the retained
research decision matrix supplies the historical relationship judgments.
Crosswalk labels such as Exact and Standard narrower are descriptive research
categories. They do not assert formal vocabulary mappings or certify compliance.

## Corrections

A correction should identify the field, authoritative source, proposed
relationship, and application-specific reason. The
[contributing page](./contributing.md) describes evidence expectations.
