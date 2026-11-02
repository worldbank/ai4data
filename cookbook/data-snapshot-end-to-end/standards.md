---
id: standards
title: Standards and contracts used in this guide
sidebar_position: 90
description: This page identifies the representation contracts and references used by the recipes.
---

# Standards and contracts used in this guide

## Metadata representation

| Standard or contract | Use | World Bank use | In this guide |
|---|---|---|---|
| [Data Snapshot Metadata Schema v1.4](/docs/data-snapshots/metadata-schema/) | This project contract represents semantic metadata for one data snapshot. | The production snapshot package uses the canonical Pydantic models. | Chapters 3 and 4 validate and review records. |

## Evaluation and provenance

| Reference | Use | In this guide |
|---|---|---|
| [Snapshot benchmark paper](https://arxiv.org/html/2606.06242v1) | This reference defines analytical regions and distinguishes detection from spatial extraction quality. | Chapters 4 and 8 describe review and future evaluation. |
| [Pydantic validation](https://docs.pydantic.dev/latest/concepts/models/) | This library validates the canonical model. | The inspection script checks representation; factual accuracy requires review. |

These references do not imply certification of metadata against external registries or legal permission to redistribute source publications.
