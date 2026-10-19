---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The formats, tools, frameworks, and program resources each chapter builds on, with the ones the World Bank uses marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published formats and tools wherever one exists. Where
the program's workstreams provide measurements or tools, the guide uses
them.

## Model formats, cards, and licences

| Standard | Use | In this guide |
|---|---|---|
| [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards) and [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993) | The card format and sections a model needs before the organization considers it | Chapter [2](./candidates.mdx) |
| [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), [MIT](https://opensource.org/license/mit), and [Responsible AI Licenses](https://www.licenses.ai/) | The licence families the classification distinguishes | Chapter 2 |
| [safetensors](https://github.com/huggingface/safetensors) and GGUF | Weight formats that load without executing code | Chapters [3](./serving.mdx) and [6](./security.mdx) |
| [Sigstore model transparency](https://github.com/sigstore/model-transparency) | Signing and verifying model files | Chapter 6 |

## Serving and adaptation

| Tool | Use | In this guide |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggml-org/llama.cpp), [Ollama](https://ollama.com/) | Serving stacks from workstation to shared service | Chapter 3 |
| [Sentence Transformers](https://www.sbert.net/) and the [PI-FT toolkit](/pift-toolkit/pipeline) | Embedding models and their fine-tuning on metadata | Chapter [4](./adaptation.mdx) |
| [Hugging Face PEFT](https://github.com/huggingface/peft) | Parameter-efficient fine-tuning of small generators | Chapter 4 |

## Evaluation and governance

| Resource | Use | In this guide |
|---|---|---|
| [Evaluation suites cookbook](/cookbook/evaluation-suites/) | Suites, required scores, gates, report cards | Chapters [1](./when.mdx), 4, [5](./comparison.mdx), and 6 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard), [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | Public benchmarks for shortlisting | Chapters 2 and 5 |
| [Efficient and Inclusive AI Applications](/docs/inclusive-ai/) | The program's measurements of model size against task accuracy and its deployment guidance | Chapters 1, 3, 5, and [7](./cost.mdx) |
| [OWASP Top 10 for LLM applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | Supply chain and deployment risks | Chapter 6 |
| [AI component register](pathname:///cookbook-files/ai-ready-dissemination/ai_component_register.csv) | Where choices, versions, and shared assets are recorded | All chapters |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
