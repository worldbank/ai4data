# Efficient and Inclusive AI Applications

## Overview

The most capable AI models are large, expensive, and trained mostly on English-language text. Without deliberate design choices, AI for development data gives better tools to well-resourced institutions and poorer ones to the rest.

The Efficient and Inclusive AI Applications workstream addresses this directly. It focuses on making AI capabilities accessible and practical for data practitioners, researchers, and institutions in low- and middle-income countries (LMICs), particularly those working with limited compute, budgets, and English-language coverage.

---

## The Efficiency Dimension

Many data quality and metadata tasks can be done with smaller models at lower cost. The table gives the program's working guidance by task:

### Choosing the Right Model for the Task

| Task | Suitable Model Size | Rationale |
|---|---|---|
| Metadata field generation (title, description) | Small–Medium (Claude Haiku, GPT-4o mini) | Short outputs, well-defined structure |
| Anomaly explanation | Medium (Claude Sonnet, GPT-4o) | Requires world knowledge and reasoning |
| Semantic embedding | Small (all-MiniLM, BGE-small) | Embedding quality saturates at smaller sizes for short texts |
| Multi-step data analysis | Large (Claude Opus, GPT-4) | Complex reasoning, long context |
| Classification with schema | Small–Medium | Structured output reduces hallucination risk |

The ai4data project defaults to capable models where appropriate. For example, `DataDictionaryAugmentor` defaults to `claude-sonnet-4-6` for variable group curation and self-consistency QA. The model can be overridden independently for curation (`model`) and QA (`qa_model`) to balance cost and quality.

### Batch Processing

Large-scale processing (e.g., explaining thousands of anomalies or augmenting a large microdata catalog) is made practical through batch APIs. OpenAI and Gemini both offer asynchronous batch endpoints that are typically 50% cheaper than synchronous calls and support higher throughput. Batch mode is recommended for large runs of the anomaly explanation pipeline (see [Running at Scale](../anomaly/explanation/elicitation-pipeline.md#running-at-scale)).

### On-device and Open-weight Models

For organizations that cannot use third-party cloud APIs, because of data sensitivity, cost, or connectivity, the pipeline is provider-agnostic. Through `litellm`, the metadata augmentation and other LLM-dependent workflows can be configured to use locally hosted open-weight models (e.g., Llama 3, Mistral, Qwen) via Ollama or similar local inference servers:

```python
from ai4data.metadata.augmentation import DataDictionaryAugmentor

# Use a locally hosted model via Ollama
augmentor = DataDictionaryAugmentor(model="ollama/llama3.1")
result = augmentor.augment("variables.csv")
```

---

## The Inclusion Dimension

### Low-Resource Languages

Most LLMs perform significantly better on English than on other languages, and substantially worse on low-resource languages spoken in Sub-Saharan Africa, South Asia, and other regions where the World Bank's development data programs are concentrated. Several mitigations are relevant:

**Multilingual embedding models.** Models such as `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) and `LaBSE` support over 50 languages with strong cross-lingual alignment. Semantic search and clustering can use these models to operate across multilingual metadata without translation.

**Translation as preprocessing.** For LLM-based tasks such as anomaly explanation and metadata generation, translating non-English metadata to English before the LLM call, with a translation model or API, often gives better results than prompting in the source language. The result is measured on the organization's own question set before the choice is fixed.

**Culturally grounded prompting.** For country-specific contexts, prompts that include the country name and relevant regional context improve LLM explanation quality, particularly for less-documented geographies. The anomaly explanation prompts include geography name as a required context field precisely for this reason.

### Capacity Building

Tools are usable in practice only where practitioners can run them. The program provides:

- **Open-source code and documentation.** The code in this repository is MIT-licensed and documented for independent deployment.
- **Jupyter notebooks.** Notebooks accompany the anomaly detection, metadata quality, and Metadata Reviewer workstreams under `notebooks/`. They run in a local Jupyter environment; a hosted notebook platform needs the same API keys and dependencies.
- **Configurable model and provider.** Each LLM-dependent tool takes the model and provider as configuration. The defaults are hosted models (the augmentor defaults to `claude-sonnet-4-6`); batch APIs reduce the cost of large runs, and local open-weight models are an option where the code routes through `litellm`.

### Data Equity

The Data for AI dimension of the program also has an inclusion component: ensuring that development data from LMICs is well-represented, well-documented, and AI-ready. If AI systems are trained and evaluated primarily on data from high-income countries, they will perform less well for low-income country contexts. Improving metadata quality, discoverability, and structural documentation for LMIC datasets directly contributes to more equitable AI.

---

## Guiding Principles

The program's approach to inclusive AI is guided by:

1. **Efficiency by default.** The smallest model that reaches adequate quality for the task is the one used.
2. **Provider choice.** No tool hardcodes an API provider; the abstraction layer (litellm) leaves the choice to the institution.
3. **Low-connectivity contexts.** Offline-capable or cached components are preferred where they exist.
4. **Documentation at every level.** Technical documentation for engineers and plain summaries for non-technical readers.
5. **Measured inclusion.** Language coverage, country coverage, and cost profiles of deployed tools are tracked.

---

## Limitations

- The model size table above is guidance from the program's own tasks, and the model names in it date quickly; the sizes are the point, and the names are examples from the date of writing. Each organization measures its tasks with its own evaluation suite before choosing a size; the [small and open models cookbook](/cookbook/small-and-open-models/) gives the comparison method.
- Cost figures for batch processing are those the providers publish and change with their pricing.
- Multilingual embedding models cover many languages unevenly, and coverage of a language in a model's training data is rarely documented. A question set in the organization's language is the only measure that holds.
- Running models locally needs hardware and operations capacity that the [hardware chapter](/cookbook/small-and-open-models/serving) sizes.
