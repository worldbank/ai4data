# Efficient and Inclusive AI Applications

## Overview

As AI tools become central to data quality, discoverability, and analysis, a critical question emerges: *who benefits?* The most capable AI models are large, expensive, and predominantly trained on English-language text. Without deliberate design choices, AI for development data risks amplifying existing inequalities—better tools for well-resourced institutions, poorer ones for the rest.

The Efficient and Inclusive AI Applications workstream addresses this directly. It focuses on making AI capabilities accessible and practical for data practitioners, researchers, and institutions in low- and middle-income countries (LMICs), particularly those working with limited compute, budgets, and English-language coverage.

---

## The Efficiency Dimension

Not all tasks require the most powerful (and most expensive) LLM. Many data quality and metadata tasks can be addressed effectively with smaller, more efficient models:

### Choosing the Right Model for the Task

| Task | Suitable Model Size | Rationale |
|---|---|---|
| Metadata field generation (title, description) | Small–Medium (Claude Haiku, GPT-4o mini) | Short outputs, well-defined structure |
| Anomaly explanation | Medium (Claude Sonnet, GPT-4o) | Requires world knowledge and reasoning |
| Semantic embedding | Small (all-MiniLM, BGE-small) | Embedding quality saturates at smaller sizes for short texts |
| Multi-step data analysis | Large (Claude Opus, GPT-4) | Complex reasoning, long context |
| Classification with schema | Small–Medium | Structured output reduces hallucination risk |

The ai4data project defaults to efficient models where appropriate. For example, `DataDictionaryAugmentor` defaults to `claude-haiku-4-5-20251001` for variable group curation—a fast, cost-effective choice for short, structured outputs. The model can be overridden for tasks requiring more reasoning depth.

### Batch Processing

Large-scale processing (e.g., explaining thousands of anomalies or augmenting a large microdata catalog) is made practical through batch APIs. OpenAI and Gemini both offer asynchronous batch endpoints that are typically 50% cheaper than synchronous calls and support higher throughput. The anomaly explanation pipeline uses batch mode by default for production runs (see [Running at Scale](../anomaly/explanation/elicitation-pipeline.md#running-at-scale)).

### On-device and Open-weight Models

For organizations that cannot use third-party cloud APIs—due to data sensitivity, cost, or connectivity—the pipeline is designed to be provider-agnostic. Through `litellm`, the metadata augmentation and other LLM-dependent workflows can be configured to use locally hosted open-weight models (e.g., Llama 3, Mistral, Qwen) via Ollama or similar local inference servers:

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

**Translation as preprocessing.** For LLM-based tasks (anomaly explanation, metadata generation), translating non-English metadata to English before the LLM call—using a translation model or API—often yields better results than prompting directly in the source language.

**Culturally grounded prompting.** For country-specific contexts, prompts that include the country name and relevant regional context improve LLM explanation quality, particularly for less-documented geographies. The anomaly explanation prompts include geography name as a required context field precisely for this reason.

### Capacity Building

Technical tools are only inclusive if practitioners can use them. The program supports:

- **Open-source code and documentation** — All tools in this repository are MIT-licensed and documented for independent deployment.
- **Jupyter notebooks** — Each methodology is accompanied by a step-by-step notebook that can be run on free platforms (Google Colab, Kaggle) without a local compute setup.
- **Configurable infrastructure** — Tools default to free-tier or low-cost API options, with clear documentation for switching to local or alternative backends.

### Data Equity

The Data for AI dimension of the program also has an inclusion component: ensuring that development data from LMICs is well-represented, well-documented, and AI-ready. If AI systems are trained and evaluated primarily on data from high-income countries, they will perform less well for low-income country contexts. Improving metadata quality, discoverability, and structural documentation for LMIC datasets directly contributes to more equitable AI.

---

## Guiding Principles

The program's approach to inclusive AI is guided by:

1. **Default to efficiency** — Use the smallest model that achieves adequate quality for the task.
2. **Support provider choice** — Never hardcode a specific API provider; use abstractions (litellm) that allow institutional choice.
3. **Build for low-connectivity contexts** — Prefer offline-capable or cached components where possible.
4. **Document for all levels** — Provide both technical documentation for engineers and accessible summaries for non-technical stakeholders.
5. **Measure inclusion** — Track language coverage, country coverage, and cost profiles of deployed tools.
