# Metadata Augmentation Methodology

This section describes the methodology implemented by the World Bank's Development Data Group for **automatically generating DDI-style variable groups** for data dictionary variables in microdata and administrative datasets. The approach uses semantic clustering and LLM-elicited curation to organize hundreds or thousands of survey variables into meaningful groups—without manual labeling.

The methodology is implemented in the `ai4data.metadata.augmentation` package. A step-by-step notebook is provided: Microdata Theme Generation with LLMs.

**Please cite this methodology as follows:**
World Bank. 2025. "Metadata Augmentation Methodology." AI for Data – Data for AI. Available at [https://worldbank.github.io/ai4data](https://worldbank.github.io/ai4data).

---

## The Problem

Microdata catalogs—household surveys, censuses, administrative registers—are among the richest sources of development data. A single Demographic and Health Survey (DHS) may contain 500–1,000 variables capturing household composition, health behaviors, asset ownership, economic activity, and more. The World Bank Microdata Library catalogs thousands of such surveys.

These variables are documented in **data dictionaries**: machine-readable files listing each variable's name, label, value codes, and sometimes a question text. Data dictionaries are essential for any analyst working with microdata—but they present a navigation challenge: with hundreds of variables, finding the right ones for a specific research question requires reading through the entire list.

**Thematic organization** solves this problem by grouping variables into coherent topics: "Child Health and Nutrition," "Household Asset Ownership," "Agricultural Land Use," "Women's Empowerment." Thematic tags make data dictionaries navigable, improve catalog search, and enable cross-survey comparability by aligning conceptually similar variables.

But thematic organization has historically been manual—a time-consuming task that is routinely skipped or done inconsistently. The result: most microdata catalogs have poorly organized or entirely unthematized variable lists.

This pipeline automates the first pass: generating thematic proposals that data curators can review, accept, or correct.

---

## Non-Technical Summary

The methodology is organized in five steps:

**Step 1. Loading the data dictionary**
The pipeline loads a data dictionary from a CSV or JSON file (or a NADA microdata catalog). Each variable is represented as a structured record with a name, label, optional description, and optional value codes. Adapters handle diverse source formats, mapping source column names to a canonical schema.

**Step 2. Embedding variable labels and descriptions**
Each variable is encoded as a dense numerical vector using a sentence-transformer model (`BAAI/bge-small-en-v1.5` by default). The embedding captures the *semantic meaning* of the variable label: variables about child nutrition will cluster together even if their labels use different words ("stunting prevalence," "height-for-age Z-score," "child wasting"). For large dictionaries, dimensionality reduction (TruncatedSVD) is applied to improve clustering quality.

**Step 3. Clustering into semantic groups**
Variables are grouped using Agglomerative Clustering (Ward linkage). The number of clusters is either specified by the user or estimated automatically using silhouette score optimization. A post-processing step ensures that each cluster's variable list fits within the LLM context window.

**Step 4. Curating variable groups with an LLM**
For each cluster, the pipeline calls an LLM (Claude, OpenAI, or Gemini via litellm) and asks it to curate a DDI-style variable group: select the variables that belong, and produce a label, description, definition, and optional universe/notes. The LLM output is structured JSON, validated against a Pydantic schema. Deterministic fields (`vgid`, `group_type`, `variable_groups`) are assembled by the framework.

**Step 4b. Self-consistency QA**
After curation, a second LLM call (the QA agent) assesses whether the label, description, definition, and selected variables are mutually coherent. Groups that fail QA are flagged (`qa_passed=False`) but kept in the output for human review.

**Step 5. Assembling and exporting the augmented dictionary**
The generated variable groups and curated variable assignments are assembled into an `AugmentedDictionary` object that can be exported to JSON or converted to a pandas DataFrame for further use.

---

## Design Rationale

**Why clustering before LLM?** Sending all variables to the LLM at once would exceed context limits for large dictionaries and produce poor theme quality (too broad, too generic). Clustering first ensures that each LLM call receives a semantically coherent subset of variables that can be meaningfully summarized in 2–6 words.

**Why sentence-transformers instead of INSTRUCTOR?** Modern general-purpose embedding models (`bge-small`, `MiniLM`) achieve competitive semantic similarity performance on short text (variable labels) without requiring task-specific instruction prefixes. They are also faster, lighter, and easier to deploy. INSTRUCTOR models were used in earlier iterations of this work (in the `llm4data` project) but are no longer necessary.

**Why Ward linkage?** Ward linkage minimizes within-cluster variance, producing compact, well-separated clusters that are semantically coherent. Unlike k-means, it is deterministic and does not require centroid initialization. For semantic embeddings, it consistently outperforms average and complete linkage in producing interpretable clusters.

**Why LLM elicitation rather than cluster labeling heuristics?** Heuristic approaches (e.g., using the most frequent word in a cluster's variable labels) produce labels that are often too technical, too generic, or not meaningful to non-specialist users. LLMs can generate human-readable theme names that accurately reflect the cluster's conceptual content.

---

## Contents

- [Methodology: Technical Details](methodology.md) — Embeddings, clustering, token budget management, LLM configuration
- `ai4data.metadata.augmentation` — Python package implementation

---

## Quick Start

```python
from ai4data.metadata.augmentation import DataDictionaryAugmentor

# One-call usage
augmentor = DataDictionaryAugmentor()
result = augmentor.augment("data_dictionary.csv")
augmentor.export("augmented_dictionary.json")

# Inspect results
for group in result.variable_groups:
    print(f"{group.vgid}: {group.label} (qa_passed={group.qa_passed})")
    print(f"  {group.txt}")
    print(f"  Variables: {group.variables}")
    print()

# Filter groups that failed self-consistency QA
failed = [g for g in result.variable_groups if g.qa_passed is False]
print(f"{len(failed)} groups failed QA")
```

Step-by-step usage with custom settings:

```python
from ai4data.metadata.augmentation import (
    DataDictionaryAugmentor,
    ConfigurableDictionaryAdapter,
)

# Custom column mapping for non-standard CSV
adapter = ConfigurableDictionaryAdapter({
    "variable_name": "var_id",
    "label": "variable_label",
    "description": "question_text",
})

augmentor = DataDictionaryAugmentor(
    model="gpt-4o-mini",    # any litellm-supported model
    n_clusters=15,          # specify cluster count, or omit for auto-detection
)

result = (
    augmentor
    .load("custom_dictionary.csv", adapter=adapter)
    .embed(show_progress_bar=True)
    .cluster()
    .generate_variable_groups()
)

# Export to DataFrame
df = augmentor.to_dataframe()
print(df.head())
```

---

## Installation

```bash
uv pip install ai4data[metadata]
```

Requirements: Python ≥ 3.11, sentence-transformers, scikit-learn, litellm, pydantic.
An API key for your chosen LLM provider must be set as an environment variable (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
