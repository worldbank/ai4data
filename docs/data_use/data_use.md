# Monitoring of Data Use

## Overview

Understanding how development data is used, in research, policy documents, program evaluations, and news reporting, is essential for measuring the impact of data programs and for prioritizing future data investments. The Data Use Monitoring workstream applies natural language processing (NLP) to extract and track **mentions of specific datasets** from large document collections.

When a policy report cites the World Development Indicators, a research paper uses DHS survey data, or a government document references a national statistical publication, these references constitute evidence of data use. Systematically extracting and cataloging such references provides insight into which datasets have the most influence, where data gaps exist, and how data quality improvements translate into downstream impact.

---

## The Challenge of Dataset Mention Extraction

Extracting dataset mentions from free text is more difficult than it might appear:

- Dataset names are highly variable: "World Development Indicators," "WDI," "World Bank Development Data," and "the Bank's development indicators" all refer to the same resource.
- Dataset names overlap with other named entities (organizations, projects, publications) and common phrases.
- The same survey series has different editions, vintages, and country-specific variants ("DHS 2019," "Ethiopia DHS 2016," "Demographic and Health Surveys Program").
- Documents span many formats: PDFs, web pages, Word documents, reports, and academic papers.

Standard keyword search finds only exact matches. The `ai4data.data_use` module addresses this with **named entity recognition (NER)** trained specifically for dataset mentions.

---

## How It Works

The extraction pipeline uses **GLiNER2** (`fastino/gliner2-large-v1`), a schema-driven successor of GLiNER, the generalist NER model that identifies entities of arbitrary types from a label prompt (Zaratiana et al., 2023). The base model is loaded once and fine-tuned LoRA adapters trained by the program are applied on it: `ai4data/datause-extraction` for the mention spans, `ai4data/datause-relation-v0` for the relations of each mention (organization, acronym, timeframe, geography, user), and `ai4data/datause-impact-v0` for the classification of usage, typology, and purpose. A classifier adapter (`ai4data/datause-classifier`) screens chunks before extraction so that boilerplate pages are skipped. Implementation: [`src/ai4data/data_use/models/model_manager.py`](../../src/ai4data/data_use/models/model_manager.py).

### Pipeline Overview

```
Document (PDF, text) → Text Extraction → NER (GLiNER2 with adapters) → Raw Mentions
        → Harmonization → Deduplication → Structured Output
```

**1. Text Extraction** — PDFs are parsed using PyMuPDF. Text is extracted by page, preserving document structure. Implementation: [`src/ai4data/data_use/utils/document_parser.py`](../../src/ai4data/data_use/utils/document_parser.py).

**2. NER** — The GLiNER2 model with the extraction adapter identifies spans in the extracted text that represent dataset mentions. The model returns each span with a confidence score. Implementation: [`src/ai4data/data_use/extractors/dataset_extractor.py`](../../src/ai4data/data_use/extractors/dataset_extractor.py).

**3. Harmonization** — Raw mentions ("WDI," "World Development Indicators," "World Bank Development Data") are mapped to canonical dataset identifiers using fuzzy string matching (RapidFuzz) and semantic similarity (sentence-transformers). Implementation: [`src/ai4data/data_use/extractors/harmonization.py`](../../src/ai4data/data_use/extractors/harmonization.py).

**4. Deduplication** — Near-duplicate mentions within the same document or across a corpus are deduplicated. Implementation: [`src/ai4data/data_use/extractors/deduplication.py`](../../src/ai4data/data_use/extractors/deduplication.py).

---

## Quick Start

```python
# Install with data use dependencies
# uv pip install ai4data[datause]

from ai4data.data_use import extract_from_document, extract_from_text

# A whole PDF (path or URL): one result per page, with the page text and
# the mentions found on it. Reference pages are skipped by default.
results = extract_from_document("policy_report.pdf", include_confidence=True)
for page in results:
    for ds in page["datasets"]:
        name = ds.get("mention_name") or {}
        print(page["page"], repr(name.get("text")), name.get("confidence"), ds.get("specificity_tag"))

# A text string: one dict with the input text and the mentions.
result = extract_from_text("We used the 2019 DHS and the World Development Indicators.", include_confidence=True)
print(result["datasets"])
```

`extract_from_document` reads the PDF with `DocumentParser` ([`utils/document_parser.py`](../../src/ai4data/data_use/utils/document_parser.py)); the same class can be called directly with `load_pdf_chunks(...)` or `load_pdf_as_markdown(...)` when the text is needed for other steps. Each mention is a dict whose `mention_name` entry carries the span text, its character offsets, and a confidence; the other keys hold the tag (named, descriptive, or vague), the typology, whether the dataset is used in the document, and the relations found for it.

**Example output** (illustrative; values depend on the document and the model version):
```
3 'World Development Indicators' 0.94 named_data
3 'DHS 2019' 0.88 named_data
7 'national poverty line estimates' 0.72 descriptive_data
```

---

## Harmonization and Deduplication

Raw extraction produces many surface forms of the same dataset. Harmonization maps these to a canonical identifier:

```python
from ai4data.data_use import deduplicate_extraction, harmonize_datasets, initialize_harmonization_env

# Deduplicate the mentions of one document (output of extract_from_text or extract_from_document)
dedup = deduplicate_extraction(results, fuzzy_threshold=85)

# Harmonize the deduplicated outputs of a corpus into canonical families.
# input_folder holds one folder per project with its dedup/*.json files;
# the results (families, aliases, country-specific prototypes) are written to output_dir.
env = initialize_harmonization_env()
harmonize_datasets(
    input_folder="./extraction_outputs/",
    output_dir="./harmonization_outputs/",
    env=env,
    sim_threshold=0.5,
)
```

The harmonization step uses a combination of:
- **Exact match** against a known dataset name dictionary
- **Fuzzy match** (token sort ratio ≥ threshold) via RapidFuzz
- **Semantic match** (cosine similarity of sentence-transformer embeddings) for cases where string similarity fails

---

## Downstream Applications

Once dataset mentions are extracted and harmonized, the structured output supports:

- **Impact measurement**: Count how many documents cite each dataset, by year, region, topic, or document type.
- **Gap analysis**: Identify countries or topics where data is produced but not cited—a signal of discoverability or relevance issues.
- **Evidence base tracking**: Map the citation chain from dataset to policy decision.
- **Catalog enrichment**: Use extracted citations as a signal to improve dataset metadata (datasets that are often cited together may share themes or user populations).

---

## Installation

```bash
uv pip install ai4data[datause]
```

Optional harmonization (adds sentence-transformers for semantic matching):

```bash
uv pip install ai4data[harmonization]
```

---

## Limitations

- Recall on informal mentions is lower than on formal citations with identifiers. The [monitoring cookbook](/cookbook/monitoring-data-use/) combines citation services, access counts, and mention extraction so that each covers what the others miss.
- Harmonization needs a dictionary of the organization's dataset names and their variants; without it, mentions are extracted and left unmapped.
- The extractor reads text. Mentions inside figures, scanned pages without OCR, and tables rendered as images are missed.
- A mention shows that a dataset was used. Whether the cited figure is current is a separate check.
- Status on the [tools catalogue](/tools#data-use): pre-release.
