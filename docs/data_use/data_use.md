# Monitoring of Data Use

## Overview

Understanding how development data is actually used—in research, policy documents, program evaluations, and news reporting—is essential for measuring the impact of data programs and for prioritizing future data investments. The Data Use Monitoring workstream applies natural language processing (NLP) to extract and track **mentions of specific datasets** from large document collections.

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

The extraction pipeline uses **GliNER** (Generalist and Lightweight NER), a zero-shot NER model that identifies named entities of arbitrary types given a label prompt at inference time (Zaratiana et al., 2023). By prompting GliNER with the entity type "dataset," "survey," or "data source," the model can identify dataset mentions even in novel contexts without task-specific fine-tuning.

### Pipeline Overview

```
Document (PDF, text) → Text Extraction → NER (GliNER) → Raw Mentions
        → Harmonization → Deduplication → Structured Output
```

**1. Text Extraction** — PDFs are parsed using PyMuPDF. Text is extracted by page, preserving document structure. Implementation: [`src/ai4data/data_use/utils/document_parser.py`](../../src/ai4data/data_use/utils/document_parser.py).

**2. NER** — The GliNER model identifies spans in the extracted text that represent dataset mentions. The model returns each span with a confidence score. Implementation: [`src/ai4data/data_use/extractors/dataset_extractor.py`](../../src/ai4data/data_use/extractors/dataset_extractor.py).

**3. Harmonization** — Raw mentions ("WDI," "World Development Indicators," "World Bank Development Data") are mapped to canonical dataset identifiers using fuzzy string matching (RapidFuzz) and semantic similarity (sentence-transformers). Implementation: [`src/ai4data/data_use/extractors/harmonization.py`](../../src/ai4data/data_use/extractors/harmonization.py).

**4. Deduplication** — Near-duplicate mentions within the same document or across a corpus are deduplicated. Implementation: [`src/ai4data/data_use/extractors/deduplication.py`](../../src/ai4data/data_use/extractors/deduplication.py).

---

## Quick Start

```python
# Install with data use dependencies
# uv pip install ai4data[datause]

from ai4data.data_use.extractors.dataset_extractor import DatasetExtractor
from ai4data.data_use.utils.document_parser import extract_text_from_pdf

# Extract text from a PDF document
pages = extract_text_from_pdf("policy_report.pdf")
text = "\n".join(pages)

# Extract dataset mentions
extractor = DatasetExtractor()
mentions = extractor.extract(text)

for mention in mentions:
    print(f"{mention.span_text!r:40s}  score={mention.score:.2f}")
```

**Example output:**
```
'World Development Indicators'      score=0.94
'DHS 2019'                          score=0.88
'Demographic and Health Surveys'    score=0.91
'national poverty line estimates'   score=0.72
```

---

## Harmonization and Deduplication

Raw extraction produces many surface forms of the same dataset. Harmonization maps these to a canonical identifier:

```python
from ai4data.data_use.extractors.harmonization import HarmonizationAdapter

harmonizer = HarmonizationAdapter()
# Map raw mention to canonical dataset
canonical = harmonizer.map("WDI")
# Returns: {"canonical_name": "World Development Indicators",
#           "dataset_id": "WDI", "confidence": 0.98}
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
