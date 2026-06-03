---
license: unknown
task_categories:
- object-detection
- image-segmentation
tags:
- pdf
- document-layout-analysis
- data-extraction
language:
- en
- fr
- es
size_categories:
- n<1K
configs:
- config_name: annotations
  data_files:
  - split: unhcr
    path: "annotations/unhcr/*.json"
  - split: prwp
    path: "annotations/prwp/*.json"
  - split: refugee
    path: "annotations/refugee/*.json"
- config_name: metadata
  data_files:
  - split: unhcr
    path: "metadata/unhcr/*.json"
  - split: prwp
    path: "metadata/prwp/*.json"
  - split: refugee
    path: "metadata/refugee/*.json"
- config_name: documents
  data_files:
  - split: unhcr
    path: "documents/unhcr/*.pdf"
  - split: prwp
    path: "documents/prwp/*.pdf"
  - split: refugee
    path: "documents/refugee/*.pdf"
- config_name: snapshots
  data_files:
  - split: unhcr
    path: "snapshots/unhcr/*.png"
  - split: prwp
    path: "snapshots/prwp/*.png"
  - split: refugee
    path: "snapshots/refugee/*.png"
---

# Dataset card for data-snapshot

## Dataset summary
The `data-snapshot` dataset is an annotated corpus designed for the evaluation and development of models for extracting *data snapshots* from PDF documents. A **data snapshot** is defined as a figure or table that contains quantitative data derived from statistics, indicators, or structured data sources.

## Dataset structure

The repository is organized as follows:

```
ai4data/data-snapshot/
├── annotations/<source>/*.json  # Contains annotation files per document
├── documents/<source>/*.pdf     # Actual PDFs
├── metadata/<source>/*.json     # Document-level metadata
├── schemas/*.json               # Provides the schema of the annotation and metadata files
├── snapshots/<source>/*.png     # Image files corresponding to the annotations
└── README.md
```

### Subsets
- `annotations`
  - JSON files that indicate the data snapshots: their object class (Figure / Table) and bounding box locations (in normalized `[x1, y1, x2, y2]` format, top-left origin)
  - Follows the schema provided in `schemas/data-snapshot-eval-v1.3.schema.json`
  - Provided on a per-document basis; documents that do not have data snapshots will still have an annotation file present but list of bounding boxes will be empty.
- `documents`
  - Actual PDF files that were annotated
- `metadata`
  - Document-level metadata following the [World Bank Metadata Standards (Chapter 5 — Documents)](https://worldbank.github.io/schema-guide/chapter05.html), schema provided in `schemas/metadata_schema.json`.
  - Provided on a per-document basis
  - All files across all sources share a uniform schema (same keys at every nesting level)
- `snapshots`
  - PNG files extracted from the documents and bounding box locations

### Sources
- UNHCR
- PRWP
- Refugee

## Loading the dataset using HF's `datasets` library

### Annotations

```python
>>> from datasets import load_dataset
>>> annotations = load_dataset("ajdajd/data-snapshot", name="annotations", split="unhcr")
>>> annotations[0]  # Inspect the first record
{'label_map': {'1': 'Figure', '2': 'Table'}, 'info': {'schema_version': '1.3', 'type': 'ground_truth', 'created_at': datetime.datetime(2026, 5, 20, 13, 44, 29), 'run_id': 'human-annotation-combined-e3432dce89', 'model': {'name': 'human annotation'}, 'coordinate_system': {'type': 'normalized_xyxy', 'range': [0.0, 1.0], 'origin': 'top_left'}}, 'documents': [{'doc_id': '06072015-baalbek-hermelgovernorateprofile.pdf', 'doc_name': '06072015-baalbek-hermelgovernorateprofile.pdf', 'doc_path': 'pdf_input/06072015-baalbek-hermelgovernorateprofile.pdf'}], 'predictions': [{'page_id': '06072015-baalbek-hermelgovernorateprofile.pdf::p000', 'doc_id': '06072015-baalbek-hermelgovernorateprofile.pdf', 'page_index': 0, 'objects': [{'id': '1d69f693', 'label': 'Figure', 'bbox': [0.029415499554572243, 0.1766403810171256, 0.5954839424856321, 0.7354445202645015], 'score': None}, ...}
```

### Metadata

```python
>>> metadata = load_dataset("ajdajd/data-snapshot", name="metadata", split="unhcr")
>>> metadata[0]  # Inspect the first record
{'type': 'document', 'metadata_information': {'title': 'Lebanon: Baalbek-Hermel Governorate Profile (June 2015)', 'idno': '06072015-baalbek-hermelgovernorateprofile', 'producers': [{'name': 'UNHCR', 'abbr': 'UNHCR', 'affiliation': 'UNHCR', 'role': 'Source'}], 'production_date': datetime.datetime(2026, 5, 21, 0, 0), ...}
```

### Documents

```python
>>> docs = load_dataset("ajdajd/data-snapshot", data_dir="documents/unhcr")  # Or simply data_dir="documents/" for all files
>>> docs.save_to_disk("path/to/docs_directory")  # Files are saved as an Arrow file
```

### Snapshots

```python
>>> snapshots = load_dataset("ajdajd/data-snapshot", data_dir="snapshots/unhcr")  # Or simply data_dir="snapshots/" for all snapshots
>>> snapshots.save_to_disk("path/to/snapshots_directory")  # Files are saved as an Arrow file
```

## Schema

### Annotations

The annotation files follow the **Data Snapshot Evaluation Format (v1.3)**. Below is a simplified, human-readable example of the JSON schema with explanatory comments for each field.

> **Note**: You will notice a top-level field called `predictions`. In the context of this dataset, this is a misnomer because these are actually human-labeled **annotations** (ground truth). We use the key `predictions` because we borrow this schema from the project's evaluation codebase, which uses a unified structure for both ground truth and model predictions.

```json
{
  // Canonical mapping of integer IDs to class names
  "label_map": {
    "1": "Figure",
    "2": "Table"
  },
  
  // High-level metadata about the file
  "info": {
    "schema_version": "1.3",
    "type": "ground_truth",  // Indicates these are human annotations
    "created_at": "2026-05-20T13:44:29",
    "run_id": "human-annotation-combined-e3432dce89",
    "model": {
      "name": "human annotation"
    },
    "coordinate_system": {
      "type": "normalized_xyxy",
      "range": [0.0, 1.0],  // Bounding boxes are normalized between 0 and 1
      "origin": "top_left"
    }
  },
  
  // List of documents referenced in this file
  "documents": [
    {
      "doc_id": "1_advocacy_note_mineaction_-_niger_eng.pdf",
      "doc_name": "1_advocacy_note_mineaction_-_niger_eng.pdf",
      "doc_path": "pdf_input/1_advocacy_note_mineaction_-_niger_eng.pdf"
    }
  ],
  
  // Per-page container of objects; these contain the ground truth annotations
  "predictions": [
    {
      "page_id": "1_advocacy_note_mineaction_-_niger_eng.pdf::p001",
      "doc_id": "1_advocacy_note_mineaction_-_niger_eng.pdf",
      "page_index": 0,  // 0-indexed page number
      "objects": [
        {
          "id": "obj_001",
          "label": "Figure",  // Matches a label_map entry
          "bbox": [0.029, 0.177, 0.595, 0.735],  // Normalized [x_min, y_min, x_max, y_max]
          "score": null  // Always null for ground truth
        }
      ]
    }
  ]
}
```

### Metadata

The metadata files follow the [**World Bank Document Metadata Schema**](https://worldbank.github.io/schema-guide/chapter05.html). See `schemas/metadata_schema.json` for the formal JSON schema definition.

All metadata files across all sources share a uniform schema (same keys at every nesting level, same types) to ensure compatibility with Apache Arrow and HuggingFace streaming.

Top-level fields:
-  `type`
- `metadata_information`
- `document_description`
- `provenance`
- `tags`
- `schematype`
- `additional` - contains source-specific fields (e.g. `additional.unhcr_*` for UNHCR, `additional.wds_*` for WDS API-sourced datasets).

## Dataset creation
The annotations were produced through human labeling using Label Studio.

## Licensing information
[TBD]

## Citation information
[TBD]
