---
trigger: always_on
---

# Project Context — Data Snapshot Extraction Evaluation

## Objective

The objective of this project is to **evaluate existing layout detection solutions** for extracting *data snapshots* from PDF documents.

A **data snapshot** is defined as:

> A figure or table that contains quantitative data derived from statistics, indicators, or structured data sources.

The project does **not involve training or developing new detection models**.
Instead, the focus is on testing and comparing **off-the-shelf or existing solutions** against a human-annotated corpus.

The goal is to identify solutions that are suitable for downstream snapshot extraction workflows.

---

## Evaluation Framework

Model performance is evaluated according to the official evaluation contract defined in:

* `evaluation_spec.md`

The framework measures two main aspects:

### 1. Detection Accuracy

* True Positives (TP)
* False Positives (FP)
* False Negatives (FN)
* Precision and Recall at configurable IoU thresholds (default = 0.5)

Matching between predictions and ground truth follows:

* IoU computation between bounding boxes
* Threshold filtering
* Greedy 1:1 assignment
* Evaluation performed per document, per page, and per object class (Figure / Table)

Metrics are aggregated using **micro-averaging across the corpus**.

### 2. Spatial Accuracy

Computed only for matched prediction–ground-truth pairs:

* Mean IoU
* Coverage (Area Recall)
* Purity (Area Precision)

These metrics measure the quality of spatial localization independently from detection correctness.

---

## Data Contract

All evaluation inputs must conform to the **Unified Evaluation Schema v1.3**, defined in:

* `data-snapshot-eval-v1.3.schema.json`

Each evaluation run requires:

* One Ground Truth JSON file
* One Prediction JSON file

Both must:

* Use normalized bounding box coordinates in `[x1, y1, x2, y2]` format
* Use coordinate range `[0,1]`
* Use `top_left` origin
* Contain document-level metadata and per-page object containers

Prediction objects may include a confidence `score`.

---

## Adapter Development

Each layout detection solution must be integrated via an **adapter module**.

The adapter is responsible for converting raw model outputs into the Unified Evaluation Schema v1.3 format.

Adapter responsibilities include:

* Converting coordinates into normalized XYXY format
* Mapping model-specific labels to the canonical `label_map`
* Generating consistent `doc_id` and `page_id`
* Populating prediction metadata (`run_id`, `model_name`, etc.)
* Ensuring schema compliance prior to evaluation

Adapters enable standardized and fair comparison across heterogeneous model outputs.

---

## Ground Truth

Ground truth annotations are produced through **human labeling**.

They define:

* Snapshot existence
* Snapshot class (Figure / Table)
* Snapshot bounding box location

Ground truth is treated as the authoritative reference during evaluation.

---

## Expected Outcomes

The project aims to:

* Develop a robust evaluation framework for snapshot detection
* Develop adapter codes for each layout detection solution under evaluation
* Generate comparable performance metrics across models
