# Document Snapshot Detection — Evaluation Specification  
Version: 1.3  
Status: Active  
Last Updated: 2026-02-22  

---

## 1. Purpose

This specification defines the evaluation contract for layout detection models used to extract **data snapshots** (Figures and Tables) from PDF documents.

The evaluation framework measures:

1. Detection accuracy (object-level correctness)
2. Spatial accuracy (region-level overlap quality)

All model outputs must conform to the Unified Evaluation Schema v1.3.

---

## 2. Scope

### Supported Object Classes

- Figure
- Table

Additional classes may be added in future versions but must be reflected in `label_map`.

---

## 3. Coordinate System

All bounding boxes must satisfy:

- Format: `normalized_xyxy`
- Structure: `[x1, y1, x2, y2]`
- Range: `[0.0, 1.0]`
- Origin: `top_left`

Constraints:

- `0 ≤ x1 < x2 ≤ 1`
- `0 ≤ y1 < y2 ≤ 1`

All evaluation is resolution-agnostic.

---

## 4. Data Contract

Each evaluation run requires:

- Ground Truth file (type = ground_truth)
- Prediction file (type = prediction)

Both must conform to:

- Unified Evaluation Schema v1.3
- Shared `label_map`
- Shared coordinate system

### Required Top-Level Keys

- `label_map`
- `info`
- `documents`
- `predictions`

---

## 5. Matching Procedure

Evaluation is performed per:

- Document
- Page
- Class (Figure/Table)

### 5.1 IoU Computation

Intersection over Union (IoU):

IoU = |Prediction ∩ GroundTruth| / |Prediction ∪ GroundTruth|

### 5.2 IoU Threshold

Default detection threshold:

- T = 0.5

Optional stricter threshold:

- T = 0.75

### 5.3 Matching Strategy

- Compute IoU matrix between predicted and GT boxes
- Filter candidate pairs where IoU ≥ T
- Sort candidates by IoU descending
- Perform greedy 1:1 assignment
  - Each GT can match at most one prediction
  - Each prediction can match at most one GT

Unmatched predictions → False Positives  
Unmatched GT boxes → False Negatives  

---

## 6. Detection Metrics

Metrics are computed using micro-averaging across the entire corpus.

### 6.1 True Positives (TP)

Number of matched prediction–GT pairs.

### 6.2 False Positives (FP)

Number of unmatched predictions.

### 6.3 False Negatives (FN)

Number of unmatched GT objects.

### 6.4 Precision@T

Precision = TP / (TP + FP)

### 6.5 Recall@T

Recall = TP / (TP + FN)

Metrics are reported per class.

F1 is optional and not required in baseline reporting.

---

## 7. Spatial Accuracy Metrics

Spatial metrics are computed only for matched pairs.

Let:

- I = |Prediction ∩ GT|
- P = |Prediction|
- G = |GT|

### 7.1 IoU

IoU = I / (P + G − I)

### 7.2 Coverage (Area Recall)

Coverage = I / G

Interpretation:
Fraction of ground-truth area captured by the prediction.

Equivalent to pixel-level recall.

### 7.3 Purity (Area Precision)

Purity = I / P

Interpretation:
Fraction of predicted area belonging to the ground-truth object.

Equivalent to pixel-level precision.

### 7.4 Aggregation

For each class:

- Mean IoU
- Mean Coverage
- Mean Purity

---

## 8. Aggregation Policy

Primary reporting uses micro-averaging:

- Aggregate TP, FP, FN globally
- Compute Precision and Recall from global totals

Spatial metrics are averaged across all matched objects.

Per-document macro averages are optional but not primary.

---

## 9. Validation Rules

Before evaluation:

1. `schema_version` must equal "1.3"
2. `label_map` must be identical between GT and prediction files
3. Every prediction `doc_id` must exist in `documents`
4. All bounding boxes must be within [0,1]
5. For prediction files:
   - Each object must contain `score`
6. For ground truth files:
   - No object may contain `score`

Cross-field validations are enforced in code.

---

## 10. Reporting Template

For each model:

### Detection Metrics (T=0.5)

Per Class:
- Precision
- Recall

### Optional Detection Metrics (T=0.75)

Per Class:
- Precision
- Recall

### Spatial Metrics

Per Class:
- Mean IoU
- Mean Coverage
- Mean Purity

---

## 11. Design Rationale

This evaluation framework separates:

- Object detection quality (did we detect the object?)
- Spatial extraction quality (did we crop it correctly?)

This is necessary because snapshot lifting requires:

- High Recall (avoid missing objects)
- High Coverage (avoid losing titles/metadata)
- Reasonable Purity (avoid excessive over-cropping)

---

## 12. Future Extensions

Possible future additions:

- mAP (COCO-style evaluation)
- Class weighting
- Per-document macro metrics
- Segmentation mask support
- Error taxonomy reporting

---

End of Specification