# Benchmarking Open-Source Layout Detection Models for Data Snapshot Extraction from Institutional PDFs

**Authors:**  
AJ Carl P. Dy \<ady@worldbank.org\>, Aivin V. Solatorio \<asolatorio@worldbank.org\>

**Code Repository:** https://github.com/ajdajd/data-snapshot-annotation  
**Dataset:** https://huggingface.co/datasets/ai4data/data-snapshot

---

# Abstract

Institutional PDF archives contain substantial amounts of operational and analytical information embedded within figures and tables. Existing document layout benchmarks largely optimize for generic layout extraction and do not distinguish semantically meaningful analytical content from irrelevant layout objects such as decorative images, logos, or tables of contents. This creates a gap between benchmark-oriented layout detection and operationally useful data extraction.

This work presents a benchmark dataset and evaluation framework for semantically meaningful figure and table extraction from institutional PDFs spanning humanitarian reports, World Bank policy research working papers, and project appraisal documents. We benchmark multiple off-the-shelf open-source layout detection systems and evaluate both detection accuracy and spatial extraction quality using metrics designed for downstream analytical usability.

Our results show that current document layout models struggle to generalize to operational institutional documents despite strong performance on conventional academic benchmarks. We release the dataset, source PDFs, annotations, and benchmarking code to support future research in operational document intelligence.

---

# 1. Introduction

## 1.1 Operational Knowledge in Institutional PDFs

Institutional PDF archives remain one of the largest repositories of operational knowledge in development, humanitarian response, public policy, and international finance. Organizations such as the World Bank, UNHCR, and humanitarian coordination clusters continuously publish reports containing statistical tables, monitoring dashboards, financing summaries, implementation matrices, geospatial maps, and analytical visualizations.

However, much of this information remains inaccessible to downstream AI systems because it is embedded within figures and tables rather than narrative text. Modern retrieval systems, OCR pipelines, and large language models typically prioritize extracted text streams, causing significant information loss when operational insights exist only inside visual structures.

This issue is particularly important in operational settings where narrative text may summarize only a narrow subset of findings while additional information remains embedded within tables and figures. Humanitarian reports may describe high-level trends in prose while detailed incident distributions remain visible only inside charts and dashboards. Policy research papers may discuss conclusions narratively while econometric outputs and comparative statistics remain embedded within tables. Project appraisal documents may describe interventions textually while financing allocations, procurement structures, and implementation details appear only within structured matrices.

## 1.2 Defining Data Snapshots

We refer to the extraction of semantically meaningful visual regions from institutional PDFs as **data snapshot extraction**.

A **data snapshot** is defined as a bounded visual region within a document page containing structured or semi-structured information intended for analytical interpretation or operational reuse. Examples include statistical tables, analytical charts, operational dashboards, financing matrices, monitoring summaries, and geospatial visualizations.

Importantly, not all figures and tables are data snapshots. Humanitarian and institutional PDFs frequently contain decorative photographs, logos, cover images, tables of contents, recommendation blocks, and formatting tables that are technically classified as figures or tables but do not contain reusable analytical information.

Existing layout benchmarks generally do not distinguish between generic layout objects and semantically meaningful data snapshots. As a result, models trained on these datasets are encouraged to optimize for broad layout extraction rather than operationally useful extraction.

This distinction creates what we refer to as the **data snapshot extraction gap**:
> the inability of current document AI systems to reliably identify and isolate semantically meaningful visual regions containing reusable operational information.

## 1.3 Why Specialized Snapshot Detection Still Matters

Recent multimodal large language models are capable of extracting information from figures and tables directly from page images. However, using general-purpose multimodal pipelines for large-scale institutional document processing remains computationally expensive.

In many operational workflows, only a small fraction of document content contains analytically relevant visual information. Data snapshots often occupy less than 30% of a page and may appear in only a handful of pages within long institutional reports. As a result, pipelines that process entire documents uniformly may spend substantial computational resources analyzing visually irrelevant content.

This creates a strong incentive for efficient snapshot localization systems that can isolate semantically meaningful visual regions prior to downstream processing. In practice, accurate snapshot localization can reduce both the total amount of visual content requiring expensive multimodal processing and the amount of irrelevant document context propagated into downstream extraction pipelines.

## 1.4 Benchmarking Existing Open-Source Systems

Despite recent advances in document layout analysis, most existing benchmarks focus on scientific publishing corpora, OCR-centric parsing tasks, or generic segmentation objectives. Operational institutional PDFs introduce substantially different challenges, including multilingual layouts, infographic-heavy pages, embedded dashboards, annex-heavy reports, mixed raster/vector content, and heterogeneous formatting accumulated across decades of institutional publishing.

This work presents a benchmark for evaluating semantically meaningful figure and table extraction from operational institutional PDFs. The benchmark spans humanitarian reports, policy research working papers, and project appraisal documents, emphasizing real-world operational document complexity rather than clean academic layouts.

Rather than proposing a new model architecture, our objective is to establish realistic baselines for current open-source capabilities. We benchmark multiple off-the-shelf document layout detection systems in order to evaluate how well existing models generalize to operational institutional document ecosystems.

Our work makes four primary contributions. First, we construct a benchmark dataset for semantically meaningful data snapshot extraction spanning humanitarian reports, policy research papers, and project appraisal documents. Second, we develop an evaluation framework that separates object detection quality from spatial extraction quality. Third, we benchmark multiple off-the-shelf open-source layout detection systems under operational document conditions. Finally, we release source PDFs, annotations, metadata, and benchmarking code to support future research in operational document intelligence.

---

# 2. Related Work

## 2.1 Document Layout Analysis

Document layout analysis has historically focused on segmentation and classification of document regions such as titles, paragraphs, tables, figures, and lists. Recent advances in transformer-based and multimodal architectures have substantially improved performance on layout understanding tasks, supported by large-scale benchmarks such as PubLayNet, DocLayNet, and TableBank.

These benchmarks have accelerated progress in document segmentation, OCR-aware parsing, and generic layout understanding. However, most existing datasets emphasize scientific publishing corpora and relatively clean academic layouts. More importantly, current layout benchmarks generally treat all figures and tables as equally relevant extraction targets.

This assumption creates a mismatch between benchmark objectives and operational workflows. In institutional PDFs, many layout objects classified as figures or tables do not contain reusable analytical information. Humanitarian reports frequently contain photographs, logos, or decorative imagery that are technically figures but are not meaningful data snapshots. Similarly, project appraisal documents often contain tables of contents, administrative formatting tables, or recommendation blocks that are technically tabular structures but are not operationally useful analytical objects.

As a result, models trained on generic layout benchmarks often optimize for broad layout extraction rather than semantically meaningful snapshot extraction. This distinction becomes particularly important in operational workflows where precision and semantic completeness directly affect downstream retrieval and analytical usability.

---

# 3. The data-snapshot Benchmark

## 3.1 Benchmark Motivation

The benchmark was motivated by practical operational workflows requiring scalable extraction of structured visual information from institutional PDF archives.

During early exploration, we surveyed existing datasets, layout detection systems, OCR pipelines, and multimodal document models. While existing systems demonstrated strong performance on conventional layout benchmarks, we found that they frequently extracted all figures and tables indiscriminately without distinguishing semantically meaningful data snapshots from irrelevant layout objects.

In practice, institutional PDFs contain many figures and tables that are not analytically useful. Humanitarian reports commonly include decorative photographs and cover images. Institutional reports often contain tables of contents, recommendation blocks, formatting tables, and administrative structures that do not contain reusable operational information. Existing benchmarks rarely capture this distinction.

At the same time, semantically meaningful operational snapshots often appear in highly complex layouts such as:
- infographic-heavy dashboards,
- monitoring summaries,
- compact statistical panels,
- embedded operational charts,
- annex-heavy reports.

These structures are underrepresented in conventional document layout datasets.

This motivated the creation of a dedicated benchmark focused specifically on semantically meaningful data snapshot extraction rather than generic layout segmentation.

## 3.2 Corpus Composition

The benchmark aggregates documents from three major institutional document families.

1. **UNHCR / ReliefWeb**. This corpus includes humanitarian and protection-related reports containing displacement summaries, operational dashboards, incident visualizations, protection monitoring summaries, and geospatial maps.

2. **Policy Research Working Papers (PRWP)**. This corpus includes World Bank Policy Research Working Papers containing econometric tables, comparative statistics, analytical figures, and dense academic layouts. Representative examples include infrastructure investment analysis and macroeconomic research papers.

3. **Project Appraisal Documents (PADs)**. This corpus includes refugee-related project appraisal documents containing financing matrices, procurement tables, implementation summaries, institutional diagrams, and results frameworks. Representative examples include refugee inclusion and health-sector support projects.

## 3.3 Dataset Statistics

| Corpus | PDFs | Pages | Figures | Tables |
|---|---:|---:|---:|---:|
| UNHCR / ReliefWeb | 338 | 2,765 | 2,198 | 282 |
| PRWP | 100 | 3,023 | 339 | 590 |
| Refugee PADs | 38 | 1,929 | 62 | 437 |
| **Total** | **476** | **7,717** | **2,599** | **1,309** |

## 3.4 Annotation Workflow

Annotations were produced using a semi-assisted human-in-the-loop workflow. Two pretrained layout detection systems were first used to generate candidate annotations:
- DocLayout-YOLO for figures
- YOLO11 for tables

These annotations were then manually reviewed and corrected page-by-page using Label Studio. All final annotations were manually verified by a human annotator.

This workflow significantly accelerated annotation throughput while maintaining strict human quality control.

[PLACEHOLDER — Annotation Workflow Figure]

[PLACEHOLDER — Example Annotation Visualization]

## 3.5 Released Assets

The benchmark release includes:
- original PDFs,
- document metadata,
- bounding box annotations,
- benchmarking code,
- evaluation scripts,
- model adapters.

The dataset is released publicly on HuggingFace, with evaluation and benchmarking code released on GitHub.

---

# 4. Evaluation Framework

## 4.1 Detection Evaluation

We evaluate data snapshot extraction as a bounding box detection task over two classes:
- Figure
- Table

Predicted bounding boxes are matched against ground-truth annotations using Intersection-over-Union (IoU). A prediction is considered correct if its IoU with a ground-truth object exceeds a threshold of 0.5. Matching is performed using greedy one-to-one assignment, ensuring that each prediction and each ground-truth object can participate in at most one match.

We report:
- Precision
- Recall

Precision measures the fraction of predicted snapshots that correspond to valid ground-truth objects, while Recall measures the fraction of ground-truth snapshots successfully detected by the model.

We intentionally report results using a fixed threshold of IoU = 0.5 rather than COCO-style mAP sweeping across multiple thresholds. Our objective is not generic object detection benchmarking but operational extraction usability. In practice, the more important question is whether a model successfully isolates a usable analytical snapshot rather than how detection performance varies across many IoU thresholds.

## 4.2 Spatial Extraction Quality

Conventional object detection metrics alone are insufficient for evaluating operational data snapshot extraction. A prediction may technically satisfy IoU = 0.5 while still excluding critical information such as:
- table headers,
- chart legends,
- subtitles,
- footnotes,
- axes labels.

To capture downstream extraction usability, we separately evaluate spatial extraction quality using three complementary metrics:
- Area Recall (Coverage)
- Area Precision (Purity)
- Intersection-over-Union (IoU)

### Area Recall (Coverage)

Area recall measures the fraction of the ground-truth region captured by the predicted bounding box.

High coverage is important because downstream extraction pipelines require semantically complete crops. Missing titles, legends, or explanatory footnotes may significantly reduce the usefulness of the extracted snapshot.

[PLACEHOLDER — Equation]

### Area Precision (Purity)

Area precision measures the fraction of the predicted bounding box belonging to the ground-truth object.

High purity indicates that the extracted crop contains minimal irrelevant surrounding content.

[PLACEHOLDER — Equation]

### Intersection-over-Union (IoU)

Unlike Area Recall and Area Precision, which separately measure completeness and cleanliness of extraction, Intersection-over-Union (IoU) provides a single metric that jointly captures both properties. It measures the overlap between a predicted bounding box and its corresponding ground-truth annotation. It is computed as the ratio between the area of overlap and the area of the union between the two regions.

[PLACEHOLDER — Equation]

[PLACEHOLDER — Visualization Showing Why Spatial Metrics Matter]

In this benchmark, IoU is used both for determining valid detections and for evaluating overall spatial alignment quality between predicted and annotated data snapshots.

## 4.3 Bounding Box Filtering

During benchmarking, we applied a post-processing filter removing predictions whose normalized bounding box area was smaller than `0.008`, which are typically too small to be relevant.

This filtering substantially improved precision across all evaluated models while having negligible impact on recall.

---

# 5. Benchmarked Models

We benchmark four open-source layout detection systems spanning OCR-aware architectures, YOLO-based document detectors, and transformer-based document encoders.

**TF-ID-Large**

[TF-ID-Large](https://huggingface.co/yifeihu/TF-ID-large) is a vision-language document layout detection model developed by Yifei Hu for extracting tables and figures from academic papers. The model is built by fine-tuning the large variant of Florence-2, trained on the custom TF-ID arXiv Papers dataset, which contains manually annotated academic-paper page images with bounding boxes for tables and figures.

**DocLayout-YOLO**

[DocLayout-YOLO](https://huggingface.co/papers/2410.12628) is a real-time document layout analysis model proposed by Zhiyuan Zhao et al. that extends the YOLOv10 object detection architecture for document understanding tasks such as detecting text blocks, tables, figures, titles, and other layout elements. To improve generalization, the authors introduced the large-scale synthetic pretraining corpus DocSynth-300K. The model is first pretrained on DocSynth-300K and then fine-tuned on downstream layout-analysis datasets including DocLayNet and D4LA. During inference, DocLayout-YOLO performs direct object detection on rendered document pages, predicting bounding boxes and semantic layout labels in a single-stage detection pipeline without requiring OCR or multimodal text encoding.

**YOLOv11 for Advanced Document Layout Analysis (medium)**

[YOLOv11 for Advanced Document Layout Analysis](https://huggingface.co/Armaggheddon/yolo11-document-layout) is a family of lightweight document layout analysis models built on the Ultralytics YOLO11 object detection architecture and fine-tuned specifically for document understanding tasks on the DocLayNet benchmark dataset. The models are designed to detect semantic document layout classes such as text blocks, titles, tables, figures, captions, and lists, using a purely vision-based single-stage detection pipeline. Architecturally, the models inherit the convolutional backbone, feature pyramid aggregation, and anchor-free detection head introduced in YOLO11, optimized for fast real-time inference while maintaining strong localization accuracy. During inference, document pages are rendered as images and passed directly through the YOLO detector, which predicts bounding boxes and class labels for layout regions in a single forward pass without requiring OCR or multimodal text embeddings.

We used the medium variant for the benchmarks in this paper.

**YOLOv26 for Advanced Document Layout Analysis (medium)**

[YOLOv26 for Advanced Document Layout Analysis](https://huggingface.co/Armaggheddon/yolo26-document-layout) is based on the newer Ultralytics YOLO26 architecture and the updated DocLayNet v1.2 dataset.

We used the medium variant for the benchmarks in this paper.

---

# 6. Results

## 6.1 Overall Detection Performance and Spatial Extraction Quality

### 6.1.1 Figure

|     Category     |     Metric     | TF-ID-Large | DocLayout-YOLO | YOLOv11 medium | YOLOv26 medium |
|:----------------:|:--------------:|:-----------:|:--------------:|:--------------:|:--------------:|
| Detection        | Precision      |       0.628 |          0.547 |          0.378 |          0.388 |
|                  | Recall         |       0.488 |          0.802 |          0.761 |          0.721 |
| Spatial Accuracy | IoU            |       0.877 |          0.820 |          0.817 |          0.817 |
|                  | Area precision |       0.935 |          0.996 |          0.990 |          0.990 |
|                  | Area recall    |       0.938 |          0.824 |          0.826 |          0.826 |

### 6.1.1 Table
|     Category     |     Metric     | TF-ID-Large | DocLayout-YOLO | YOLOv11 medium | YOLOv26 medium |
|:----------------:|:--------------:|:-----------:|:--------------:|:--------------:|:--------------:|
| Detection        | Precision      |       0.562 |          0.468 |          0.487 |          0.460 |
|                  | Recall         |       0.861 |          0.893 |          0.862 |          0.891 |
| Spatial Accuracy | IoU            |       0.919 |          0.834 |          0.817 |          0.824 |
|                  | Area precision |       0.972 |          0.993 |          0.994 |          0.994 |
|                  | Area recall    |       0.946 |          0.841 |          0.822 |          0.829 |

## 6.2 Split by Corpus

### 6.2.1 UNHCR / ReliefWeb

#### 6.2.1.1 Figure

|     Category     |     Metric     | TF-ID-Large | DocLayout-YOLO | YOLOv11 medium | YOLOv26 medium |
|:----------------:|:--------------:|:-----------:|:--------------:|:--------------:|:--------------:|
| Detection        | Precision      |       0.614 |          0.549 |          0.399 |          0.398 |
|                  | Recall         |       0.435 |          0.794 |          0.758 |          0.709 |
| Spatial Accuracy | IoU            |       0.867 |          0.822 |          0.819 |          0.818 |
|                  | Area precision |       0.922 |          0.995 |          0.988 |          0.989 |
|                  | Area recall    |       0.941 |          0.826 |          0.829 |          0.829 |

#### 6.2.1.2 Table

|     Category     |     Metric     | TF-ID-Large | DocLayout-YOLO | YOLOv11 medium | YOLOv26 medium |
|:----------------:|:--------------:|:-----------:|:--------------:|:--------------:|:--------------:|
| Detection        | Precision      |       0.460 |          0.510 |          0.462 |          0.408 |
|                  | Recall         |       0.826 |          0.947 |          0.936 |          0.936 |
| Spatial Accuracy | IoU            |       0.913 |          0.855 |          0.838 |          0.849 |
|                  | Area precision |       0.969 |          0.997 |          0.994 |          0.999 |
|                  | Area recall    |       0.942 |          0.858 |          0.844 |          0.850 |

### 6.2.2 PRWP

#### 6.2.2.1 Figure

|     Category     |     Metric     | TF-ID-Large | DocLayout-YOLO | YOLOv11 medium | YOLOv26 medium |
|:----------------:|:--------------:|:-----------:|:--------------:|:--------------:|:--------------:|
| Detection        | Precision      |       0.737 |          0.607 |          0.431 |          0.430 |
|                  | Recall         |       0.802 |          0.829 |          0.794 |          0.829 |
| Spatial Accuracy | IoU            |       0.904 |          0.794 |          0.804 |          0.806 |
|                  | Area precision |       0.976 |          0.998 |          0.998 |          0.996 |
|                  | Area recall    |       0.925 |          0.796 |          0.806 |          0.810 |

#### 6.2.2.2 Table

|     Category     |     Metric     | TF-ID-Large | DocLayout-YOLO | YOLOv11 medium | YOLOv26 medium |
|:----------------:|:--------------:|:-----------:|:--------------:|:--------------:|:--------------:|
| Detection        | Precision      |       0.797 |          0.689 |          0.698 |          0.694 |
|                  | Recall         |       0.941 |          0.914 |          0.902 |          0.924 |
| Spatial Accuracy | IoU            |       0.942 |          0.812 |          0.802 |          0.802 |
|                  | Area precision |       0.982 |          0.998 |          0.996 |          0.997 |
|                  | Area recall    |       0.960 |          0.813 |          0.805 |          0.804 |

### 6.2.3 Refugee PADs

#### 6.2.3.1 Figure

|     Category     |     Metric     | TF-ID-Large | DocLayout-YOLO | YOLOv11 medium | YOLOv26 medium |
|:----------------:|:--------------:|:-----------:|:--------------:|:--------------:|:--------------:|
| Detection        | Precision      |       0.432 |          0.331 |          0.098 |          0.128 |
|                  | Recall         |       0.661 |          0.919 |          0.694 |          0.548 |
| Spatial Accuracy | IoU            |       0.919 |          0.885 |          0.814 |          0.851 |
|                  | Area precision |       0.964 |          0.998 |          0.998 |          0.985 |
|                  | Area recall    |       0.954 |          0.887 |          0.815 |          0.865 |

#### 6.2.3.1 Table

|     Category     |     Metric     | TF-ID-Large | DocLayout-YOLO | YOLOv11 medium | YOLOv26 medium |
|:----------------:|:--------------:|:-----------:|:--------------:|:--------------:|:--------------:|
| Detection        | Precision      |       0.423 |          0.305 |          0.337 |          0.324 |
|                  | Recall         |       0.776 |          0.831 |          0.760 |          0.817 |
| Spatial Accuracy | IoU            |       0.886 |          0.852 |          0.825 |          0.839 |
|                  | Area precision |       0.959 |          0.981 |          0.992 |          0.985 |
|                  | Area recall    |       0.926 |          0.870 |          0.832 |          0.852 |

## 6.4 Qualitative Examples

[PLACEHOLDER — Successful Extraction Examples]

[PLACEHOLDER — Failure Case Examples]

---

# 7. Analysis and Findings

Our experiments reveal several recurring patterns across models and document domains.

Figure precision is frequently degraded by models detecting non-analytical visual content such as decorative images, situational photographs, and organizational graphics. This issue is particularly pronounced in humanitarian reports containing mixed visual content.

Figure recall is frequently limited by infographic-heavy layouts and compact operational dashboards. Models often fail to detect data cards, embedded analytical panels, and highly compressed monitoring summaries that are underrepresented in conventional layout datasets.

Table precision similarly suffers from extraction of non-analytical tabular structures such as tables of contents, formatting tables, recommendation blocks, and administrative layouts. Existing document layout models generally lack mechanisms for distinguishing semantically meaningful analytical tables from generic tabular structures.

We also observe that spatial extraction quality remains a major weakness even when detection succeeds. Compared to TF-ID-Large, YOLO-family models frequently under-crop relevant snapshot regions by excluding titles, subtitles, legends, or footnotes. These omissions significantly reduce downstream usability despite still qualifying as successful detections under IoU = 0.5.

At the same time, Area Precision remains consistently high across most models. This suggests that current systems generally avoid excessive over-cropping but frequently fail to capture the complete semantic extent of the snapshot. In practice, missing relevant contextual information is a more significant issue than including unnecessary surrounding pixels.

More broadly, our results demonstrate that operational institutional PDFs remain substantially harder than conventional academic layout datasets. All evaluated systems experience noticeable degradation on multilingual humanitarian reports, annex-heavy institutional documents, and infographic-dense operational dashboards.

These findings highlight a persistent gap between benchmark-oriented layout understanding and operationally useful data snapshot extraction.

---

# 8. Discussion

Our findings suggest that current open-source document layout systems remain insufficient for robust operational deployment on institutional PDF archives.

This limitation directly affects downstream workflows involving:
- retrieval systems,
- multimodal RAG pipelines,
- operational analytics,
- policy analysis,
- humanitarian monitoring,
- institutional knowledge extraction.

More fundamentally, our benchmark highlights a mismatch between existing layout detection objectives and operational extraction requirements. Current benchmarks largely optimize for generic layout segmentation, while real-world institutional workflows require semantically meaningful extraction of reusable analytical content.

We therefore argue that data snapshot extraction should be treated as a first-class document AI problem rather than a secondary extension of generic layout detection.

---

# 9. Limitations

This work has several limitations.

First, current annotations focus only on figures and tables. Second, the benchmark evaluates only object detection quality and spatial extraction quality rather than downstream semantic interpretation. Third, the dataset currently emphasizes development and humanitarian document ecosystems and may not fully capture all institutional publishing formats.

Finally, no train/validation/test split is currently provided. Our primary objective in this release is establishing a public benchmark corpus and evaluation framework for operational data snapshot extraction.

Future work may include expanded layout classes, segmentation masks, hierarchical annotations, and broader institutional document coverage.

---

# 10. Conclusion

We introduced **data-snapshot**, a benchmark dataset and evaluation framework for semantically meaningful figure and table extraction from development and humanitarian PDFs.

Our benchmark demonstrates that existing open-source layout detection systems struggle under operational institutional document complexity and that conventional layout benchmarks insufficiently capture real-world extraction requirements.

More importantly, our findings highlight a broader conceptual distinction between generic layout extraction and operationally useful data snapshot extraction. Not all figures and tables are analytically meaningful, and successful operational extraction requires models capable of identifying semantically relevant visual structures rather than indiscriminately extracting all layout objects.

We release the dataset, annotations, source PDFs, and benchmarking code to support future research in operational document intelligence and institutional document AI workflows.