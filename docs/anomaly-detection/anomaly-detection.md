# Anomaly Detection in Development Data

Timeseries development indicators—GDP growth rates, poverty headcounts, child mortality, school enrollment—carry the weight of policy decisions and resource allocation across hundreds of countries and decades. When these series exhibit anomalies, whether abrupt reversals, implausible gaps, or sudden discontinuities, the consequences of misinterpretation are real: misinformed budgets, misaligned programs, and eroded trust in data systems.

This section covers the detection and **explanation** of anomalies in timeseries indicator data. We combine statistical and machine-learning methods for detection with **LLM elicitation** to produce structured explanations—classifications, evidence citations, and confidence scores—that support human review and data quality assurance.

---

## Background: The Scale of the Problem

The World Bank's World Development Indicators (WDI) database contains over 1,400 indicators spanning 217 economies, some with annual records going back to the 1960s. The Corporate Scorecard and Scorecard-linked databases add hundreds more series at higher resolutions. Across this breadth of data, statistical anomaly detectors routinely flag thousands of data points per review cycle as potential outliers.

The deeper challenge is **explanation**. Identifying that a value is statistically unusual is the beginning of a quality assurance workflow. The critical question is: *why* did this happen? Is it a data entry error, a major external event (conflict, pandemic, economic shock), a legitimate methodological revision (rebasing, new census), or something else entirely?

Answering that question historically required a domain expert with institutional knowledge about the specific country, indicator, and time period. At the scale of thousands of anomalies across hundreds of indicators and countries, that approach does not scale. This program automates the first pass—systematically generating structured hypotheses about causes that reviewers can validate or reject.

---

## Detection vs. Explanation

This pipeline has two distinct phases:

**Detection** uses statistical and machine-learning methods to flag data points that deviate from expected patterns. Common methods include:
- Z-score thresholding (|z| > threshold)
- Isolation Forest
- Seasonal decomposition residuals

Detection produces a score for each data point (e.g., `anomaly_score`, `outlier_count`). It says: *this value is unusual relative to its series*.

**Explanation** uses LLMs to classify the cause of flagged anomalies and provide evidence. It says: *this value is probably unusual because of X, based on evidence Y*. The explanation is structured—classification, confidence, evidence source, verifiability—so it can be reviewed, audited, and exported.

These phases are intentionally decoupled. The pipeline accepts anomaly scores from any detection system, provided the data is in canonical format. This allows teams to use their own detection algorithms while benefiting from the LLM explanation layer.

---

## End-to-End Workflow

```
Raw Indicator Data (wide CSV, scorecard, or custom format)
        │
        ▼
[1] Detection (z-score, IsolationForest, etc.)
        │  anomaly_score per (indicator, geography, year)
        ▼
[2] Adapter (ScorecardWideAdapter or ConfigurableAdapter)
        │  canonical long-format DataFrame
        ▼
[3] Context Extraction (extract_anomaly_contexts)
        │  per-anomaly context: timeseries snippet + indicator def + geography
        ▼
[4] LLM Elicitation (batch_builder → batch_runner → output_parser)
        │  structured JSON: classification, confidence, evidence, explanation
        ▼
[5] Judge / Arbiter (optional, for multi-model runs)
        │  harmonized single classification per anomaly
        ▼
[6] Review Export (export_for_review)
        │  JSON review payload for the web app
        ▼
[7] Human Review (FastAPI web app)
        │  approve / reject / suggest correction
        ▼
[8] Feedback Export (CSV)
        │  audit trail, retraining data, coverage metrics
```

---

## Quick Start

```python
from ai4data.anomaly.explanation import (
    ScorecardWideAdapter,
    extract_anomaly_contexts,
    run_batch,
    parse_batch_output,
    export_for_review,
)

# Step 1: Load and canonicalize data
adapter = ScorecardWideAdapter()
df = adapter.load("scorecard_wide.csv", "anomaly_scores.csv")

# Step 2: Extract context for each anomaly window
contexts = extract_anomaly_contexts(df)

# Step 3: Build and run batch (OpenAI or Gemini)
batch_id = run_batch(contexts, model="gpt-4o-mini")

# Step 4: Parse responses
explanations = parse_batch_output(batch_id)

# Step 5: Export for human review
export_for_review(explanations, output_path="review.json")
```

Then start the review application:

```bash
uv run python -m apps.anomaly_review review.json feedback.json
# Navigate to http://localhost:8000
```

See the [methodology handbook](../anomaly/explanation/index.md) and the [step-by-step notebook](../notebooks/data-anomaly/timeseries-anomaly-explanation-with-llms.md) for the full pipeline with all options.

---

## Contents

- [Anomaly Explanation Methodology Handbook](../anomaly/explanation/index.md) — Full methodology: non-technical summary, motivation, pipeline, schema, and review process
- [Motivation: Why LLM Elicitation](../anomaly/explanation/motivation.md) — The problem, why LLMs, and the elicitation design choice
- [Elicitation Pipeline](../anomaly/explanation/elicitation-pipeline.md) — Detailed pipeline, schema reference, code examples
- [Reviewer Feedback System](feedback-system.md) — Collecting and using reviewer feedback
- [Timeseries Anomaly Explanation with LLMs](../notebooks/data-anomaly/timeseries-anomaly-explanation-with-llms.md) — Step-by-step implementation notebook
