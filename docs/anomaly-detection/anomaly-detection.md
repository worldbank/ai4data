# Anomaly Detection in Development Data

Timeseries development indicators—GDP growth rates, poverty headcounts, child mortality, school enrollment—carry the weight of policy decisions and resource allocation across hundreds of countries and decades. When these series exhibit anomalies, whether abrupt reversals, implausible gaps, or sudden discontinuities, the consequences of misinterpretation are real: misinformed budgets, misaligned programs, and eroded trust in data systems.

This section covers the **explanation** of anomalies in timeseries indicator data. Detection is supplied by the user's own detector; the package takes its scores and applies **LLM elicitation** to produce structured explanations, with a classification, evidence citations, and a confidence score, that support human review and data quality assurance.

---

## Background: The Scale of the Problem

The World Bank's World Development Indicators (WDI) database contains over 1,400 indicators spanning 217 economies, some with annual records going back to the 1960s. The Corporate Scorecard adds its own series. Across this breadth of data, statistical anomaly detectors flag thousands of data points per review cycle as potential outliers.

The deeper challenge is **explanation**. Identifying that a value is statistically unusual is the beginning of a quality assurance workflow. The critical question is: *why* did this happen? Is it a data entry error, a major external event (conflict, pandemic, economic shock), a legitimate methodological revision (rebasing, new census), or something else entirely?

Answering that question historically required a domain expert with institutional knowledge about the specific country, indicator, and time period. At the scale of thousands of anomalies across hundreds of indicators and countries, that approach does not scale. This program automates the first pass—systematically generating structured hypotheses about causes that reviewers can validate or reject.

---

## Detection vs. Explanation

This pipeline has two distinct phases:

**Detection** flags data points that deviate from expected patterns. The package does not include a detector (`ai4data.anomaly.detection` is a placeholder): the scores come from the organization's own method, for example a z-score threshold, an isolation forest, or the residuals of a seasonal decomposition, and the adapter reads them in. Detection produces a score for each data point (`anomaly_score`) and, where several detectors run, the number that agree (`outlier_count`). It says: *this value is unusual relative to its series*.

**Explanation** uses LLMs to classify the cause of flagged anomalies and provide evidence. It says: *this value is probably unusual because of X, based on evidence Y*. The explanation is structured—classification, confidence, evidence source, verifiability—so it can be reviewed, audited, and exported.

These phases are intentionally decoupled. The pipeline accepts anomaly scores from any detection system, provided the data is in canonical format. This allows teams to use their own detection algorithms while benefiting from the LLM explanation layer.

---

## End-to-End Workflow

```
Raw Indicator Data (wide CSV, scorecard, or custom format)
        │
        ▼
[1] Detection (the organization's own detector)
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
    build_batch_file,
    run_batch,
    parse_batch_output,
    export_for_review,
)

# Step 1: Load and canonicalize data (wide indicator file plus the detector's scores)
adapter = ScorecardWideAdapter()
canonical_df = adapter.load("scorecard_wide.csv", "anomaly_scores.csv")

# Step 2: Shortlist the series to explain and build the name maps
filtered = canonical_df[(canonical_df["outlier_count"] >= 3) & (~canonical_df["is_imputed"])]
shortlist = (
    filtered.groupby(["indicator_id", "geography_id"])["anomaly_score"].sum()
    .sort_values(ascending=False).reset_index()
)
geo_map = canonical_df.set_index("geography_id")["geography_name"].to_dict()
ind_map = canonical_df.set_index("indicator_id")["indicator_name"].to_dict()
source_df = canonical_df.set_index(["indicator_id", "geography_id"]).sort_index()

# Step 3: Build the batch file (OpenAI, Gemini, or Anthropic) and run it
path, count = build_batch_file("batch_input.jsonl", shortlist, source_df, geo_map, ind_map, provider="openai")
run_batch("openai", path, "batch_output.jsonl", poll_interval=60)

# Step 4: Parse responses into a DataFrame of explanations
anomalies_df = parse_batch_output("batch_output.jsonl", "openai", ind_map, geo_map)

# Step 5: Export for human review, with the series around each anomaly
export_for_review(anomalies_df, timeseries_df=canonical_df, output_path="review.json")
```

`build_batch_file` extracts the context of each series with `extract_anomaly_contexts` and writes one request per series, plus a sidecar file that maps the request ids back to the series. `run_batch` submits the file, polls until the batch completes, and downloads the output; `submit_batch`, `wait_for_batch`, and `download_batch_output` are the same steps as separate calls.

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

---

## Limitations

- An explanation is a hypothesis with cited evidence. The reviewer decides whether the value is an error, an event, or a revision, and the feedback export is the record of the decision.
- Evidence quality depends on what the model can retrieve for a country, an indicator, and a period. Sparse contexts produce low-confidence explanations, and the confidence score is the model's own.
- Detection is outside the package: the pipeline takes anomaly scores from any detector in the canonical format. The choice of detector and threshold decides what is flagged.
- Batch runs through the providers' asynchronous APIs take hours for large sets, and the pipeline needs an API key for a hosted provider or a local model served through a compatible endpoint.
- Status on the [tools catalogue](/tools#anomaly): pre-release.
