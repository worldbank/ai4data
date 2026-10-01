# Anomaly Explanation Methodology Handbook

This handbook describes the methodology implemented by the World Bank's Development Data Group for **eliciting structured explanations** of anomalies in timeseries development data—such as World Development Indicators (WDI), Corporate Scorecard, or custom indicators—using large language models (LLMs). The approach produces classifications, evidence citations, confidence scores, and verifiability levels that support human review and data quality assurance.

The next section summarizes the methodology in non-technical terms using six steps. The summary is followed by chapters that describe each step in detail, the design choices, and the pipeline implementation.

All decisions, assumptions, and protocols described in this handbook align with World Bank Data Quality standards and internationally accepted practices for official statistics.

**Please cite this handbook as follows:**
World Bank. 2025. "Anomaly Explanation Methodology Handbook." AI for Data – Data for AI. Available at [https://worldbank.github.io/ai4data](https://worldbank.github.io/ai4data).

---

## Non-Technical Summary

The methodology is organized in six steps:

**Step 1. Acquiring anomaly signals**
Anomaly scores are obtained from statistical or machine-learning detectors applied to timeseries indicators. These signals identify periods (windows) where values deviate from expected patterns—sudden drops, spikes, gaps, or implausible magnitudes. Any detection method is supported provided it produces a score per (indicator, geography, period) tuple.

**Step 2. Adapting input data to canonical format**
Source data (wide-format CSVs, scorecard outputs, or custom schemas) are transformed into a canonical long-format with standardized columns: indicator, geography, period, value, imputation flags, and anomaly scores. This allows a single pipeline to process diverse data sources. For example, a wide-format scorecard CSV with year columns (2010, 2011, …, 2024) is melted and merged with anomaly scores to produce a unified long-format frame.

**Step 3. Extracting context for each anomaly window**
For each flagged window, the pipeline extracts the relevant timeseries snippet, geography metadata, indicator definition, and imputation status. This context is passed to the LLM so it can reason over the right time period and unit. For instance, for a GDP anomaly window of 2019–2020, the context includes all GDP values from 2010 to 2024, the country name, and the indicator definition—giving the LLM the necessary background to propose a COVID-19 explanation.

**Step 4. Eliciting structured explanations from one or more LLMs**
Each LLM (e.g., OpenAI, Gemini, Claude) is prompted with system rules and the extracted context. It returns *structured* outputs (not free-form text): a classification (e.g., data error, external driver, measurement update), confidence, evidence sources with verifiability levels, and a short explanation. Outputs are constrained by a JSON schema to ensure consistency and parsability.

**Step 5. Running the judge (when multiple explainers)**
When multiple LLMs produce explanations for the same anomaly, a **judge** (arbiter) LLM evaluates them and selects a single primary classification. The judge receives the timeseries context plus each explainer's output, then harmonizes disagreements (e.g., one model says `external_driver`, another says `insufficient_data`) into a final classification for the reviewer. This step is optional when using a single explainer.

**Step 6. Reviewing and exporting for quality assurance**
Human reviewers use a web application to view explanations alongside timeseries charts, approve or reject them, and optionally suggest corrections. Feedback is exported for audit, coverage analysis, or model improvement. LLM outputs are treated as hypotheses to be validated.

---

## When to Use This Methodology

This methodology is appropriate when:

- You have a collection of timeseries indicators with known anomaly scores or flagged periods.
- You want to systematically categorize the likely causes (data error, external event, methodological revision) at scale.
- You have domain experts available to validate LLM-generated explanations but need to prioritize their time.
- You need an auditable, traceable record of quality assurance decisions.

It is less appropriate when:

- Indicators are so sparsely populated that there is insufficient context for LLM reasoning.
- The data is highly sensitive and cannot be sent to third-party LLM APIs.
- You need guaranteed factual accuracy without any human review (the pipeline requires human oversight as a final step).

For large-scale runs across thousands of anomalies, use the batch mode described in the [Elicitation Pipeline](elicitation-pipeline.md) chapter, which submits asynchronous batch jobs to the LLM API.

---

## Limitations and Known Challenges

**LLM knowledge cutoff.** LLMs have a training data cutoff and may lack awareness of very recent events. For anomalies in the most recent 1–2 years, evidence quality may be lower. This should prompt reviewers to be more skeptical and to verify suggested causes independently.

**Hallucination risk.** Despite schema enforcement and conservative prompting rules, LLMs may occasionally invent plausible-sounding but incorrect event descriptions. The evidence verifiability levels (`well_documented`, `partially_documented`, `uncertain`) are designed to surface uncertainty, but reviewer validation remains essential.

**Non-English indicators.** The pipeline works best with English indicator names and definitions. Performance may degrade for indicators with non-English metadata or country-specific terminology not well represented in LLM training data.

**Geography-indicator interaction.** Some explanations require very specific country-level knowledge (e.g., a particular rebasing event in Nigeria). LLM accuracy varies by country and indicator; smaller or less-documented countries may receive weaker explanations.

---

## Contents

- [Motivation: Why LLM Elicitation](motivation.md) — The problem of anomaly explanation, why LLMs, and the design choice of elicitation versus generation
- [Elicitation Pipeline](elicitation-pipeline.md) — Pipeline steps, schema, judge/arbiter, code examples, and batch mode
- [Reviewer Feedback System](../../anomaly-detection/feedback-system.md) — Reviewer feedback system and export formats
- [Timeseries Anomaly Explanation with LLMs](../../notebooks/data-anomaly/timeseries-anomaly-explanation-with-llms.md) — Step-by-step implementation notebook
