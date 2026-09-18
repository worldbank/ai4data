# Elicitation Pipeline: How It Works

This chapter describes the technical workflow for LLM-elicited anomaly explanation: from canonical data through context extraction, LLM invocation, and human review. It also defines the explanation schema, shows code examples for each stage, and documents the design choices that govern elicitation behavior.

---

## Pipeline Overview

The pipeline has six stages:

1. **Canonical data** — Indicators in long format with `indicator_id`, `geography_id`, `period`, `value`, `is_imputed`, `anomaly_score`, and related metadata. Source data (e.g., wide-format Scorecard CSVs) are adapted via configurable adapters.
2. **Context extraction** — For each anomaly window, the pipeline extracts the timeseries snippet, geography metadata, indicator name, and imputation flags. This context is formatted for the LLM.
3. **LLM elicitation** — A system prompt defines the assistant role and rules; a user prompt injects the extracted context. One or more LLMs (e.g., OpenAI, Gemini) produce structured outputs. The response is constrained by a JSON schema so that outputs are parseable and type-safe.
4. **Judge (arbiter)** — When multiple explainers are used, a **judge** LLM evaluates their outputs and harmonizes them into a single primary classification per anomaly. This step is optional when using a single explainer.
5. **Parsed output** — Responses are validated against the schema. Each anomaly record includes `classification`, `confidence`, `explanation`, `evidence_strength`, and `evidence_source`.
6. **Review** — Human reviewers use a web app to view explanations alongside timeseries charts, approve or reject them, and optionally suggest corrections. Feedback is exportable for audit or model improvement.

The implementation is available in the `ai4data.anomaly.explanation` package. A step-by-step notebook is provided: [Timeseries Anomaly Explanation with LLMs](../../notebooks/data-anomaly/timeseries-anomaly-explanation-with-llms.md).

---

## Stage 1: Adapting to Canonical Format

Source data in various formats is converted to a canonical long-format DataFrame using adapters from [`src/ai4data/anomaly/explanation/adapters.py`](../../../src/ai4data/anomaly/explanation/adapters.py).

```python
from ai4data.anomaly.explanation import ScorecardWideAdapter, adapter_from_config

# For World Bank Scorecard data (wide CSV + anomaly scores CSV)
adapter = ScorecardWideAdapter()
df = adapter.load("scorecard_wide.csv", "anomaly_scores.csv")

# For custom formats, specify the column mapping
adapter = adapter_from_config({
    "indicator_id": "IndicatorCode",
    "indicator_name": "IndicatorName",
    "geography_id": "CountryCode",
    "geography_name": "CountryName",
    "period": "Year",
    "value": "Value",
    "is_imputed": "IsImputed",
    "anomaly_score": "ZScore",
    "outlier_count": "OutlierCount",
})
df = adapter.load_csv("custom_data.csv")
```

The resulting `df` has canonical column names regardless of the source format.

---

## Stage 2: Context Extraction

For each anomaly window (a flagged `(indicator_id, geography_id, window)` triple), the pipeline extracts the relevant timeseries slice and associated metadata, as implemented in [`src/ai4data/anomaly/explanation/context.py`](../../../src/ai4data/anomaly/explanation/context.py).

```python
from ai4data.anomaly.explanation import extract_anomaly_contexts

contexts = extract_anomaly_contexts(df)
# contexts: list of dicts, each containing:
# - indicator_id, indicator_name, geography_id, geography_name
# - window: [start_year, end_year]
# - timeseries: list of {period, value, is_imputed, anomaly_score}
```

A typical context object passed to the LLM looks like:

```json
{
  "indicator_id": "NY.GDP.MKTP.KD.ZG",
  "indicator_name": "GDP growth (annual %)",
  "geography_id": "NGA",
  "geography_name": "Nigeria",
  "window": [2015, 2017],
  "timeseries": [
    {"period": 2013, "value": 6.67, "is_imputed": false, "anomaly_score": 0.4},
    {"period": 2014, "value": 6.31, "is_imputed": false, "anomaly_score": 0.5},
    {"period": 2015, "value": 2.65, "is_imputed": false, "anomaly_score": 1.2},
    {"period": 2016, "value": -1.62, "is_imputed": false, "anomaly_score": 2.8},
    {"period": 2017, "value": 0.80, "is_imputed": false, "anomaly_score": 1.1},
    {"period": 2018, "value": 1.93, "is_imputed": false, "anomaly_score": 0.3}
  ]
}
```

---

## Stage 3: LLM Elicitation

Prompts are built and sent to one or more LLMs. Implementations are in [`src/ai4data/anomaly/explanation/prompts.py`](../../../src/ai4data/anomaly/explanation/prompts.py) and [`src/ai4data/anomaly/explanation/llm_client.py`](../../../src/ai4data/anomaly/explanation/llm_client.py).

For large-scale runs, use the **batch mode** (see [Running at Scale](#running-at-scale) below). For small exploratory runs:

```python
from ai4data.anomaly.explanation import run_batch, parse_batch_output

batch_id = run_batch(
    contexts,
    model="gpt-4o-mini",       # or "gemini-2.0-flash", etc.
    provider="openai",
)
explanations = parse_batch_output(batch_id, provider="openai")
```

### Example LLM Input/Output

**Input (user prompt excerpt):**
```
Indicator: GDP growth (annual %)
Country: Nigeria
Anomaly window: [2015, 2016]
Timeseries: 2013: 6.67%, 2014: 6.31%, 2015: 2.65%, 2016: -1.62%, 2017: 0.80%
```

**Output (structured JSON from LLM):**
```json
{
  "anomalies": [
    {
      "window": [2015, 2016],
      "is_anomaly": true,
      "classification": "external_driver",
      "confidence": 0.92,
      "explanation": "Nigeria experienced a sharp GDP contraction in 2016, its first recession in 25 years, driven by the collapse in global oil prices beginning in 2014–2015 and militant attacks on oil infrastructure in the Niger Delta.",
      "evidence_strength": "strong_direct",
      "evidence_source": [
        {
          "name": "Nigeria GDP recession 2016, oil price collapse",
          "date_range": "2015-2016",
          "source_type": "economic_crisis",
          "verifiability": "well_documented"
        }
      ],
      "source": "llm_inferred"
    }
  ]
}
```

---

## Stage 4: The Judge (Arbiter)

When multiple explainers (e.g., OpenAI and Gemini) produce explanations for the same anomaly, they may disagree. The **judge** (arbiter) LLM evaluates all outputs and selects a single primary classification for the reviewer.

The judge receives:
- The timeseries context (values, geography, indicator, time window)
- Each explainer's classification, explanation, evidence sources, and confidence

It then returns a harmonized classification. The judge uses the same schema and conservative bias as the explainers.

**Example disagreement and resolution:**

| Explainer | Classification | Confidence |
|---|---|---|
| OpenAI GPT-4o | `external_driver` | 0.88 |
| Gemini 2.0 Flash | `measurement_system_update` | 0.72 |
| **Judge verdict** | `external_driver` | — |

In this case, the judge favored the better-evidenced explanation, noting that the GDP contraction coincides with documented oil price shocks, whereas a methodological revision would typically affect only the level, not the growth rate trajectory.

This step is invoked via `export_for_review_with_explainers(..., run_arbiter=True)`. The implementation is in [`src/ai4data/anomaly/explanation/arbiter.py`](../../../src/ai4data/anomaly/explanation/arbiter.py).

---

## Explanation Schema

Each elicited explanation follows a strict schema defined in [`src/ai4data/anomaly/explanation/schemas.py`](../../../src/ai4data/anomaly/explanation/schemas.py).

### Primary Classifications

| Classification | Description |
|---|---|
| `data_error` | Placeholder, rounding artifact, rebasing artifact, template issue, ingestion error, logical impossibility |
| `external_driver` | Macroeconomic or geopolitical event, conflict, policy reform, disaster, pandemic, global cycle |
| `measurement_system_update` | Rebasing, SNA/PPP revision, new census benchmark, classification change |
| `modeling_artifact` | Anomaly detector or transformation artifact |
| `insufficient_data` | No verifiable cause identified |

### Evidence Strength

| Level | Meaning |
|---|---|
| `strong_direct` | Clearly linked, well-documented event or revision. Only this level counts as valid confirmation for `is_anomaly=true`. |
| `moderate_contextual` | Plausible contextual relationship but without direct documentation |
| `weak_speculative` | Weak or uncertain linkage, minimal documentation |
| `no_evidence` | Used when no `evidence_source` applies (e.g., data error, insufficient data) |

### Verifiability

| Level | Meaning |
|---|---|
| `well_documented` | Event or source is widely recognized and documented |
| `partially_documented` | Some documentation exists |
| `uncertain` | Little or unclear documentation |
| `not_applicable` | Used when evidence_source is empty (e.g., for data errors) |

### Example Classification

> A 15% drop in GDP for Country X in 2020
> → **Classification:** `external_driver`
> → **Evidence:** COVID-19 pandemic (2020-03 to 2021), `strong_direct`, `well_documented`

---

## Running at Scale

For production runs across thousands of anomaly contexts, use the asynchronous batch APIs rather than sequential per-request calls. The batch pipeline uses [`batch_builder.py`](../../../src/ai4data/anomaly/explanation/batch_builder.py) and [`batch_runner.py`](../../../src/ai4data/anomaly/explanation/batch_runner.py):

```python
from ai4data.anomaly.explanation import (
    build_batch_file,
    submit_batch,
    wait_for_batch,
    download_batch_output,
)

# Build the JSONL batch file
batch_file_path = build_batch_file(contexts, output_path="batch_input.jsonl")

# Submit to OpenAI Batch API
batch_id = submit_batch(batch_file_path, provider="openai")

# Wait for completion (polls every 60 seconds by default)
wait_for_batch(batch_id, provider="openai")

# Download and parse results
raw_output = download_batch_output(batch_id, provider="openai")
explanations = parse_batch_output(raw_output)
```

Batch APIs offer significantly lower cost (typically 50% cheaper than synchronous API calls) and higher throughput for large-scale runs.

---

## Elicitation Design Choices

The following design choices govern how the LLM is prompted and how outputs are constrained. Each choice addresses a specific failure mode observed in early iterations.

**Constrained output.** Prompts instruct the model to output *only* valid JSON matching the schema. No markdown, comments, or extra text. Without this constraint, models frequently wrap JSON in code blocks (````json\n...\n````), add explanatory preambles, or include trailing commas—all of which cause parsing failures.

**Conservative bias.** Rules such as "Prefer `insufficient_data` over creative inference when unsure" and "When in doubt, lower confidence ≤ 0.5" discourage speculation. Without this constraint, models tend toward confident, narrative explanations that sound plausible but are not verifiable. The conservative bias produces outputs that are honest about uncertainty rather than superficially confident.

**Evidence sourcing.** The model must cite specific, verifiable events (e.g., "COVID-19 pandemic", "2014 GDP rebasing, Nigeria"). Generic placeholders (e.g., "international organization") are disallowed. Without this constraint, models produce vague citations that reviewers cannot validate independently.

**Schema enforcement.** Using OpenAI or Gemini structured-output APIs with JSON Schema ensures responses conform to the expected shape. Invalid or hallucinated fields are rejected at the API boundary—before the Python parser ever sees them. This reduces the need for complex post-processing cleanup.

---

## Human-in-the-Loop

The [Reviewer Feedback System](../../anomaly-detection/feedback-system.md) supports:

- Approving, rejecting, or flagging explanations for further review
- Suggesting an alternative classification when the reviewer disagrees
- Optional free-text comments
- Export to CSV for audit, coverage analysis, or model improvement

This loop ensures that LLM-elicited explanations are treated as hypotheses to be validated, not as final authority. The methodology prioritizes traceability and human oversight in line with official statistics principles.

---

## References

- [OpenAI: Structured model outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Gemini: Structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [World Bank: Data Quality and Effectiveness](https://datahelpdesk.worldbank.org/knowledgebase/articles/906534-data-quality-and-effectiveness)
