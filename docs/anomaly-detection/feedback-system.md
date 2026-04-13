# Reviewer Feedback System

The Anomaly Explanation Reviewer application supports collecting structured feedback from domain experts and data stewards. This document describes the feedback system, how to run the application, and how to use collected feedback downstream.

---

## Why Reviewer Feedback Matters

LLM-elicited anomaly explanations are hypotheses, not facts. Even with schema enforcement and conservative prompting, models can propose incorrect classifications, cite the wrong events, or miss domain-specific knowledge that only an expert would know. The feedback system closes the human-in-the-loop by creating a structured mechanism for reviewers to:

1. **Validate** LLM outputs that are correct (approved verdicts build confidence in the pipeline)
2. **Correct** wrong classifications or evidence (rejected + suggested_classification improves coverage)
3. **Surface uncertainty** by flagging explanations that need further investigation before a decision is made
4. **Build a labeled dataset** of reviewed anomalies that can be used for model improvement, inter-rater agreement analysis, or audit
5. **Separate dimensions of quality** (classification vs. narrative vs. evidence vs. anomaly validity) and **per-model judgments** when multiple explainers are compared

Without a feedback mechanism, AI-assisted quality assurance is a one-way system—outputs flow in only one direction and there is no way to measure accuracy, track improvements, or detect systematic errors.

---

## Overall verdict vs. facet ratings

**Overall verdict** (`approved` | `rejected` | `needs_review`) is a **QA gate** on the anomaly *item*: whether the reviewed case is acceptable for downstream use, needs more work, or should not be relied on as-is. It is **not** synonymous with “every facet is perfect.”

**Facet ratings** capture *why* a case might be mixed—for example, the **classification** label can be correct while the **explanation** text is misleading. In that situation, a reviewer might:

- Set classification → `correct` and explanation → `incorrect` (or `partially_correct`) for the relevant explainer column, and
- Choose an overall verdict of `needs_review` or `rejected` depending on policy (e.g. reject if the narrative cannot be published even when the label is right).

**Reference explainer** identifies which model output is primary for audit, training, or policy (e.g. the explainer whose column you treat as the main hypothesis).

**Best explainer** (optional, shown when multiple explainers exist) records which model was closest overall when outputs disagree.

---

## Feedback Schema

Each feedback entry records:

| Field | Type | Description |
|---|---|---|
| `item_id` | int | Index into the review items list (navigation index at submit time) |
| `indicator_code` | str | Indicator code (e.g., `NY.GDP.MKTP.KD.ZG`) |
| `geography_code` | str | ISO 3166-1 alpha-3 code (e.g., `NGA`) |
| `window_str` | str | Anomaly window as `"start-end"` (e.g., `"2015-2016"`) |
| `verdict` | str | `approved`, `rejected`, or `needs_review` (QA gate; see above) |
| `comment` | str (optional) | Free-text reviewer comment |
| `suggested_classification` | str (optional) | Alternative classification if the classification facet is wrong |
| `facets` | object (optional) | Map: **facet name** → **explainer name** → **rating** (see below) |
| `reference_explainer` | str (optional) | Which explainer is primary for audit / training |
| `best_explainer` | str (optional) | Which explainer was best when models disagreed |
| `overall_basis` | str (optional) | `explicit` (reviewer-chosen overall) or `derived` (reserved for future use) |
| `timestamp` | ISO8601 | When feedback was submitted |

### Facet keys and ratings

**Facet keys** (each cell is optional):

| Facet | Meaning |
|--------|--------|
| `anomaly_validity` | Whether the flagged window is plausibly anomalous for this series |
| `classification` | Whether the assigned classification label is appropriate |
| `explanation` | Whether the written rationale matches the series and is not misleading |
| `evidence` | Whether cited evidence (sources, dates, claims) is appropriate |

**Ratings** (per facet, per explainer name such as `OpenAI` or `Gemini`):

- `correct`
- `partially_correct`
- `incorrect`
- `not_applicable`
- `unsure`

Omitting a cell means “not rated” for that facet and explainer.

The combination of (`indicator_code`, `geography_code`, `window_str`) forms the **stable key** for a feedback entry. Resubmitting feedback for the same stable key updates the existing entry (upsert), so reviewers can revise their verdicts without creating duplicates.

**Backward compatibility:** Older JSON files without `facets`, `reference_explainer`, or `best_explainer` still load; missing fields are treated as empty.

---

## Running the Review Application

The review application is a FastAPI server with a single-page UI. It loads a review JSON payload and stores feedback to disk.

```bash
# Start with a review file and a feedback persistence file
uv run python -m apps.anomaly_review path/to/review.json feedback.json

# Navigate to http://localhost:8000
```

The `review.json` file is produced by `export_for_review()` or `export_for_review_with_explainers()` from the explanation pipeline:

```python
from ai4data.anomaly.explanation import export_for_review

export_for_review(explanations, output_path="review.json")
```

The review UI displays:

- A navigation list of all anomaly items (indicator + country + window)
- A timeseries chart for each item, highlighting the anomaly window
- The LLM-generated classification, confidence, explanation, and evidence sources
- When multiple explainers are used: tabs per explainer; switching tabs highlights the matching column in the facet matrix
- Feedback controls: facet matrix (per explainer), reference explainer, optional “best overall” when multiple models exist, overall verdict buttons, suggested classification, and free-text comment

---

## API Endpoints

The review app exposes a REST API for programmatic integration:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/review` | Full review payload (all items) |
| `GET` | `/api/items` | Navigation list (id, indicator, geography, window) |
| `GET` | `/api/items/{item_id}` | Single item with full explanation details |
| `POST` | `/api/feedback` | Submit feedback (JSON body matching schema) |
| `GET` | `/api/feedback` | List all feedback (optional `?item_id=N` filter) |
| `GET` | `/api/feedback/item` | Get feedback by stable key (`indicator_code`, `geography_code`, `window_str`) |
| `GET` | `/api/feedback/schema` | Feedback schema for integration |
| `GET` | `/api/feedback/export` | Export all feedback as CSV |

---

## Exporting Feedback

Feedback can be exported at any time during or after the review session:

```bash
# Via HTTP
curl http://localhost:8000/api/feedback/export > feedback_export.csv

# Or via the Python API (when running programmatically)
from apps.anomaly_review.feedback import export_feedback_csv, init_feedback_store

init_feedback_store("feedback.json")
export_feedback_csv("feedback_export.csv")
```

The CSV export includes flat columns for all top-level fields, a **`facets_json`** column with the nested `facets` object serialized as JSON, and a **`stable_key`** column (`indicator|geography|window`) for joining with the review data.

---

## Analyzing Collected Feedback

The CSV export supports several downstream analyses:

### Coverage analysis

```python
import pandas as pd

fb = pd.read_csv("feedback_export.csv")
total = len(fb)
approved = (fb["verdict"] == "approved").sum()
rejected = (fb["verdict"] == "rejected").sum()
needs_review = (fb["verdict"] == "needs_review").sum()

print(f"Total reviewed: {total}")
print(f"Approval rate:  {approved/total:.1%}")
print(f"Rejection rate: {rejected/total:.1%}")
```

### Facet-level analysis (example)

```python
import json

def facet_series(df, facet_key, explainer):
    out = []
    for raw in df["facets_json"].fillna("{}"):
        obj = json.loads(raw)
        inner = obj.get(facet_key) or {}
        out.append(inner.get(explainer))
    return out

fb["classification_openai"] = facet_series(fb, "classification", "OpenAI")
```

### Classification corrections

```python
# Compare LLM classification to reviewer's suggested classification
corrections = fb[fb["suggested_classification"].notna() & (fb["suggested_classification"] != "")]
print(corrections[["indicator_code", "geography_code", "window_str",
                    "verdict", "suggested_classification"]].head())
```

### Inter-rater agreement (when multiple reviewers)

```python
# If multiple reviewers submit feedback for the same item,
# compute Cohen's kappa over verdict values
from sklearn.metrics import cohen_kappa_score

# Join on stable_key where reviewer_id differs
# ... (join logic depends on your data collection setup)
```

### Feeding back into the pipeline

Reviewed and corrected labels can be used to:

1. **Evaluate** pipeline quality by computing precision/recall of LLM classifications against expert verdicts and facet ratings
2. **Retrain or fine-tune** a downstream classifier using `(context, classification)` pairs
3. **Refine prompts** by identifying systematic misclassifications (e.g., the model consistently confuses `measurement_system_update` with `external_driver` for particular indicator types)
4. **Update classification labels** in the source system where the LLM explanation was correct and the original label was missing or wrong

---

## Implementation Reference

The feedback system implementation is in [`apps/anomaly_review/feedback.py`](../../../apps/anomaly_review/feedback.py) and [`apps/anomaly_review/main.py`](../../../apps/anomaly_review/main.py).
