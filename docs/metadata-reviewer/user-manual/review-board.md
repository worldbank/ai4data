# AI Suggestion Review Board

:::{note} Companion tool, not part of the package
The AI Suggestion Review Board is a standalone, browser-based
prototype (`ai_suggestion_review_board.html`). It is shipped separately
from the Python package and is used **after** a review run, to let a
human triage what the pipeline detected.
:::

## Overview

The board is a single self-contained HTML file. Open it in any modern
web browser — no server or installation is needed. It loads an Excel
workbook of review results and presents them in three linked views:

1. **Metadata Projects** — one row per reviewed record, with counts
   of issues broken down by severity and by category.
2. **Detected Issues** — for the selected project, every issue listed
   and sorted from most to least severe.
3. **Current vs. Suggested** — for the selected issue, a
   side-by-side, word-level diff of the current value against the
   suggested correction.

## Preparing the Excel input

The board reads the first worksheet of an `.xlsx` (or `.xls`) file and looks
for two columns:

| Column | Contents |
|---|---|
| `ME_project` | A label identifying the metadata record/project. The alternate name `metadata_project` is also accepted. |
| `detected_issues` | The pipeline's output array for that project, stored as a JSON string. The alternate name `issue_list` is also accepted. |

Each row is one reviewed project. You build this workbook from your job
results — typically while iterating over a catalogue of records:

```python
import json
import pandas as pd
from ai4data.metadata.reviewer import MetadataReviewerClient

client = MetadataReviewerClient.from_openai(model="gpt-4o", api_key="sk-...")

rows = []
for project_name, metadata in catalogue.items():  # your dict of records
    job = client.submit(metadata)
    issues = job.wait_sync(timeout=300)
    rows.append({
        "ME_project": project_name,
        "detected_issues": json.dumps(issues, ensure_ascii=False),
    })

pd.DataFrame(rows).to_excel("review_board_input.xlsx", index=False)
```

:::{note} Schema match
Each object inside the JSON array should carry `detected_issue`,
`issue_category`, `issue_severity`, `current_metadata`, and
`suggested_metadata` — exactly the schema the pipeline produces. The
board reads these fields directly; missing categories or severities
simply render as "N/A".
:::

## Using the interface

1. **Load Excel.** Click the **Load Excel** button and choose your
   workbook. The board parses the first sheet and populates the
   Metadata Projects table.
2. **Pick a project.** Click any row in the projects table. Its
   severity and category counts are shown inline (for example, "2
   Critical, 5 High" and "3 Typo / Language, 4 Inconsistency /
   Conflict").
3. **Scan the issues.** The Detected Issues table lists that project's
   issues sorted by severity, highest first, each with a colored
   severity pill and category pill, plus the key path of the affected
   field.
4. **Inspect a correction.** Click an issue row to open the diff panels
   below it.

## Reading the diff view

When an issue is selected, two panels appear side by side:

- **Current Metadata** (left) — the value as it is now. Text that
  differs from the suggestion is highlighted in **red**.
- **Suggested Metadata** (right) — the proposed value. Text that is
  new or changed is highlighted in **green**.

The highlighting is a word-level diff, so for a small change like a
single typo only the changed token is colored, making it easy to confirm
the suggestion at a glance before accepting or rejecting it. A legend
beneath the panels restates the color meaning.

:::{note} Prototype status
The board is labelled a prototype (version 0.3). It is a review aid:
it displays and diffs the pipeline's suggestions but does not itself
write changes back to your metadata source. Treat acceptance and
application of fixes as a separate, deliberate step in your own
workflow.
:::
