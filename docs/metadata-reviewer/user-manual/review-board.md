# Reviewing the output

The package returns a ranked JSON array of issues and writes nothing to a
catalog. The review of those issues happens in one of two places.

## Inside the Metadata Editor

The reviewer ships as an optional add-on to the Metadata Editor's FastAPI
service. When it is enabled, **Assess metadata** on a project's **Issues**
page runs the pipeline and lists the AI-generated issues with their
suggested corrections, and each one can be commented on, accepted,
rejected, dismissed, or applied through the Editor's issues workflow. The
[Editor's installation guide](https://worldbank.github.io/metadata-editor-docs/tech_installation_metadata_reviewer.html)
describes the setup, and the [assess metadata](https://worldbank.github.io/metadata-editor-docs/assess_metadata.html)
page describes the review.

## Outside the Metadata Editor

When the package runs on its own, the results of a catalogue review can be
assembled into one table for triage in a spreadsheet or in the
organization's own review tool. The table has one row per record, with
the issue array stored as a JSON string:

```python
import json
import pandas as pd
from ai4data.metadata.reviewer import MetadataReviewerClient

client = MetadataReviewerClient.from_openai(model="gpt-4o", api_key="sk-...")

rows = []
for project_name, metadata in catalogue.items():  # a dict of records
    job = client.submit(metadata)
    issues = job.wait_sync(timeout=300)
    rows.append({
        "ME_project": project_name,
        "detected_issues": json.dumps(issues, ensure_ascii=False),
    })

pd.DataFrame(rows).to_excel("review_input.xlsx", index=False)
```

Each object inside the JSON array carries `detected_issue`,
`issue_category`, `issue_severity`, `current_metadata`, and
`suggested_metadata`, the schema the pipeline produces. Sorting by
`issue_severity` gives the triage order, and the key paths in
`current_metadata` and `suggested_metadata` identify the field to change.
Applying an accepted suggestion to the metadata source is a separate,
deliberate step; the [metadata curation cookbook](/cookbook/metadata-curation-with-llms/review)
gives a review procedure and a decision log for it.

:::note[No bundled review interface]
Earlier versions of this manual described a standalone HTML review board.
That file is not distributed with this repository.
:::
