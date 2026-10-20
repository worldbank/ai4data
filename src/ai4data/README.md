# ai4data

Python package of the **AI for Data – Data for AI** program of the World Bank's
Development Data Group and Office of the Chief Statistician. The program applies
AI to the production, curation, and dissemination of development data, and
prepares development data for use by AI systems.

Documentation: https://worldbank.github.io/ai4data/
Repository: https://github.com/worldbank/ai4data

## Installation

```bash
pip install ai4data              # core
pip install "ai4data[all]"       # every capability
```

Each capability is an extra, so a deployment installs only what it uses.

| Extra | Capability |
|---|---|
| `datause` | Dataset mention extraction from text and PDF documents |
| `harmonization` | Clustering and deduplication of dataset names across a corpus |
| `anomaly` | Anomaly detection and explanation of flagged points in time series |
| `search` | Semantic search over catalog records |
| `metadata` | Schema mapping, PDF-to-metadata extraction, and metadata augmentation |
| `discovery` | NADA catalog access, metadata templates, and PDF helpers |
| `metadata-reviewer` | The five-agent metadata review pipeline, combined with one provider extra: `openai`, `anthropic`, `azure`, or `ollama` |

## Examples

Dataset mentions in a PDF, one result per page:

```python
from ai4data.data_use import extract_from_document

for page in extract_from_document("policy_report.pdf", include_confidence=True):
    for ds in page["datasets"]:
        name = ds.get("mention_name") or {}
        print(page["page"], name.get("text"), name.get("confidence"), ds.get("specificity_tag"))
```

Review of a metadata record by the five-agent pipeline:

```python
from ai4data.metadata.reviewer import MetadataReviewerClient

client = MetadataReviewerClient.from_openai(model="gpt-4o", api_key="sk-...")
job = client.submit(metadata_dict)
issues = job.wait_sync(timeout=300)
```

The documentation site describes each module: monitoring of data use,
anomaly detection and explanation, data discoverability, metadata
augmentation, and the Metadata Reviewer user manual.

## License

The code is licensed under the MIT License together with the World Bank IGO
Rider, both in the repository's LICENSE file.
