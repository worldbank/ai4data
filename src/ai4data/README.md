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

Each capability is an optional extra.

| Extra | Capability |
|---|---|
| `datause` | Dataset mention extraction from text and PDF documents |
| `harmonization` | Clustering and deduplication of dataset names across a corpus |
| `anomaly` | Anomaly detection and explanation of flagged points in time series |
| `search` | Semantic search over catalog records |
| `metadata` | Schema mapping, PDF-to-metadata extraction, and metadata augmentation |
| `discovery` | NADA catalog access, metadata templates, and PDF helpers |
| `metadata-reviewer` | The five-agent metadata review pipeline, combined with one provider extra: `openai`, `anthropic`, `azure`, or `ollama` |
| `data-snapshot-doclayoutyolo` | Figure and table images from PDFs with the default DocLayout-YOLO extractor |
| `data-snapshot-metadata` | Validated Schema v1.4 metadata from snapshot images through OpenRouter or OpenAI |

For the Data Snapshot pipeline, combine `data-snapshot-doclayoutyolo` and
`data-snapshot-metadata`. These extras are absent from `ai4data==0.1.0`; until a
release includes them, install the checkout as described in the
[Data Snapshot usage guide](https://github.com/worldbank/ai4data/blob/main/src/ai4data/data_snapshot/README.md).
In this checkout, `all` includes DocLayout-YOLO and snapshot metadata extraction.
TF-ID requires a separate environment; the guide lists its installation steps.

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
Data Snapshot documentation currently covers repository installation and usage;
concepts, methods, worked examples, and user guides for the published site are
planned in a separate PR.

## License

The code is licensed under the MIT License together with the World Bank IGO
Rider, both in the repository's LICENSE file.
