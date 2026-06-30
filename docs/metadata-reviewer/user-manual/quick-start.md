# Quick Start

Using the package is three steps: build a client, submit metadata, and
retrieve the results.

## Step 1 — build a client

Pick the factory method for your provider. Each returns a ready-to-use
client.

### OpenAI

```python
from ai4data.metadata.reviewer import MetadataReviewerClient

client = MetadataReviewerClient.from_openai(
    model="gpt-4o",
    api_key="sk-...",
)
```

### Anthropic Claude

```python
client = MetadataReviewerClient.from_anthropic(
    model="claude-sonnet-4-6",
    api_key="sk-ant-...",
)
```

### Azure OpenAI

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

client = MetadataReviewerClient.from_azure(
    model="gpt-4o",
    azure_endpoint="https://<resource>.openai.azure.com/",
    azure_deployment="<deployment>",
    api_version="2024-02-01",
    azure_ad_token_provider=token_provider,
)
```

### Ollama (local)

```python
client = MetadataReviewerClient.from_ollama(
    model="llama3.2",
    port=11434,  # host defaults to http://localhost
)
```

## Step 2 — submit metadata

Call `submit()` with your metadata (a dict or a JSON string). It returns a
Job handle **immediately**; the five-agent pipeline runs in a background
thread.

```python
job = client.submit(metadata_dict)
print(job)  # Job(id='...', status='running')
```

## Step 3 — retrieve the results

Block until the job finishes with `job.wait_sync(timeout=...)`. It
returns the list of detected issues, or raises `RuntimeError` if the job
failed or was cancelled.

```python
result = job.wait_sync(timeout=300)

for issue in result:
    print(f"[{issue['issue_severity']}/5] {issue['detected_issue']}")
    print(f"  Category: {issue['issue_category']}")
    print(f"  Current: {issue['current_metadata']}")
    print(f"  Suggested: {issue['suggested_metadata']}")
```

## A complete worked example

The following end-to-end example reviews a single indicator metadata
record. The values are illustrative; substitute your own.

### Input metadata

```python
metadata = {
    "series_description": {
        "idno": "WB_WDI_NY.GDP.PCAP.CD",
        "name": "GDP per capita (curent US$)",
        "database_id": "WDI",
        "measurement_unit": "Constant 2015 US$",
        "periodicity": "Annual",
        "definition_long": (
            "GDP per capita is gross domestic product divided by "
            "midyear population, expressed in current US dollars."
        ),
        "aggregation_method": "Weighted average",
        "time_coverage": "1960-2023",
    }
}
```

### Run it

```python
from ai4data.metadata.reviewer import MetadataReviewerClient

client = MetadataReviewerClient.from_anthropic(
    model="claude-sonnet-4-6", api_key="sk-ant-...",
)

job = client.submit(metadata)
issues = job.wait_sync(timeout=300)
```

### Representative output

The pipeline returns a ranked list. Here the run finds a typo and a unit
contradiction (the name and `definition_long` both say **current** US
dollars, but `measurement_unit` says **constant** 2015 US$):

```json
[
  {
    "detected_issue": "Typo in series name: 'curent' should be 'current'.",
    "issue_category": "Typo / Language",
    "issue_severity": 2,
    "current_metadata": {"series_description.name": "GDP per capita (curent US$)"},
    "suggested_metadata": {"series_description.name": "GDP per capita (current US$)"}
  },
  {
    "detected_issue": "Measurement unit contradicts the name and definition, which both state current US dollars.",
    "issue_category": "Inconsistency / Conflict",
    "issue_severity": 4,
    "current_metadata": {"series_description.measurement_unit": "Constant 2015 US$"},
    "suggested_metadata": {"series_description.measurement_unit": "Current US$"}
  }
]
```

:::{note} What you will NOT see
Notice the example has no issue raised about `idno` or about the date
format in `time_coverage`. Those fall under the built-in exclusion
rules ([Advanced Usage](advanced-usage.md)): the `idno` field is excluded entirely, and
formatting/style-only differences are filtered by the critic. This is
by design — the pipeline favors actionable content errors over
cosmetic noise.
:::
