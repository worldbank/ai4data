# Metadata Reviewer

`MetadataReviewerClient` uses a multi-agent LLM pipeline to scan dataset series metadata for quality issues: incorrect content, inconsistencies, typos, missing fields, and more. It returns a ranked list of detected issues with suggested corrections.

---

## The Problem

Metadata quality assurance at scale is difficult. Manually reviewing hundreds of metadata records for typos, inconsistencies, incorrect values, and missing fields is time-consuming and inconsistent. Reviewers applying the same criteria across many records will drift in their judgments over time, and the process leaves no structured audit trail.

LLM-only approaches reduce the labor cost but do not by themselves address persistence, governance, or auditability. A multi-agent pipeline addresses these gaps by decomposing the review task into specialized, sequential agents—each with a defined role—so that detection, filtering, classification, and scoring are handled separately and consistently.

---

## How It Works

Each metadata submission is processed by a five-agent pipeline:

| Agent | Role |
|---|---|
| **primary** | Initial scan for issues: typos, inconsistencies, missing or redundant fields |
| **secondary** | Independent re-scan to catch issues the primary agent missed |
| **critic** | Filters false positives by applying exclusion rules to the combined issue list |
| **categorizer** | Assigns one of six issue categories to each confirmed issue |
| **severity_scorer** | Assigns a severity score (1–5) to each confirmed issue |

The six issue categories assigned by the categorizer are:

- Typo / Language
- Formatting / Structure
- Missing / Redundant Information
- Inconsistency / Conflict
- Incorrect / Invalid Content
- Ambiguity / Unclear

---

## Output Schema

Each detected issue is returned as a structured record:

| Field | Description |
|---|---|
| `detected_issue` | Description of the problem identified |
| `issue_category` | One of the six category labels above |
| `issue_severity` | Integer 1–5 (see scale below) |
| `current_metadata` | The problematic field and its current value |
| `suggested_metadata` | Proposed correction for the field |

**Severity scale:**

| Score | Label | Meaning |
|---|---|---|
| 1 | Trivial | Minor cosmetic issue; no practical impact |
| 2 | Low | Small error unlikely to cause confusion |
| 3 | Moderate | Noticeable issue that may mislead users |
| 4 | High | Significant error that affects usability or trust |
| 5 | Critical | Incorrect or missing information that renders the metadata unreliable |

---

## LLM-Agnostic Design

The metadata reviewer is designed to be independent of any specific LLM provider. The `MetadataReviewerCore` holds a `model_client` — any object implementing AutoGen's `ChatCompletionClient` protocol — and passes it directly to each agent in the pipeline. No provider-specific logic lives inside the reviewer itself.

This means the same five-agent pipeline runs identically regardless of whether the underlying model is a hosted API, an Azure deployment, or a locally served model. Swapping providers requires only changing how the client is constructed.

### Factory Classmethods

The four built-in factory classmethods cover the most common providers. Each uses a **lazy import**: the provider package is only imported when that classmethod is called, so installing only `autogen-ext[openai]` will not cause import errors when `from_anthropic` is never used.

| Provider | Classmethod | Key Parameters |
|---|---|---|
| OpenAI | `MetadataReviewerClient.from_openai(...)` | `model`, `api_key` |
| Azure OpenAI | `MetadataReviewerClient.from_azure(...)` | `model`, `azure_endpoint`, `azure_deployment`, `api_version`, `azure_ad_token_provider`, `azure_ad_token` |
| Anthropic Claude | `MetadataReviewerClient.from_anthropic(...)` | `model`, `api_key` |
| Ollama (local) | `MetadataReviewerClient.from_ollama(...)` | `model`, `port` (default `11434`) |


## Installation

Install with the extras for your chosen LLM provider:

```bash
pip install ai4data[metadata_reviewer,openai]      # OpenAI
pip install ai4data[metadata_reviewer,anthropic]   # Anthropic
pip install ai4data[metadata_reviewer,azure]       # Azure OpenAI
pip install ai4data[metadata_reviewer,ollama]      # Ollama (local)
```

Requirements: Python >= 3.11, `autogen-agentchat`, and credentials for your chosen provider.

---

## Quick Start

**OpenAI**

```python
from ai4data.metadata.reviewer import MetadataReviewerClient

client = MetadataReviewerClient.from_openai(model="gpt-4o", api_key="sk-...")
```

**Anthropic**

```python
client = MetadataReviewerClient.from_anthropic(model="claude-sonnet-4-6", api_key="sk-ant-...")
```

**Azure OpenAI**

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
client = MetadataReviewerClient.from_azure(
    model="gpt-4o",
    azure_endpoint="https://<resource>.openai.azure.com/",
    azure_deployment="<deployment>",
    api_version="2024-02-01",
    azure_ad_token_provider=token_provider,
)
```

**Ollama (local)**

```python
client = MetadataReviewerClient.from_ollama(model="llama3.2", port=11434)
```

**Submit and retrieve results**

```python
# Submit metadata for review — returns immediately
job = client.submit(metadata_dict)

# Block until complete
result = job.wait_sync(timeout=300)

# Each item is a detected issue
for issue in result:
    print(f"[{issue['issue_severity']}/5] {issue['detected_issue']}")
    print(f"  Category: {issue['issue_category']}")
    print(f"  Suggested fix: {issue['suggested_metadata']}")
```

`client.submit()` returns a `Job` handle immediately; the pipeline runs in a background thread. Use `job.wait_sync(timeout)` to block, or `await job.wait(timeout)` in an async context.

---

### Bringing Your Own Client

For full control over model client configuration — or for any AutoGen-compatible provider not covered by the factory classmethods — construct the client yourself and pass it directly to the constructor:

```python
from ai4data.metadata.reviewer import MetadataReviewerClient

client = MetadataReviewerClient(model_client=your_model_client)
```

Any object that conforms to AutoGen's `ChatCompletionClient` interface works. For example, using `AzureOpenAIChatCompletionClient` with a static token instead of a token provider:

```python
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from ai4data.metadata.reviewer import MetadataReviewerClient

model_client = AzureOpenAIChatCompletionClient(
    model="gpt-4o",
    azure_endpoint="https://<resource>.openai.azure.com/",
    azure_deployment="<deployment>",
    api_version="2024-02-01",
    azure_ad_token="<static-token>",
)

client = MetadataReviewerClient(model_client=model_client)
```



