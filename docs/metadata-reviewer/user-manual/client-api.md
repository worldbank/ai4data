# The Client API

## The constructor

If you have already built an AutoGen-compatible model client, construct
`MetadataReviewerClient` directly:

```python
MetadataReviewerClient(
    model_client,       # a pre-built AutoGen ChatCompletionClient
    assets_dir=None,    # directory holding agent-manifest YAML files
)
```

| Parameter | Description |
|---|---|
| `model_client` | Any object implementing AutoGen's `ChatCompletionClient` protocol. The factory methods below build this for you. |
| `assets_dir` | Optional path to a directory of manifest YAML files. Defaults to the `agents_manifest/` directory bundled inside the package. Set this to point at your own custom manifests. |

:::{note} assets_dir vs. manifest_file
The directory of manifests is set **once, on the constructor** via
`assets_dir`. The specific file to use is chosen **per submission** via
the `manifest_file` argument to `submit()`. Do not pass `assets_dir` to
`submit()` — it is not a parameter there.
:::

## Factory classmethods

These four classmethods cover the common providers. All four also accept
the optional `assets_dir` argument.

### from_openai

| Parameter | Description |
|---|---|
| `model` | Model name, e.g. `"gpt-4o"`, `"gpt-4o-mini"`. |
| `api_key` | Your OpenAI API key. |

For reproducibility the client is built with a fixed `seed=1029` and
`temperature=0` (or `1` for gpt-5 models, which require it). It also
advertises JSON / structured-output support to AutoGen.

### from_anthropic

| Parameter | Description |
|---|---|
| `model` | Model name, e.g. `"claude-sonnet-4-6"`, `"claude-haiku-4-5"`. |
| `api_key` | Your Anthropic API key. |

### from_azure

| Parameter | Description |
|---|---|
| `model` | Model name, e.g. `"gpt-4o"`. |
| `azure_endpoint` | Azure OpenAI endpoint URL. |
| `azure_deployment` | Azure deployment name. |
| `api_version` | API version string, e.g. `"2024-02-01"`. |
| `azure_ad_token_provider` | Optional token-provider callable from `azure.identity`. |
| `azure_ad_token` | Optional static Azure AD token; use when no provider is available. |

### from_ollama

| Parameter | Description |
|---|---|
| `model` | Model name, e.g. `"llama3.2"`, `"mistral"`. |
| `host` | Host of the Ollama server. Defaults to `"http://localhost"`. |
| `port` | Port of the Ollama server. Defaults to `11434`. |

## Submitting jobs: submit vs. submit_async

Both methods accept the same arguments and return a Job immediately.
They differ only in how the pipeline is scheduled.

| Argument | Description |
|---|---|
| `metadata` | The metadata to scan, as a dict or a JSON string. |
| `manifest_file` | Name of the YAML manifest file inside `assets_dir`. Defaults to the bundled `default_agents_manifest.yml`. |
| `team_preset` | AutoGen team routing strategy (see [Advanced Usage](advanced-usage.md)). Defaults to `"RoundRobinGroupChat"`. |

| Method | Use when |
|---|---|
| `submit(...)` | You are in ordinary synchronous code (a script, a REPL, a notebook). The pipeline runs in a daemon thread with its own event loop, so it is safe to call even when no event loop exists. |
| `await submit_async(...)` | You are already inside an async context. It schedules the pipeline as an asyncio Task in the current event loop. |

Synchronous example:

```python
job = client.submit(metadata, team_preset="RoundRobinGroupChat")
result = job.wait_sync(timeout=300)
```

Asynchronous example:

```python
async def review(metadata):
    job = await client.submit_async(metadata)
    return await job.wait(timeout=300)
```

## Job-management methods

The client tracks every job it creates so you can look them up later.

| Method | Description |
|---|---|
| `get_job(job_id)` | Return the Job with the given ID, or raise `KeyError`. |
| `list_jobs()` | Return all tracked jobs. |
| `cleanup_jobs(keep_statuses=None)` | Remove finished jobs from the registry and return how many were removed. By default it keeps only pending and running jobs. |
| `list_manifests()` | Return the available YAML manifest file names found in `assets_dir`. |

```python
# discard everything that has finished, keeping only active jobs
removed = client.cleanup_jobs()
print(f"Cleared {removed} finished jobs")

# which manifests are available?
print(client.list_manifests())  # ['default_agents_manifest.yml', ...]
```
