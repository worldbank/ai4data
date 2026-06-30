# Advanced Usage

## Team presets

The `team_preset` argument controls how AutoGen routes messages between
the agents. The default, `RoundRobinGroupChat`, steps through the agents
in manifest order and is what you want for an auditable, predictable
pipeline. The alternatives change the routing strategy:

| Preset | Routing | When to use |
|---|---|---|
| `RoundRobinGroupChat` | Fixed sequential order | Standard pipeline; predictable, auditable turn order. (Default.) |
| `SelectorGroupChat` | An LLM selects the next agent | When the agent order should vary with the content of prior output. |
| `MagenticOneGroupChat` | A dedicated orchestrator | Complex multi-step reasoning where an orchestrator decomposes the task. |
| `Swarm` | Agent-to-agent handoff | Distributed, loosely coupled execution where agents pick their own successors. |

```python
job = client.submit(metadata, team_preset="SelectorGroupChat")
```

:::{note} Turns are bounded by the agent count
Internally the team is run with `max_turns` equal to the number of
agents in the manifest. With the default five-agent manifest, the
conversation runs exactly five turns — one per agent. This matters
when you write a custom manifest: the team will take as many turns as
you have agents, so the agent that produces your final JSON must be
the **last** one.
:::

## Custom agents manifest

To change the agents' instructions, supply your own YAML manifest.
Point the constructor's `assets_dir` at the directory that contains it,
and name the file in `submit()`:

```python
client = MetadataReviewerClient.from_openai(
    model="gpt-4o",
    api_key="sk-...",
    assets_dir="/path/to/my/manifests/",
)

job = client.submit(metadata, manifest_file="custom_manifest.yml")
```

The YAML structure is a top-level `agents_manifest` list. Each entry needs
a `name` and a `system_message`. The names define the agents' identities in
the pipeline; the system message is passed straight to the AutoGen
agent. Entries missing either field are skipped with a warning.

```yaml
agents_manifest:
  - name: primary
    system_message: |
      Examine the metadata and list any issues that are incorrect,
      missing, or inconsistent. Output a JSON array using the
      standard schema.
  - name: severity_scorer
    system_message: |
      Assign issue_severity (1-5) to each finding based on impact.
      Output a JSON array. Print a final line: TERMINATE
```

:::{note} Critical: end with TERMINATE
The pipeline stops when an agent's message contains the exact word
**TERMINATE**. Your final agent's system message must instruct it to
print that word, or the run will continue until it hits the turn
limit and may not return clean output. The final JSON the package
returns is extracted from the **last** agent's message, so make sure
your last agent emits the complete result array.
:::

## The built-in exclusion rules

In the default manifest, the **critic** removes any candidate matching
the exclusion classes below, and the **severity scorer** applies the
same classes as a down-weighting safety net — anything that slips past
the critic but still matches is forced to severity 1. Understanding
these rules explains why certain "issues" never appear in your output.

### General exclusions (removed entirely)

Capitalization-only; spacing or whitespace; style or stylistic
preference; CRLF, newlines, blank lines, or trailing spaces; formatting
or encoding; abbreviations; code-related issues; empty lists or empty
arrays; missing fields; schema or schema-structure issues; mixed-type
objects that reflect structural variation; and URL-structure issues.

### Field-level exclusions (issues on these fields are removed)

`idno`, `proj_idno`, `version_statement`, `prod_date`, `version_date`, `changed`,
`changed_by`, `contacts`, `topics`, `tags`, `database_id`, `visualization`, and `uri`.

### Data-state exclusions (removed)

Issues about null or empty fields; empty lists; nested empty lists; and
placeholder-only values with no semantic content.

If you need the reviewer to flag, say, capitalization or a
normally-excluded field, write a [custom manifest](advanced-usage.md) whose critic
omits the relevant rule.

## Bring your own client

For full control — or for any AutoGen-compatible provider the factory
methods do not cover — build the model client yourself and pass it to
the constructor.

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

Any object conforming to AutoGen's `ChatCompletionClient` interface
works. This is also the route to use a static Azure AD token instead of
a token provider, as shown above.
