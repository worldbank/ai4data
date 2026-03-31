# Implementation of ai4data.metadata.reviewer

This page describes the internal architecture of `ai4data.metadata.reviewer`: how its two main classes divide responsibility, how the five-agent pipeline processes a metadata record, how jobs are managed, and how the design remains independent of any specific LLM provider.

The implementation is in [`src/ai4data/metadata/reviewer/`](../../src/ai4data/metadata/reviewer/).

---

## Architecture Overview

The implementation is split across two classes with distinct roles:

```
MetadataReviewerClient          (public API)
MetadataReviewerCore            (pipeline engine)
```

**`MetadataReviewerClient`** is the entry point for all external use. It handles job submission in both synchronous and asynchronous modes, tracks job state, and provides cancellation. It owns the `model_client` instance (the LLM connection) and passes it to `MetadataReviewerCore` at construction time.

**`MetadataReviewerCore`** manages the AutoGen agent sessions. On each `run()` call it loads the agents manifest, constructs the AutoGen agent objects, assembles the team, and runs the conversation pipeline against the input metadata. It holds no provider-specific logic; all LLM communication goes through the `model_client` reference it receives.

**`Job`** (in `jobs.py`) represents a single submitted request. It carries the job's `job_id`, current `status`, `result` (once complete), and `error` (on failure). Both submission modes return a `Job` immediately; the caller inspects or awaits it to retrieve the outcome.

---

## The Five-Agent Pipeline

Each run passes the metadata record through five agents in sequence. The table below summarises each agent's role.

| Agent | Receives | Outputs | Key behaviour |
|---|---|---|---|
| **primary** | Raw metadata record | JSON array of candidate issues (`detected_issue`, `current_metadata`, `suggested_metadata`) | Independent first-pass scan for all issue types: incorrect, inconsistent, contradictory, missing, duplicated, unclear, typos. Precision preferred; obvious issues must not be omitted. |
| **secondary** | Raw metadata record | JSON array of candidate issues (same schema) | Independent re-scan. Does NOT rely on primary's output. Surfaces issues the primary may have missed. |
| **critic** | Combined candidate list from primary and secondary | Filtered JSON array | Removes issues matching general, field-level, and data-state exclusion rules (see below). No speculation; only unambiguous issues pass. |
| **categorizer** | Critic's filtered list | Same array with `issue_category` added | Assigns one of six exact category strings. Does not add or remove findings; only annotates. Tie-breaker rules applied in fixed order. |
| **severity_scorer** | Categorizer's annotated list | Final array with `issue_severity` (integer 1–5) added | Assigns severity based on impact, not category alone. Down-weights issues matching exclusion classes to severity 1. Terminates with "DONE" to signal pipeline end. |

Candidates accumulate through the primary and secondary passes. The critic applies a defined filter that removes noise. The categorizer annotates surviving issues without altering them. The severity scorer closes the pipeline with an impact-based score and the termination signal.

---

## Agent Exclusion Rules (Critic and Severity Scorer)

The critic removes issues matching any of the three exclusion classes below. The severity scorer applies the same classes as down-weighting rules: any issue that passes the critic but still matches these conditions receives `issue_severity = 1` rather than being removed.

### General Exclusions

Issues of these types are removed entirely: capitalization-only, spacing or whitespace, style or stylistic preference, CRLF or newline or blank-line or trailing-space, formatting or encoding, abbreviation, code, empty list, missing fields, schema or schema structure, mixed-type objects reflecting structural variation, and URL structure.

### Field-Level Exclusions

Issues involving any of the following metadata fields are removed:

`idno`, `proj_idno`, `version_statement`, `prod_date`, `version_date`, `changed`, `changed_by`, `contacts`, `topics`, `tags`, `database_id`, `visualization`

### Data-State Exclusions

Issues related to null or empty fields, empty lists, nested empty lists, or placeholder-only values with no semantic content are removed.

---

## Job Lifecycle

A submitted job moves through the following states:

```
pending → running → done
                 → failed
                 → cancelled
```

**Synchronous submission** (`client.submit(metadata)`) spawns a daemon thread with its own `asyncio.run()` event loop and returns a `Job` immediately. The pipeline runs in that background thread. This mode is safe to call from any context, including a REPL or a script with no existing event loop.

**Asynchronous submission** (`await client.submit_async(metadata)`) creates an asyncio `Task` in the caller's event loop and returns a `Job` immediately. This mode is appropriate when the caller is already running inside an async context.

**Waiting for results:**

- `job.wait_sync(timeout=...)` — blocks the calling thread until the job completes, fails, or the timeout expires.
- `await job.wait(timeout=...)` — async equivalent; suspends the coroutine instead of blocking the thread.

Both raise `RuntimeError` if the job ends in a `failed` or `cancelled` state.

**Cancellation** is handled through an `ExternalTermination` handle stored in the session. Calling `job.cancel()` sets the cancellation flag; the running pipeline observes it and stops at the next agent boundary. The job transitions to `cancelled`.

---

## Team Presets

The team preset controls how AutoGen routes messages between agents. The default is `RoundRobinGroupChat`, which steps through agents in the order defined in the manifest. Alternative presets can be passed via the `team_preset` parameter on `submit()` or `submit_async()`.

| Preset | Routing mechanism | When to use |
|---|---|---|
| `RoundRobinGroupChat` (default) | Fixed sequential order | Standard pipeline execution; predictable, auditable turn order |
| `SelectorGroupChat` | LLM selects the next agent | Dynamic routing based on prior output; useful when agent order should vary by content |
| `MagenticOneGroupChat` | Dedicated orchestrator agent | Complex multi-step reasoning; orchestrator manages task decomposition |
| `Swarm` | Agent-to-agent handoff | Distributed, loosely coupled execution; agents decide their own successors |

---

## Custom Agents Manifest

The default manifest (`agents_manifest/default_agents_manifest.yml`) is bundled inside the package. To use a custom manifest, pass `assets_dir` to the `MetadataReviewerClient` constructor and `manifest_file` to `submit()` or `submit_async()`.

The YAML structure is a top-level `agents_manifest` list. Each entry has a `name` and a `system_message`. The `name` values determine agent identity within the pipeline; the `system_message` is passed directly to the AutoGen agent at construction.

Minimal custom manifest:

```yaml
agents_manifest:
  - name: primary
    system_message: |
      Examine the metadata and list any issues that are incorrect, missing, or inconsistent.
      Output a JSON array using the standard schema.

  - name: severity_scorer
    system_message: |
      Assign issue_severity (1–5) to each finding based on impact.
      Output a JSON array. Print a final line: DONE
```

To use it:

```python
client.submit(
    metadata,
    assets_dir="/path/to/my/manifest/",
    manifest_file="custom_manifest.yml",
)
```

---


## References

- [`src/ai4data/metadata/reviewer/core.py`](../../src/ai4data/metadata/reviewer/core.py) — `MetadataReviewerCore` implementation
- [`src/ai4data/metadata/reviewer/client.py`](../../src/ai4data/metadata/reviewer/client.py) — `MetadataReviewerClient` implementation
- [`src/ai4data/metadata/reviewer/jobs.py`](../../src/ai4data/metadata/reviewer/jobs.py) — `Job` class
- [`src/ai4data/metadata/reviewer/agents_manifest/default_agents_manifest.yml`](../../src/ai4data/metadata/reviewer/agents_manifest/default_agents_manifest.yml) — Default agents manifest
- [Microsoft AutoGen documentation](https://microsoft.github.io/autogen/) — Multi-agent framework
- [autogen-agentchat](https://pypi.org/project/autogen-agentchat/) — AutoGen conversation and team orchestration library
