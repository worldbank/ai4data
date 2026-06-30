# Core Concepts

## The five-agent pipeline

Every metadata record passes through five agents in a fixed sequence.
Candidate issues accumulate through the first two passes; the critic
filters them; the categorizer labels the survivors; the severity scorer
assigns impact and closes the pipeline.

| Agent | Receives | Outputs | Role |
|---|---|---|---|
| **primary** | The raw metadata record | JSON array of candidate issues | Independent first-pass scan for all issue types. Precision is preferred, but obvious errors must not be omitted. |
| **secondary** | The raw metadata record | JSON array of candidate issues | An independent re-scan that does not rely on the primary's output, to surface issues the primary may have missed. |
| **critic** | Combined list from primary + secondary | Filtered JSON array | Removes false positives by applying the general, field-level, and data-state exclusion rules (see [Advanced Usage](advanced-usage.md)). Only unambiguous issues pass. |
| **categorizer** | The critic's filtered list | Same array, with `issue_category` added | Annotates each surviving issue with exactly one category. Adds or removes nothing. |
| **severity_scorer** | The categorizer's list | Final array, with `issue_severity` added | Assigns a 1–5 impact score to each issue and emits the termination signal that ends the run. |

:::{note} Why two detectors?
Running two independent first-pass scans (primary and secondary)
increases recall: a single pass tends to miss issues, while a second,
independent pass catches them. The critic then trims the combined
list back down to only the certain errors, trading a little extra LLM
cost for higher coverage.
:::

## Issue categories

The categorizer assigns exactly one label to each confirmed issue. The
bundled default manifest uses these five categories:

| Category | Meaning |
|---|---|
| Typo / Language | Clear typos, spelling, grammar, punctuation, or wording errors. |
| Formatting / Structure | Malformed text, invalid format patterns, or broken structure. |
| Missing / Redundant Information | Unquestionably missing required information, or accidental duplication. |
| Inconsistency / Conflict | Direct contradictions across fields. |
| Incorrect / Invalid Content | Clearly wrong facts, values, units, methods, or invalid values. |

## Severity scale

The severity scorer assigns an integer from 1 to 5 based on **impact,
not certainty** — every issue reaching this stage has already passed
the certainty filter, so even minor issues are scored rather than
dropped.

| Score | Label | Meaning |
|---|---|---|
| 1 | Trivial | A clear error, but cosmetic with minimal practical impact. |
| 2 | Low | A minor confirmed quality issue; the meaning remains mostly clear. |
| 3 | Moderate | A confirmed issue that may confuse users or reduce trust. |
| 4 | High | A confirmed issue likely to mislead or to affect correct use. |
| 5 | Critical | A confirmed issue with serious risk of misuse or reputational harm. |

## Output schema

The pipeline returns a JSON array. Each element describes one detected
issue using the following fields:

| Field | Description |
|---|---|
| `detected_issue` | A brief description of the problem identified. |
| `issue_category` | One of the category labels from [Issue categories](#issue-categories). |
| `issue_severity` | An integer 1–5 from the [severity scale](#severity-scale). |
| `current_metadata` | A single-entry object mapping the problematic field's key path to its current value. |
| `suggested_metadata` | A single-entry object mapping the same key path to the proposed corrected value. |

Key paths use dot notation with array indices in brackets — for
example `series_description.name` or `series_description.topics[0].name`.
Both `current_metadata` and `suggested_metadata` contain exactly one item,
so an issue always points at one specific field.

## Architecture: Client, Core, and Job

The implementation is split across three classes with distinct
responsibilities:

| Class | Responsibility |
|---|---|
| `MetadataReviewerClient` | The public entry point. Builds the model client, submits jobs (synchronously or asynchronously), tracks job state, and exposes cancellation and cleanup. This is the only class most users touch. |
| `MetadataReviewerCore` | The pipeline engine. On each run it loads the agent manifest, constructs the AutoGen agents, assembles the team, runs the conversation, and extracts the final JSON. It holds no provider-specific logic. |
| `Job` | A handle for one submitted request. Carries the `job_id`, current `status`, the `result` once complete, and any `error`. Returned immediately from every submission. |

The client owns the `model_client` (the LLM connection) and passes it down
to the core, which in turn passes it to every agent. This is what keeps
the design provider-agnostic: all LLM communication flows through that
single injected reference.
