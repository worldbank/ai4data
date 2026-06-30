# Troubleshooting & FAQ

## The job finished but the result is empty or None

- **An empty list** `[]` is a valid, expected result — it means the
  pipeline found no confirmed issues. Remember the critic and the
  exclusion rules deliberately remove anything that is not a certain
  error.
- **A `None` result** usually means the run produced no parseable JSON, or
  it was cancelled before completing. Check `job.status` and, for custom
  manifests, confirm the last agent actually emitted a JSON array.

## The job failed

Read `job.error` for the message. Common causes are an invalid API key, an
unreachable endpoint, an exhausted rate limit, or a model that does not
support the requested options. The exception is also logged with the
job's ID.

## TLS / SSL or connection errors

The OpenAI and Azure factory methods already disable TLS verification to
tolerate inspecting proxies. If you still cannot reach the endpoint, the
traffic is likely blocked entirely — route through your approved
proxy, or switch to a local model with `from_ollama`. For self-built
clients ([Advanced Usage](advanced-usage.md)) you control the HTTP client, so configure proxy
and TLS settings there.

## A custom manifest never terminates or returns nothing useful

- Ensure the final agent's system message instructs it to print the
  exact word **TERMINATE** ([Advanced Usage](advanced-usage.md)).
- Remember the team runs one turn per agent. The agent that should
  produce the final array must be the **last** entry in the manifest.
- Each manifest entry must have both `name` and `system_message`; entries
  missing either are silently skipped.

## cancel() did not stop the job immediately

Cancellation is cooperative and takes effect at the next agent boundary,
not mid-call. An LLM request already in flight will complete; the next
agent simply will not start, and the job then settles into the cancelled
state.

## The review board shows "No valid rows found"

- Confirm the worksheet has the `ME_project` and `detected_issues` columns
  (or their accepted aliases).
- Confirm `detected_issues` contains valid JSON. Building it with
  `json.dumps(...)` as shown in [Review Board](review-board.md) avoids quoting problems.
- The board reads the **first** worksheet only; make sure your data is
  on it.

## Can I make results reproducible?

The OpenAI factory pins a fixed seed and a zero temperature, which makes
output as deterministic as the provider allows. Exact reproducibility
still depends on the provider and model; treat runs as highly consistent
rather than bit-for-bit identical.
