# Jobs

## Job states

A job moves through the following states, exposed as strings on
`job.status` and as constants on `JobStatus`:

| State | Constant | Meaning |
|---|---|---|
| `pending` | `JobStatus.PENDING` | Created but not yet started. |
| `running` | `JobStatus.RUNNING` | The pipeline is executing. |
| `done` | `JobStatus.DONE` | Completed successfully; `job.result` holds the issue list. |
| `failed` | `JobStatus.FAILED` | An exception occurred; `job.error` holds the message. |
| `cancelled` | `JobStatus.CANCELLED` | Stopped in response to `job.cancel()`. |

The valid transitions are:

```
pending → running → done → failed → cancelled
```

## Waiting for results

| Method | Behaviour |
|---|---|
| `job.wait_sync(timeout=None)` | Blocks the calling thread until the job completes, fails, or the timeout expires, then returns `job.result`. |
| `await job.wait(timeout=None)` | The async equivalent; suspends the coroutine instead of blocking the thread. |

Both raise `RuntimeError` if the job ended in the failed or cancelled
state, so wrap the call when failure is possible:

```python
try:
    result = job.wait_sync(timeout=300)
except RuntimeError as exc:
    print("Review did not complete:", exc)
```

:::{note} Timeout behaviour
If the timeout elapses before the job finishes, the wait returns
rather than raising, but the result will reflect the job's current
(possibly unfinished) state. Check `job.status` if you need to
distinguish "still running" from "done" after a timeout.
:::

## Cancelling a job

Call `job.cancel()` to request cancellation. This sets an internal flag;
the running pipeline observes it and stops at the next agent boundary,
after which the job transitions to cancelled. Cancellation is therefore
cooperative — it does not interrupt an in-flight LLM call mid-stream,
but it prevents the next agent from starting.

```python
job = client.submit(large_metadata)

# ... later, decide to stop ...
job.cancel()

# the job will settle into the 'cancelled' state at the next boundary
```

## Inspecting and cleaning up jobs

Each Job carries everything you need to inspect its outcome without
re-running:

- `job.job_id` — the unique identifier (a UUID string).
- `job.status` — the current state.
- `job.result` — the list of detected issues once done.
- `job.error` — the error message if the job failed.

Because the client keeps a registry of jobs, call `cleanup_jobs()`
periodically in long-running services to release the memory held by
completed jobs.
