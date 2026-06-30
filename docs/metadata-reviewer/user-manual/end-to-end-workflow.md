# End-to-End Workflow

Putting the pieces together, a typical metadata-QA cycle for a catalogue
of records looks like this:

1. **Configure a client** once, choosing a provider ([Quick Start](quick-start.md)).
2. **Submit each record** and collect its results. For many records,
   submit them and wait in turn, or submit a batch and wait on each Job
   handle.
3. **Assemble a workbook** with one row per record and the issue array
   as a JSON string ([Review Board](review-board.md)).
4. **Triage in the review board** — a human opens the workbook, works
   through projects highest-severity-first, and confirms which
   suggestions to apply ([Review Board](review-board.md)).
5. **Apply accepted fixes** back to your metadata system as a separate
   step, using the key paths in `current_metadata` / `suggested_metadata`
   to target each field.

A minimal batch driver that submits everything first, then gathers
results:

```python
jobs = {name: client.submit(md) for name, md in catalogue.items()}

results = {}
for name, job in jobs.items():
    try:
        results[name] = job.wait_sync(timeout=300)
    except RuntimeError as exc:
        print(f"{name}: {exc}")
        results[name] = []
```

:::{note} Throughput tip
Submitting many jobs at once issues many concurrent LLM calls, which
can hit provider rate limits. If you see rate-limit errors surface as
failed jobs, submit in smaller waves, or run sequentially (submit,
wait, repeat) to keep concurrency low.
:::
