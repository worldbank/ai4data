# Quick API Reference

| Call | Returns / effect |
|---|---|
| `MetadataReviewerClient.from_openai(model, api_key, assets_dir=None)` | A client backed by OpenAI. |
| `MetadataReviewerClient.from_anthropic(model, api_key, assets_dir=None)` | A client backed by Anthropic Claude. |
| `MetadataReviewerClient.from_azure(model, azure_endpoint, azure_deployment, api_version, ...)` | A client backed by Azure OpenAI. |
| `MetadataReviewerClient.from_ollama(model, host, port, assets_dir=None)` | A client backed by a local Ollama server. |
| `MetadataReviewerClient(model_client, assets_dir=None)` | A client around your own AutoGen client. |
| `client.submit(metadata, manifest_file, team_preset)` | Submit synchronously; returns a Job immediately. |
| `await client.submit_async(metadata, manifest_file, team_preset)` | Submit from async code; returns a Job. |
| `client.get_job(job_id)` | Look up a tracked Job. |
| `client.list_jobs()` | All tracked jobs. |
| `client.cleanup_jobs(keep_statuses=None)` | Drop finished jobs; returns the count removed. |
| `client.list_manifests()` | Available manifest file names. |
| `job.wait_sync(timeout=None)` | Block for the result; raise on failure/cancel. |
| `await job.wait(timeout=None)` | Async wait for the result. |
| `job.cancel()` | Request cooperative cancellation. |
