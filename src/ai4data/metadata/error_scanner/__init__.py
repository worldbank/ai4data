"""ai4data.metadata.error_scanner — async AI-powered metadata error detection.

Install with: uv pip install ai4data[error_scanner]

Quickstart
----------
    from ai4data.metadata.error_scanner import ErrorScannerClient

    client = ErrorScannerClient(openai_api_key="sk-...")

    # Submit a job (returns immediately)
    job = client.submit(metadata_dict)
    print(job.job_id, job.status)   # → "pending" / "running"

    # Blocking wait
    result = job.wait_sync(timeout=300)

    # Async wait
    result = await job.wait(timeout=300)

    # Retrieve a job later by ID
    job = client.get_job(job_id)
"""

from .client import ErrorScannerClient
from .jobs import Job, JobStatus

__all__ = ["ErrorScannerClient", "Job", "JobStatus"]
