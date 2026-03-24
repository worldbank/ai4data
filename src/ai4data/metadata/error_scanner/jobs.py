import uuid
import asyncio
import threading


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job:
    """Represents an async error-scanning job."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status: str = JobStatus.PENDING
        self.result = None          # list of detected issues (JSON) when done
        self.error: str | None = None
        self._done_event = threading.Event()
        self._cancel_flag = threading.Event()

    def cancel(self):
        """Request cancellation."""
        self._cancel_flag.set()

    def wait_sync(self, timeout: float | None = None):
        """Block until the job completes. Returns result or raises on failure."""
        self._done_event.wait(timeout=timeout)
        if self.status == JobStatus.FAILED:
            raise RuntimeError(f"Job {self.job_id} failed: {self.error}")
        if self.status == JobStatus.CANCELLED:
            raise RuntimeError(f"Job {self.job_id} was cancelled.")
        return self.result

    async def wait(self, timeout: float | None = None):
        """Async wait for completion. Returns result or raises on failure."""
        await asyncio.to_thread(self._done_event.wait, timeout)
        if self.status == JobStatus.FAILED:
            raise RuntimeError(f"Job {self.job_id} failed: {self.error}")
        if self.status == JobStatus.CANCELLED:
            raise RuntimeError(f"Job {self.job_id} was cancelled.")
        return self.result

    def __repr__(self):
        return f"Job(id={self.job_id!r}, status={self.status!r})"
