from __future__ import annotations

import asyncio
import threading
import uuid
import logging
from typing import Any

from .core import ErrorScannerCore, _DEFAULT_MANIFEST_FILE, _DEFAULT_MODEL, _DEFAULT_TEAM_PRESET
from .jobs import Job, JobStatus

logger = logging.getLogger(__name__)


class ErrorScannerClient:
    """
    High-level client for submitting error-scanning jobs asynchronously.

    Parameters
    ----------
    openai_api_key : str
        OpenAI API key.
    assets_dir : str, optional
        Path to the agents-manifest directory.
        Defaults to the bundled ``agents_manifest/`` inside this package.
    """

    def __init__(
        self,
        openai_api_key: str,
        assets_dir: str | None = None,
    ):
        self._core = ErrorScannerCore(openai_api_key, assets_dir)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # ── Job management ────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> Job:
        """Return the Job with the given ID, or raise KeyError."""
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"No job with id {job_id!r}")
            return self._jobs[job_id]

    def list_jobs(self) -> list[Job]:
        """Return all tracked jobs."""
        with self._lock:
            return list(self._jobs.values())

    def cleanup_jobs(self, keep_statuses: set[str] | None = None) -> int:
        """
        Remove finished jobs from the registry.

        Parameters
        ----------
        keep_statuses : set of str, optional
            Job statuses to retain. Defaults to ``{JobStatus.PENDING, JobStatus.RUNNING}``.

        Returns
        -------
        int
            Number of jobs removed.
        """
        if keep_statuses is None:
            keep_statuses = {JobStatus.PENDING, JobStatus.RUNNING}
        with self._lock:
            to_remove = [jid for jid, j in self._jobs.items() if j.status not in keep_statuses]
            for jid in to_remove:
                del self._jobs[jid]
        return len(to_remove)

    # ── Submit (sync wrapper) ─────────────────────────────────────────────

    def submit(
        self,
        metadata: Any,
        manifest_file: str = _DEFAULT_MANIFEST_FILE,
        model: str = _DEFAULT_MODEL,
        team_preset: str = _DEFAULT_TEAM_PRESET,
    ) -> Job:
        """
        Submit a metadata error-scanning job.

        The job runs in a background thread with its own event loop.
        Returns a :class:`Job` immediately.

        Parameters
        ----------
        metadata : dict | str
            The metadata to scan (dict or JSON string).
        manifest_file : str
            Name of the YAML manifest file in ``assets_dir``.
        model : str
            GPT model identifier (e.g. ``"gpt-5.4"``, ``"gpt-5-mini"``).
        team_preset : str
            AutoGen team type: ``"RoundRobinGroupChat"``, ``"SelectorGroupChat"``,
            ``"MagenticOneGroupChat"``, or ``"Swarm"``.

        Returns
        -------
        Job
            A job object you can poll or wait on.
        """
        job = Job(str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job

        def _run_in_thread():
            asyncio.run(self._execute_job(job, metadata, manifest_file, model, team_preset))

        thread = threading.Thread(target=_run_in_thread, daemon=True, name=f"error-scanner-{job.job_id[:8]}")
        thread.start()
        logger.info("Submitted job %s (model=%s, preset=%s)", job.job_id, model, team_preset)
        return job

    # ── Submit (async) ────────────────────────────────────────────────────

    async def submit_async(
        self,
        metadata: Any,
        manifest_file: str = _DEFAULT_MANIFEST_FILE,
        model: str = _DEFAULT_MODEL,
        team_preset: str = _DEFAULT_TEAM_PRESET,
    ) -> Job:
        """
        Async variant of :meth:`submit`.

        Creates an asyncio Task in the current event loop and returns a :class:`Job`.
        """
        job = Job(str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job

        asyncio.create_task(
            self._execute_job(job, metadata, manifest_file, model, team_preset),
            name=f"error-scanner-{job.job_id[:8]}",
        )
        logger.info("Submitted async job %s (model=%s, preset=%s)", job.job_id, model, team_preset)
        return job

    def list_manifests(self) -> list[str]:
        """Return available YAML manifest file names from ``assets_dir``."""
        return self._core.list_manifests()

    # ── Internal ──────────────────────────────────────────────────────────

    async def _execute_job(
        self,
        job: Job,
        metadata: Any,
        manifest_file: str,
        model: str,
        team_preset: str,
    ):
        job.status = JobStatus.RUNNING
        try:
            result = await self._core.run(
                metadata_to_scan=metadata,
                manifest_file=manifest_file,
                model=model,
                team_preset=team_preset,
                cancel_flag=job._cancel_flag,
            )
            if job._cancel_flag.is_set():
                job.status = JobStatus.CANCELLED
            else:
                job.result = result
                job.status = JobStatus.DONE
        except Exception as exc:
            logger.exception("Job %s failed", job.job_id)
            job.error = str(exc)
            job.status = JobStatus.FAILED
        finally:
            job._done_event.set()
