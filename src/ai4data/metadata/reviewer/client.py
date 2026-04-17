from __future__ import annotations

import asyncio
import threading
import uuid
import logging
from typing import Any

from .core import MetadataReviewerCore, _DEFAULT_MANIFEST_FILE, _DEFAULT_TEAM_PRESET
from .jobs import Job, JobStatus

logger = logging.getLogger(__name__)


class MetadataReviewerClient:
    """
    High-level client for submitting jobs asynchronously.

    Parameters
    ----------
    model_client : ChatCompletionClient
        Pre-built autogen ChatCompletionClient (OpenAI, Azure, Ollama, Anthropic, etc.).
    assets_dir : str, optional
        Path to the agents-manifest directory.
        Defaults to the bundled ``agents_manifest/`` inside this package.
    """

    def __init__(
        self,
        model_client: Any,
        assets_dir: str | None = None,
    ):
        self._core = MetadataReviewerCore(model_client, assets_dir)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # ── Factory classmethods ──────────────────────────────────────────────

    @classmethod
    def from_openai(
        cls,
        model: str,
        api_key: str,
        assets_dir: str | None = None,
    ) -> "MetadataReviewerClient":
        """Create a client using OpenAI.

        Requires: ``ai4data[metadata-reviewer,openai]``

        Parameters
        ----------
        model : str
            Model name, e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``.
        api_key : str
            OpenAI API key.
        assets_dir : str, optional
            Path to the agents-manifest directory.
        """
        import httpx
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        temperature = 1 if model.startswith("gpt-5") else 0
        model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            temperature=temperature,
            seed=1029,
            http_client=httpx.AsyncClient(verify=False),
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
        )
        return cls(model_client=model_client, assets_dir=assets_dir)

    @classmethod
    def from_azure(
        cls,
        model: str,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str,
        azure_ad_token_provider=None,
        azure_ad_token: str | None = None,
        assets_dir: str | None = None,
    ) -> "MetadataReviewerClient":
        """Create a client using Azure OpenAI.

        Requires: ``ai4data[metadata-reviewer,azure]``

        Parameters
        ----------
        model : str
            Model name, e.g. ``"gpt-4o"``.
        azure_endpoint : str
            Azure OpenAI endpoint URL.
        azure_deployment : str
            Azure deployment name.
        api_version : str
            API version string, e.g. ``"2024-02-01"``.
        azure_ad_token_provider : callable, optional
            Token provider callable from ``azure.identity``.
        azure_ad_token : str, optional
            Static Azure AD token. Use when a token provider is not available.
        assets_dir : str, optional
            Path to the agents-manifest directory.
        """
        import httpx
        from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

        model_client = AzureOpenAIChatCompletionClient(
            model=model,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            azure_ad_token_provider=azure_ad_token_provider,
            azure_ad_token=azure_ad_token,
            http_client=httpx.AsyncClient(verify=False),
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
        )
        return cls(model_client=model_client, assets_dir=assets_dir)

    @classmethod
    def from_ollama(
        cls,
        model: str,
        host: str = "http://localhost",
        port: int = 11434,
        assets_dir: str | None = None,
    ) -> "MetadataReviewerClient":
        """Create a client using a local Ollama server.

        Requires: ``ai4data[metadata-reviewer,ollama]``

        Parameters
        ----------
        model : str
            Model name, e.g. ``"llama3.2"``, ``"mistral"``.
        host: str, optional
            Host of the Ollama server. Defaults to ``"http://localhost"``.
        port : int, optional
            Port of the Ollama server. Defaults to ``11434``.
        assets_dir : str, optional
            Path to the agents-manifest directory.
        """
        from autogen_ext.models.ollama import OllamaChatCompletionClient

        model_client = OllamaChatCompletionClient(
            model=model,
            host=f"{host}:{port}",
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
        )
        return cls(model_client=model_client, assets_dir=assets_dir)

    @classmethod
    def from_anthropic(
        cls,
        model: str,
        api_key: str,
        assets_dir: str | None = None,
    ) -> "MetadataReviewerClient":
        """Create a client using Anthropic Claude.

        Requires: ``ai4data[metadata-reviewer,anthropic]``

        Parameters
        ----------
        model : str
            Model name, e.g. ``"claude-sonnet-4-6"``, ``"claude-haiku-4-5"``.
        api_key : str
            Anthropic API key.
        assets_dir : str, optional
            Path to the agents-manifest directory.
        """
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient

        model_client = AnthropicChatCompletionClient(
            model=model,
            api_key=api_key,
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
        )
        return cls(model_client=model_client, assets_dir=assets_dir)

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
        team_preset: str = _DEFAULT_TEAM_PRESET,
    ) -> Job:
        """
        Submit a metadata reviewing job.

        The job runs in a background thread with its own event loop.
        Returns a :class:`Job` immediately.

        Parameters
        ----------
        metadata : dict | str
            The metadata to scan (dict or JSON string).
        manifest_file : str
            Name of the YAML manifest file in ``assets_dir``.
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
            asyncio.run(self._execute_job(job, metadata, manifest_file, team_preset))

        thread = threading.Thread(target=_run_in_thread, daemon=True, name=f"metadata-reviewer-{job.job_id[:8]}")
        thread.start()
        logger.info("Submitted job %s (preset=%s)", job.job_id, team_preset)
        return job

    # ── Submit (async) ────────────────────────────────────────────────────

    async def submit_async(
        self,
        metadata: Any,
        manifest_file: str = _DEFAULT_MANIFEST_FILE,
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
            self._execute_job(job, metadata, manifest_file, team_preset),
            name=f"metadata-reviewer-{job.job_id[:8]}",
        )
        logger.info("Submitted async job %s (preset=%s)", job.job_id, team_preset)
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
        team_preset: str,
    ):
        job.status = JobStatus.RUNNING
        try:
            result = await self._core.run(
                metadata_to_scan=metadata,
                manifest_file=manifest_file,
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
