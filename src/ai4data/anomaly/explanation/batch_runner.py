"""Submit, poll, and download LLM batch jobs for OpenAI, Gemini, and Anthropic."""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from ai4data.anomaly.explanation.llm_client import ENDPOINT_URLS


def _get_openai_client(api_key: Optional[str] = None):
    """Lazy import to avoid requiring openai when not used."""
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
    return OpenAI(api_key=key)


def _get_gemini_client(api_key: Optional[str] = None):
    """Lazy import to avoid requiring google-genai when not used."""
    from google import genai

    key = (
        api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise ValueError(
            "Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY or pass api_key."
        )
    return genai.Client(api_key=key)


def _get_anthropic_client(api_key: Optional[str] = None):
    """Lazy import to avoid requiring anthropic when not used."""
    from anthropic import Anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
        )
    return Anthropic(api_key=key)


def submit_batch(
    provider: str,
    input_path: str | Path,
    *,
    api_key: Optional[str] = None,
    endpoint: str = "responses",
    model_id: Optional[str] = None,
    completion_window: str = "24h",
) -> str:
    """Upload input file and create a batch job. Returns batch ID or job name.

    Parameters
    ----------
    provider : str
        "openai", "gemini", or "anthropic".
    input_path : str or Path
        Path to JSONL batch input file.
    api_key : str, optional
        API key. Defaults to OPENAI_API_KEY or GEMINI_API_KEY from env.
    endpoint : str
        For OpenAI: "responses" or "completions".
    model_id : str, optional
        For Gemini: model name (e.g., "gemini-2.5-flash"). Ignored for OpenAI.
    completion_window : str
        For OpenAI: "24h" or "24 hours". Ignored for Gemini.

    Returns
    -------
    str
        Batch ID (OpenAI) or job name (Gemini) for later retrieval.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if provider == "openai":
        client = _get_openai_client(api_key)
        with open(input_path, "rb") as f:
            file_resp = client.files.create(file=f, purpose="batch")
        ep = ENDPOINT_URLS.get(endpoint, "/v1/responses")
        batch = client.batches.create(
            input_file_id=file_resp.id,
            endpoint=ep,
            completion_window=completion_window,
        )
        return batch.id

    if provider == "gemini":
        client = _get_gemini_client(api_key)
        model = model_id or "gemini-2.5-flash"
        try:
            from google.genai import types

            uploaded = client.files.upload(
                file=str(input_path),
                config=types.UploadFileConfig(
                    display_name=input_path.stem,
                    mime_type="text/plain",  # workaround for jsonl KeyError in some SDK versions
                ),
            )
        except ImportError:
            uploaded = client.files.upload(file=str(input_path))
        job = client.batches.create(
            model=model,
            src=uploaded.name,
            config={"display_name": input_path.stem},
        )
        return job.name

    if provider == "anthropic":
        client = _get_anthropic_client(api_key)

        def _request_lines() -> Any:
            with open(input_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

        batch = client.messages.batches.create(requests=_request_lines())
        return batch.id

    raise ValueError(
        f"Unknown provider '{provider}'. Use 'openai', 'gemini', or 'anthropic'."
    )


def wait_for_batch(
    provider: str,
    batch_id: str,
    *,
    api_key: Optional[str] = None,
    poll_interval: int = 60,
) -> str:
    """Poll until batch is complete. Returns final status.

    Parameters
    ----------
    provider : str
        "openai", "gemini", or "anthropic".
    batch_id : str
        Batch ID from submit_batch.
    api_key : str, optional
        API key for auth.
    poll_interval : int
        Seconds between polls.

    Returns
    -------
    str
        Final status string for the provider (e.g. ``completed``, ``JOB_STATE_SUCCEEDED``, ``ended``).
        Raises on failure.
    """
    if provider == "openai":
        client = _get_openai_client(api_key)
        while True:
            batch = client.batches.retrieve(batch_id)
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                if batch.status != "completed":
                    raise RuntimeError(
                        f"Batch {batch_id} ended with status: {batch.status}"
                    )
                return batch.status
            time.sleep(poll_interval)

    if provider == "gemini":
        client = _get_gemini_client(api_key)
        while True:
            job = client.batches.get(name=batch_id)
            state = job.state.name if hasattr(job.state, "name") else str(job.state)
            if state in (
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
            ):
                if state != "JOB_STATE_SUCCEEDED":
                    err = getattr(job, "error", None) or ""
                    raise RuntimeError(
                        f"Batch {batch_id} ended with state: {state}. {err}"
                    )
                return state
            time.sleep(poll_interval)

    if provider == "anthropic":
        client = _get_anthropic_client(api_key)
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return batch.processing_status
            if batch.processing_status == "canceling":
                raise RuntimeError(f"Batch {batch_id} is canceling.")
            time.sleep(poll_interval)

    raise ValueError(f"Unknown provider '{provider}'.")


def download_batch_output(
    provider: str,
    batch_id: str,
    output_path: str | Path,
    *,
    api_key: Optional[str] = None,
) -> Path:
    """Download batch results to a local file.

    Parameters
    ----------
    provider : str
        "openai", "gemini", or "anthropic".
    batch_id : str
        Batch ID from submit_batch.
    output_path : str or Path
        Where to save the JSONL output.
    api_key : str, optional
        API key for auth.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if provider == "openai":
        client = _get_openai_client(api_key)
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed" or not batch.output_file_id:
            raise RuntimeError(
                f"Batch {batch_id} has no output (status={batch.status})"
            )
        content = client.files.content(batch.output_file_id)
        output_path.write_bytes(content.read())
        return output_path

    if provider == "gemini":
        client = _get_gemini_client(api_key)
        job = client.batches.get(name=batch_id)
        dest = getattr(job, "dest", None)
        file_name = getattr(dest, "file_name", None) if dest else None
        if not file_name:
            raise RuntimeError(
                f"Batch {batch_id} has no output file (dest.file_name is None). "
                "See https://github.com/googleapis/python-genai/issues/1527"
            )
        data = client.files.download(file=file_name)
        output_path.write_bytes(
            data if isinstance(data, bytes) else data.encode("utf-8")
        )
        return output_path

    if provider == "anthropic":
        client = _get_anthropic_client(api_key)
        decoder = client.messages.batches.results(batch_id)
        with open(output_path, "w", encoding="utf-8") as out:
            for item in decoder:
                row = (
                    item.model_dump(mode="json")
                    if hasattr(item, "model_dump")
                    else item
                )
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        return output_path

    raise ValueError(f"Unknown provider '{provider}'.")


def run_batch(
    provider: str,
    input_path: str | Path,
    output_path: str | Path,
    *,
    api_key: Optional[str] = None,
    endpoint: str = "responses",
    model_id: Optional[str] = None,
    poll_interval: int = 60,
) -> Path:
    """Submit batch, wait for completion, download output. One-shot pipeline.

    Parameters
    ----------
    provider : str
        "openai", "gemini", or "anthropic".
    input_path : str or Path
        Path to JSONL batch input (from build_batch_file).
    output_path : str or Path
        Where to save the JSONL output.
    api_key : str, optional
        API key. Defaults to env.
    endpoint : str
        For OpenAI: "responses" or "completions".
    model_id : str, optional
        For Gemini: model name.
    poll_interval : int
        Seconds between status polls.

    Returns
    -------
    Path
        Path to the written output file.
    """
    batch_id = submit_batch(
        provider,
        input_path,
        api_key=api_key,
        endpoint=endpoint,
        model_id=model_id,
    )
    wait_for_batch(provider, batch_id, api_key=api_key, poll_interval=poll_interval)
    return download_batch_output(provider, batch_id, output_path, api_key=api_key)
