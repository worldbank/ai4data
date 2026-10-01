"""Run held-out coverage validation for Data Snapshot Metadata Schema v1.1."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm


_PROMPT_DIR = Path(__file__).parent / "prompts"
_SNAPSHOT_PATTERN = re.compile(
    r"^(?P<document_id>.+)_(?P<artifact_type>figure|table)_"
    r"(?P<artifact_index>\d{3})\.png$"
)
_FIELD_PATTERN = re.compile(r"^### ([a-z][a-z0-9_]*)$", re.MULTILINE)


class ValidationObservation(BaseModel):
    """Describe one metadata concept and its fit with the frozen schema.

    Attributes
    ----------
    observation_id : str
        Identifier unique within the snapshot assessment.
    metadata_concept : str
        Reusable metadata concept evidenced by the inputs.
    evidence : str
        Concise explicit evidence supporting the observation.
    evidence_source : Literal["snapshot", "document", "both"]
        Input source containing the cited evidence.
    closest_schema_fields : list[str]
        Exact v1.1 field names that most closely represent the concept.
    fit_status : Literal["covered", "weak_fit", "no_fit", "out_of_scope", "uncertain"]
        Coverage classification for the observation.
    fit_rationale : str
        Explanation of the assigned coverage classification.
    """

    observation_id: str
    metadata_concept: str
    evidence: str
    evidence_source: Literal["snapshot", "document", "both"]
    closest_schema_fields: list[str]
    fit_status: Literal["covered", "weak_fit", "no_fit", "out_of_scope", "uncertain"]
    fit_rationale: str


class CandidateField(BaseModel):
    """Propose an evidence-backed field for later human adjudication.

    Attributes
    ----------
    proposed_name : str
        Reusable candidate field name in snake_case.
    definition : str
        Proposed definition of the candidate field.
    supporting_observation_ids : list[str]
        Observation identifiers supporting the proposal.
    why_existing_fields_are_insufficient : str
        Explanation of the material gap in the existing schema.
    operational_value : str
        Value of the field for describing and using data snapshots.
    source_level : Literal["snapshot", "document", "both"]
        Level at which evidence for the field is available.
    """

    proposed_name: str
    definition: str
    supporting_observation_ids: list[str]
    why_existing_fields_are_insufficient: str
    operational_value: str
    source_level: Literal["snapshot", "document", "both"]


class SchemaValidationResult(BaseModel):
    """Structured validation assessment for one held-out snapshot.

    Attributes
    ----------
    observations : list[ValidationObservation]
        Metadata observations evaluated against the frozen schema.
    candidate_new_fields : list[CandidateField]
        Evidence-backed field proposals requiring human adjudication.
    """

    observations: list[ValidationObservation]
    candidate_new_fields: list[CandidateField]


@dataclass(frozen=True)
class RunSummary:
    """Summarize one resumable validation run.

    Attributes
    ----------
    discovered : int
        Number of PNG snapshots discovered in the input layout.
    skipped : int
        Number skipped because a successful result already exists.
    succeeded : int
        Number successfully processed during this invocation.
    failed : int
        Number written to the error JSONL during this invocation.
    """

    discovered: int
    skipped: int
    succeeded: int
    failed: int


def run_validation(
    snapshots_dir: str | Path,
    metadata_dir: str | Path,
    schema_path: str | Path,
    results_path: str | Path,
    errors_path: str | Path,
    config_path: str | Path,
    sleep_seconds: float = 0.2,
    client: Any | None = None,
) -> RunSummary:
    """Validate a frozen schema against snapshots in a source/type layout.

    ``snapshots_dir`` must contain ``<source>/<figure|table>/*.png``. Metadata
    must be stored under ``metadata_dir/<source>`` as either
    ``<document_id>.json`` or ``<document_id>_metadata.json``. Successful
    snapshot filenames already present in ``results_path`` are skipped, while
    errors remain eligible for retry.

    Parameters
    ----------
    snapshots_dir : str | Path
        Root containing source and artifact-type subdirectories.
    metadata_dir : str | Path
        Root containing one metadata subdirectory per source corpus.
    schema_path : str | Path
        Path to the canonical v1.1 Markdown schema.
    results_path : str | Path
        JSONL file receiving successful, fully parsed assessments.
    errors_path : str | Path
        JSONL file receiving failed attempts.
    config_path : str | Path
        JSON file containing the model and Responses API settings.
    sleep_seconds : float, optional
        Delay after each API call. Defaults to 0.2 seconds.
    client : Any | None, optional
        OpenAI-compatible client used for testing. A client using
        ``OPENAI_API_KEY`` is created when omitted.

    Returns
    -------
    RunSummary
        Counts for discovered, skipped, successful, and failed snapshots.

    Raises
    ------
    ValueError
        If configuration, schema, prompts, or an existing result file is
        invalid.
    """
    snapshot_root = Path(snapshots_dir)
    metadata_root = Path(metadata_dir)
    schema_markdown = Path(schema_path).read_text(encoding="utf-8")
    schema_fields = set(_FIELD_PATTERN.findall(schema_markdown))
    if not schema_fields:
        raise ValueError("The schema Markdown does not define any level-three fields.")

    config = _load_json_object(config_path, "Validation config")
    model = config.pop("model", None)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Validation config requires a non-empty 'model' string.")
    forbidden = {"input", "text", "text_format"} & set(config)
    if forbidden:
        raise ValueError(f"Config cannot set pipeline-owned keys: {sorted(forbidden)}.")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds cannot be negative.")

    system_prompt = (_PROMPT_DIR / "system.md").read_text(encoding="utf-8")
    user_template = (_PROMPT_DIR / "user.md").read_text(encoding="utf-8")
    snapshots = sorted(snapshot_root.glob("*/*/*.png"))
    if not snapshots:
        raise ValueError(
            f"No snapshots found under {snapshot_root}/<source>/<type>/*.png."
        )
    if Path(results_path).resolve() == Path(errors_path).resolve():
        raise ValueError("results_path and errors_path must be different files.")
    completed = _load_completed_filenames(results_path)
    api_client = client if client is not None else _create_openai_client()

    skipped = succeeded = failed = 0
    for snapshot_path in tqdm(snapshots, desc="Validating snapshots", unit="snapshot"):
        if snapshot_path.name in completed:
            skipped += 1
            continue

        context: dict[str, Any] = {
            "snapshot_file_name": snapshot_path.name,
            "source": snapshot_path.parent.parent.name,
            "artifact_type": snapshot_path.parent.name,
            "artifact_index": None,
            "source_document_id": None,
        }
        stage = "input"
        started_at: float | None = None
        response: Any | None = None
        http_response: Any | None = None
        metadata_path: Path | None = None
        try:
            context = _snapshot_context(snapshot_path, snapshot_root)
            metadata_path = _find_metadata_path(
                metadata_root, context["source"], context["source_document_id"]
            )
            document_metadata = _load_json_object(metadata_path, "Document metadata")
            user_prompt = _render_user_prompt(
                user_template,
                schema_markdown,
                document_metadata,
                context,
            )
            image_url = _image_data_url(snapshot_path)
            stage = "api"
            started_at = time.perf_counter()
            request = {
                "model": model,
                "text_format": SchemaValidationResult,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {
                                "type": "input_image",
                                "image_url": image_url,
                            },
                        ],
                    },
                ],
                **config,
            }
            raw_responses = getattr(api_client.responses, "with_raw_response", None)
            if raw_responses is None:
                response = api_client.responses.parse(**request)
            else:
                http_response = raw_responses.parse(**request)
                stage = "parse"
                response = http_response.parse()
            elapsed_seconds = time.perf_counter() - started_at
            stage = "parse"
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                raise ValueError("Structured output was missing or incomplete.")
            stage = "validation"
            _validate_assessment(parsed, schema_fields)
            _append_jsonl(
                results_path,
                {
                    **context,
                    "snapshot_path": str(snapshot_path),
                    "metadata_file_name": metadata_path.name,
                    "model": model,
                    "request_config": config,
                    "response_id": getattr(response, "id", None),
                    "api_status": getattr(response, "status", None),
                    "elapsed_seconds": elapsed_seconds,
                    "usage": _serialize(getattr(response, "usage", None)),
                    "raw_response": _serialize(response),
                    "raw_output": getattr(response, "output_text", None),
                    "parsed_output": parsed.model_dump(mode="json"),
                },
            )
            completed.add(snapshot_path.name)
            succeeded += 1
        except Exception as exc:
            if isinstance(exc, ValidationError):
                stage = "parse"
            elapsed_seconds = (
                time.perf_counter() - started_at if started_at is not None else None
            )
            raw_response = _read_raw_response(http_response)
            error_response = response if response is not None else raw_response
            _append_jsonl(
                errors_path,
                {
                    **context,
                    "snapshot_path": str(snapshot_path),
                    "metadata_file_name": metadata_path.name if metadata_path else None,
                    "model": model,
                    "request_config": config,
                    "response_id": _response_value(error_response, "id"),
                    "api_status": _response_value(error_response, "status"),
                    "elapsed_seconds": elapsed_seconds,
                    "usage": _serialize(_response_value(error_response, "usage")),
                    "raw_response": _serialize(error_response),
                    "raw_output": getattr(response, "output_text", None)
                    or _validation_error_input(exc),
                    "error_stage": stage,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            failed += 1
        if started_at is not None and sleep_seconds:
            time.sleep(sleep_seconds)

    return RunSummary(len(snapshots), skipped, succeeded, failed)


def _snapshot_context(snapshot_path: Path, snapshot_root: Path) -> dict[str, Any]:
    """Parse provenance and artifact details from a snapshot path."""
    relative = snapshot_path.relative_to(snapshot_root)
    if len(relative.parts) != 3:
        raise ValueError(f"Unexpected snapshot layout: {relative}")
    source, type_directory, _ = relative.parts
    match = _SNAPSHOT_PATTERN.fullmatch(snapshot_path.name)
    if match is None:
        raise ValueError(f"Invalid snapshot filename: {snapshot_path.name}")
    values = match.groupdict()
    if values["artifact_type"] != type_directory:
        raise ValueError(
            f"Filename type {values['artifact_type']!r} does not match "
            f"directory {type_directory!r}: {snapshot_path.name}"
        )
    return {
        "snapshot_file_name": snapshot_path.name,
        "source": source,
        "artifact_type": values["artifact_type"],
        "artifact_index": int(values["artifact_index"]),
        "source_document_id": values["document_id"],
    }


def _find_metadata_path(metadata_root: Path, source: str, document_id: str) -> Path:
    """Find exact or UNHCR-style source-document metadata."""
    candidates = [
        metadata_root / source / f"{document_id}.json",
        metadata_root / source / f"{document_id}_metadata.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No metadata found for {document_id!r} in {metadata_root / source}."
    )


def _render_user_prompt(
    template: str,
    schema_markdown: str,
    document_metadata: dict[str, Any],
    context: dict[str, Any],
) -> str:
    """Render schema, metadata, and snapshot context into the user prompt."""
    replacements = {
        "{{SNAPSHOT_FILE_NAME}}": context["snapshot_file_name"],
        "{{SOURCE}}": context["source"],
        "{{ARTIFACT_TYPE}}": context["artifact_type"],
        "{{SCHEMA_MARKDOWN}}": schema_markdown,
        "{{DOCUMENT_METADATA}}": json.dumps(
            document_metadata, ensure_ascii=False, indent=2
        ),
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _validate_assessment(
    assessment: SchemaValidationResult, schema_fields: set[str]
) -> None:
    """Check cross-references and exact field names in a parsed assessment."""
    observation_ids = [item.observation_id for item in assessment.observations]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("Observation IDs must be unique within one assessment.")

    observations = {item.observation_id: item for item in assessment.observations}
    for observation in assessment.observations:
        unknown = set(observation.closest_schema_fields) - schema_fields
        if unknown:
            raise ValueError(f"Unknown v1.1 fields: {sorted(unknown)}.")
        if observation.fit_status in {"covered", "weak_fit"} and not (
            observation.closest_schema_fields
        ):
            raise ValueError(
                f"{observation.fit_status} observation "
                f"{observation.observation_id!r} requires a closest schema field."
            )

    for candidate in assessment.candidate_new_fields:
        if re.fullmatch(r"[a-z][a-z0-9_]*", candidate.proposed_name) is None:
            raise ValueError(
                f"Candidate field {candidate.proposed_name!r} is not snake_case."
            )
        if candidate.proposed_name in schema_fields:
            raise ValueError(
                f"Candidate field {candidate.proposed_name!r} already exists in v1.1."
            )
        if not candidate.supporting_observation_ids:
            raise ValueError(
                f"Candidate field {candidate.proposed_name!r} has no observations."
            )
        for observation_id in candidate.supporting_observation_ids:
            observation = observations.get(observation_id)
            if observation is None:
                raise ValueError(f"Unknown observation ID: {observation_id!r}.")
            if observation.fit_status not in {"weak_fit", "no_fit"}:
                raise ValueError(
                    f"Candidate field {candidate.proposed_name!r} is supported by "
                    f"a {observation.fit_status!r} observation."
                )

    candidate_names = [
        candidate.proposed_name for candidate in assessment.candidate_new_fields
    ]
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("Candidate field names must be unique within an assessment.")


def _load_completed_filenames(path: str | Path) -> set[str]:
    """Load successful snapshot filenames from an existing result JSONL."""
    result_path = Path(path)
    if not result_path.exists():
        return set()
    completed: set[str] = set()
    with result_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
                completed.add(record["snapshot_file_name"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid result JSONL at line {line_number}: {exc}"
                ) from exc
    return completed


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with Path(path).open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _image_data_url(path: Path) -> str:
    """Encode a PNG snapshot as a base64 data URL."""
    if path.suffix.lower() != ".png":
        raise ValueError(f"Unsupported image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _create_openai_client() -> Any:
    """Create an OpenAI client using the project environment."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key, max_retries=0)


def _serialize(value: Any) -> dict[str, Any] | None:
    """Convert an SDK object to JSON-compatible dictionary data."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", warnings=False)
    if isinstance(value, dict):
        return value
    return {
        key: getattr(value, key)
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if hasattr(value, key)
    }


def _read_raw_response(response: Any | None) -> dict[str, Any] | None:
    """Read a raw SDK response body without raising a secondary error."""
    if response is None:
        return None
    try:
        value = response.json()
    except (AttributeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _response_value(response: Any | None, name: str) -> Any | None:
    """Read a named value from an SDK model or raw response dictionary."""
    if isinstance(response, dict):
        return response.get(name)
    return getattr(response, name, None)


def _validation_error_input(error: Exception) -> str | None:
    """Extract the model text retained by a Pydantic validation error."""
    if not isinstance(error, ValidationError):
        return None
    for detail in error.errors(include_input=True):
        input_value = detail.get("input")
        if isinstance(input_value, str):
            return input_value
    return None


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON object and flush it to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
