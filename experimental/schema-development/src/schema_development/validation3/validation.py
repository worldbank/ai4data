"""Run the Appendix H comparison, historically named Validation 3."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from tqdm.auto import tqdm

from schema_development.paths import ROOT


_PROMPT_DIR = Path(__file__).parent / "prompts"
_SCHEMA_PATH = ROOT / "artifacts/appendix_h/evaluated_schema_v1.2.schema.json"
_SNAPSHOT_PATTERN = re.compile(
    r"^(?P<document_id>.+)_(?P<artifact_type>figure|table)_"
    r"(?P<artifact_index>\d{3})\.png$"
)
_SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class CandidateGap(BaseModel):
    """Describe one critical or possibly critical v1.2 coverage gap.

    Parameters
    ----------
    gap_id : str
        Identifier unique within the snapshot assessment.
    gap_status : Literal["critical", "possible"]
        Strength of the model's gap assessment.
    missing_metadata_concept : str
        Reusable snapshot metadata that may not be adequately covered.
    proposed_field_name : str
        Suggested snake-case label for human review, not an accepted field.
    evidence : str
        Concise evidence from the supplied snapshot and document context.
    why_snapshot_metadata : str
        Reason the information describes the snapshot rather than only its document.
    closest_schema_paths : list[str]
        Exact v1.2 paths that most closely represent the information.
    why_existing_fields_may_be_insufficient : str
        Possible semantic loss when using the closest existing paths.
    material_impact : Literal["interpretability", "discoverability", "both"]
        Area materially affected if the information is not covered.
    material_consequence : str
        Concrete impairment caused by omission or inadequate representation.
    uncertainty_note : str | None
        Remaining evidential or materiality uncertainty, if any.
    """

    model_config = ConfigDict(extra="forbid")

    gap_id: str
    gap_status: Literal["critical", "possible"]
    missing_metadata_concept: str
    proposed_field_name: str
    evidence: str
    why_snapshot_metadata: str
    closest_schema_paths: list[str]
    why_existing_fields_may_be_insufficient: str
    material_impact: Literal["interpretability", "discoverability", "both"]
    material_consequence: str
    uncertainty_note: str | None


class SchemaValidationResult(BaseModel):
    """Represent one confirmatory Schema v1.2 coverage assessment.

    Parameters
    ----------
    coverage_assessment : Literal["no_critical_gap_found", "critical_gap_found", "possible_gap"]
        Snapshot-level coverage result.
    critical_or_possible_gaps : list[CandidateGap]
        Evidence-backed gaps for later human adjudication.
    """

    model_config = ConfigDict(extra="forbid")

    coverage_assessment: Literal[
        "no_critical_gap_found", "critical_gap_found", "possible_gap"
    ]
    critical_or_possible_gaps: list[CandidateGap]


@dataclass(frozen=True)
class RunSummary:
    """Summarize one resumable validation run.

    Attributes
    ----------
    discovered : int
        Number of selected PNG snapshots.
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
    results_path: str | Path,
    errors_path: str | Path,
    config_path: str | Path,
    snapshot_file_names: set[str] | None = None,
    excluded_schema_fields: tuple[str, ...] = (),
    sleep_seconds: float = 0.2,
    client: Any | None = None,
) -> RunSummary:
    """Validate generated Schema v1.2 against selected snapshots.

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
    results_path : str | Path
        JSONL file receiving successful, fully parsed assessments.
    errors_path : str | Path
        JSONL file receiving failed attempts.
    config_path : str | Path
        JSON file containing the model and Responses API settings.
    snapshot_file_names : set[str] | None, optional
        Exact filenames to select, or every discovered PNG when omitted.
    excluded_schema_fields : tuple[str, ...], optional
        Root fields removed from a local schema copy for a calibration control.
    sleep_seconds : float, optional
        Delay after each API call. Defaults to 0.2 seconds.
    client : Any | None, optional
        OpenAI-compatible client used for testing. A client using
        ``OPENAI_API_KEY`` is created when omitted.

    Returns
    -------
    RunSummary
        Counts for selected, skipped, successful, and failed snapshots.

    Raises
    ------
    ValueError
        If configuration, schema, prompts, inputs, or an existing result file
        is invalid.
    """
    snapshot_root = Path(snapshots_dir)
    metadata_root = Path(metadata_dir)
    schema_json, schema_paths = _schema_context(excluded_schema_fields)
    if not schema_paths:
        raise ValueError("The generated v1.2 schema does not define any fields.")

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
    snapshots = _select_snapshots(snapshot_root, snapshot_file_names)
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
                schema_json,
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
                            {"type": "input_image", "image_url": image_url},
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
            _validate_assessment(parsed, schema_paths)
            _append_jsonl(
                results_path,
                {
                    **context,
                    "snapshot_path": str(snapshot_path),
                    "metadata_file_name": metadata_path.name,
                    "schema_version": "1.2",
                    "excluded_schema_fields": list(excluded_schema_fields),
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
                    "schema_version": "1.2",
                    "excluded_schema_fields": list(excluded_schema_fields),
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


def _schema_context(
    excluded_schema_fields: tuple[str, ...] = (),
) -> tuple[str, set[str]]:
    """Serialize the frozen v1.2 schema or a local ablation and its paths."""
    schema = json.loads(_frozen_schema_json())
    if excluded_schema_fields:
        properties = schema.get("properties", {})
        missing = set(excluded_schema_fields) - set(properties)
        if missing:
            raise ValueError(
                f"Cannot exclude unknown schema fields: {sorted(missing)}."
            )
        for field_name in excluded_schema_fields:
            properties.pop(field_name)
        if "required" in schema:
            schema["required"] = [
                name
                for name in schema["required"]
                if name not in excluded_schema_fields
            ]
        schema.pop("description", None)
        _remove_disclosing_descriptions(schema, set(excluded_schema_fields))
        _prune_unused_definitions(schema)
    return (
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        _schema_field_paths(schema),
    )


@cache
def _frozen_schema_json() -> str:
    """Read the checked-in schema used by the completed Validation 3 run."""
    return _SCHEMA_PATH.read_text(encoding="utf-8")


def _schema_field_paths(schema: dict[str, Any]) -> set[str]:
    """Return exact dot paths for properties in a generated JSON Schema."""
    definitions = schema.get("$defs", {})
    paths: set[str] = set()

    def walk(node: dict[str, Any], prefix: str, seen: frozenset[str]) -> None:
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference.rsplit("/", 1)[-1]
            if name not in seen and isinstance(definitions.get(name), dict):
                walk(definitions[name], prefix, seen | {name})
        for keyword in ("anyOf", "oneOf", "allOf"):
            for option in node.get(keyword, []):
                if isinstance(option, dict):
                    walk(option, prefix, seen)
        items = node.get("items")
        if isinstance(items, dict):
            item_path = prefix + "[]"
            paths.add(item_path)
            walk(items, item_path, seen)
        for name, child in node.get("properties", {}).items():
            path = f"{prefix}.{name}" if prefix else name
            paths.add(path)
            if isinstance(child, dict):
                walk(child, path, seen)

    walk(schema, "", frozenset())
    return paths


def _remove_disclosing_descriptions(
    value: dict[str, Any] | list[Any], excluded_fields: set[str]
) -> None:
    """Remove description paragraphs that name an ablated schema path."""
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                _remove_disclosing_descriptions(item, excluded_fields)
        return
    description = value.get("description")
    if isinstance(description, str):
        paragraphs = description.split("\n\n")
        retained = [
            paragraph
            for paragraph in paragraphs
            if not any(f"`{name}`" in paragraph for name in excluded_fields)
        ]
        if retained:
            value["description"] = "\n\n".join(retained)
        else:
            value.pop("description")
    for item in value.values():
        if isinstance(item, (dict, list)):
            _remove_disclosing_descriptions(item, excluded_fields)


def _prune_unused_definitions(schema: dict[str, Any]) -> None:
    """Remove definitions no longer reachable after a root-field ablation."""
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return
    reachable: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.rsplit("/", 1)[-1]
                if name not in reachable and name in definitions:
                    reachable.add(name)
                    collect(definitions[name])
            for key, item in value.items():
                if key != "$defs":
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(schema)
    schema["$defs"] = {
        name: definition
        for name, definition in definitions.items()
        if name in reachable
    }


def _select_snapshots(
    snapshot_root: Path, snapshot_file_names: set[str] | None
) -> list[Path]:
    """Discover snapshots and apply an optional exact filename selection."""
    discovered = sorted(snapshot_root.glob("*/*/*.png"))
    if not discovered:
        raise ValueError(
            f"No snapshots found under {snapshot_root}/<source>/<type>/*.png."
        )
    names = [path.name for path in discovered]
    if len(names) != len(set(names)):
        raise ValueError(
            "Snapshot filenames must be unique across the input directory."
        )
    if snapshot_file_names is None:
        return discovered
    unknown = snapshot_file_names - set(names)
    if unknown:
        raise ValueError(f"Requested snapshots were not found: {sorted(unknown)}.")
    selected = [path for path in discovered if path.name in snapshot_file_names]
    if not selected:
        raise ValueError("The snapshot filename selection is empty.")
    return selected


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
    schema_json: str,
    document_metadata: dict[str, Any],
    context: dict[str, Any],
) -> str:
    """Render schema, metadata, and snapshot context into the user prompt."""
    replacements = {
        "{{SCHEMA_JSON}}": schema_json,
        "{{SNAPSHOT_FILE_NAME}}": context["snapshot_file_name"],
        "{{SOURCE}}": context["source"],
        "{{ARTIFACT_TYPE}}": context["artifact_type"],
        "{{DOCUMENT_METADATA}}": json.dumps(
            document_metadata, ensure_ascii=False, indent=2
        ),
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _validate_assessment(
    assessment: SchemaValidationResult, schema_paths: set[str]
) -> None:
    """Check field proposals, exact paths, and output consistency."""
    gaps = assessment.critical_or_possible_gaps
    gap_ids = [gap.gap_id for gap in gaps]
    if len(gap_ids) != len(set(gap_ids)):
        raise ValueError("Gap IDs must be unique within one assessment.")

    proposed_names = [gap.proposed_field_name for gap in gaps]
    if len(proposed_names) != len(set(proposed_names)):
        raise ValueError("Proposed field names must be unique within one assessment.")
    schema_field_names = {
        path.rsplit(".", 1)[-1].removesuffix("[]") for path in schema_paths
    }
    for gap in gaps:
        if _SNAKE_CASE_PATTERN.fullmatch(gap.proposed_field_name) is None:
            raise ValueError(
                f"Proposed field {gap.proposed_field_name!r} is not snake_case."
            )
        if gap.proposed_field_name in schema_field_names:
            raise ValueError(
                f"Proposed field {gap.proposed_field_name!r} exists in v1.2."
            )
        unknown = set(gap.closest_schema_paths) - schema_paths
        if unknown:
            raise ValueError(f"Unknown v1.2 schema paths: {sorted(unknown)}.")

    coverage = assessment.coverage_assessment
    if coverage == "no_critical_gap_found" and gaps:
        raise ValueError("A no-gap result cannot include candidate gaps.")
    if coverage == "critical_gap_found" and not any(
        gap.gap_status == "critical" for gap in gaps
    ):
        raise ValueError("A critical-gap result requires a critical candidate.")
    if coverage == "possible_gap":
        if not gaps or any(gap.gap_status == "critical" for gap in gaps):
            raise ValueError("A possible-gap result requires possible candidates only.")


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
                file_name = record["snapshot_file_name"]
                if file_name in completed:
                    raise ValueError(f"duplicate snapshot {file_name!r}")
                completed.add(file_name)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
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
    """Create an OpenAI client with automatic retries disabled."""
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
    """Extract model text retained by a Pydantic validation error."""
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
