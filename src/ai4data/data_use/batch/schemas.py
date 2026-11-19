"""Record shapes of the batch pipeline.

``ExtractionRecord``, ``EntitySpan`` and ``CheckpointRecord`` are strict and
match ``ai4data-playground/data_use/schemas.py`` so its downstream scripts keep
working. Everything this pipeline adds (LFM output, document-level merge) goes
into separate files that join on ``corpus_id``.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

Label = Literal["NAMED_DATA", "DESCRIPTIVE_DATA", "VAGUE_DATA", "DATA_MENTION"]


class EntitySpan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str
    label: Label
    score: float
    start: int
    end: int


class ExtractionRecord(BaseModel):
    """One row of ``extractions.jsonl`` (one per chunk). Offsets are chunk-relative."""

    model_config = ConfigDict(extra="forbid", strict=True)
    corpus_id: str
    config: str
    pdf_url: str
    title: str
    page: int
    chunk: int
    input_text: str
    has_data_score: float
    entities: List[EntitySpan]


class CheckpointRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    corpus_id: str
    status: Literal["done", "failed"]
    error: Optional[str] = None


class ManifestRecord(BaseModel):
    """One row of ``parsed/<config>.jsonl``: the document and how it was parsed."""

    corpus_id: str
    config: str
    pdf_url: str
    title: str
    file: str  # path of the parsed .md
    pages: int  # pages with text in the .md
    n_pdf_pages: int  # pages in the PDF
    source_path: Optional[str] = None  # local PDF path, if any
    sha256: str  # of the PDF bytes (or of the URL when not hashable)
    size_bytes: Optional[int] = None
    parser: Dict[str, str] = {}  # library versions
    parsed_at: str
    metadata: Dict[str, Any] = {}  # extra columns from the input manifest


class EnrichmentRecord(BaseModel):
    """One row of ``enrichment.jsonl``: LFM output for one named mention.

    Joins to ``extractions.jsonl`` on ``(corpus_id, page, chunk, start, end)``; the
    passage given to the LFM model is that chunk's ``input_text``.
    """

    corpus_id: str
    page: int
    chunk: int
    start: int
    end: int
    text: str
    usage_context: str
    is_used: str
    usage_action: str
    usage_summary: str
    purpose_action: str
    theme_tags: List[str]
    source_attributes: Dict[str, Optional[str]]
    raw_usage: Optional[Dict[str, Any]] = None  # model JSON, None = no valid JSON
    raw_attributes: Optional[Dict[str, Any]] = None
