"""Dataset mention extractor."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..models.model_manager import ModelManager
from ..schemas.dataset_schema_v2 import DatasetSchema
from ..utils.document_parser import DocumentParser
from ..utils.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)

# Heuristic filter patterns for common false positives
_TABLE_FIGURE_RE = re.compile(
    r"^(table|figure|panel|annex|appendix|box|chart|graph|map|diagram)\s+[\w\s.]+$",
    re.IGNORECASE,
)
_TABLE_FIGURE_PREFIX_RE = re.compile(
    r"^(table|figure|fig|panel|annex|appendix|box|chart|graph|map|diagram)s?\s+[a-z0-9_\-\.\/]+$",
    re.IGNORECASE,
)
_DOCUMENT_SUFFIX_RE = re.compile(
    r"\b(working paper|discussion paper|policy paper|technical note|policy note|briefing note|"
    r"status report|assessment report|fiduciary assessment|systems assessment|capacity assessment|"
    r"management report|audit report|financial report|operations manual|project paper|issues paper|"
    r"resilience plan|supervision plan|evaluation report|monitoring report|progress report|"
    r"management framework|esmf|rpf|rap|cfaa|esia|essa|sesa|fsa|psr|ifr|cpar|cpip|protocol|agreement|agreements)s?\b$",
    re.IGNORECASE,
)
_DOCUMENT_NO_RE = re.compile(
    r"\b(paper|report|document|note|annex|appendix|package)\b\s+(?:no|#)\.?\s*\d+",
    re.IGNORECASE,
)

_CITATION_RE = re.compile(
    r"^[A-Z][a-z\-']+(?:,\s*[A-Z][a-z\-']+)*(?:,?\s+(?:and|&)\s+[A-Z][a-z\-']+)?(?:"
    r"\s+et\s+al\.?(?:,?\s*\(?\d{4}\)?)?"
    r"|,?\s*\(?\d{4}\)?"
    r")$"
)

_SINGLE_WORD_NOISE = frozenset(
    {
        "urban",
        "rural",
        "total",
        "national",
        "regional",
        "local",
        "male",
        "female",
        "results",
        "source",
        "sources",
        # Generic single-word terms
        "survey",
        "surveys",
        "data",
        "dataset",
        "datasets",
        "report",
        "reports",
        "assessment",
        "assessments",
        "statistics",
        "stats",
        "index",
        "indices",
        "indicator",
        "indicators",
        "questionnaire",
        "questionnaires",
        "census",
        "registry",
        "registries",
        "records",
        "database",
        "databases",
        "study",
        "studies",
        "analysis",
        "analyses",
        "information",
        "sample",
        "samples",
        "findings",
        "methodology",
        "variables",
        "variable",
    }
)


class DatasetExtractor:
    """Extract dataset mentions from text or documents."""

    @staticmethod
    def _normalize_input_text(text: str) -> str:
        """DEPRECATED. Use TextNormalizer.normalize_full instead."""
        return TextNormalizer.normalize_full(text)

    def __init__(
        self,
        model_id: Optional[str] = None,
        adapter_id: Optional[str] = None,
        threshold: float = 0.3,
        cache_dir: Optional[str] = None,
    ):
        """Initialize dataset extractor.

        Args:
            model_id: HuggingFace model ID or path to local model.
                     If None, uses default model.
            adapter_id: HuggingFace adapter repo ID to apply after loading the base model.
                       If None, falls back to ModelManager.DEFAULT_ADAPTER_ID.
                       Pass an empty string to skip adapter loading entirely.
            threshold: Default confidence threshold for extraction
            cache_dir: Directory to cache models
        """
        self.model_manager = ModelManager(cache_dir=cache_dir)
        self.model_id = model_id
        self.adapter_id = adapter_id
        self.threshold = threshold
        self._model = None
        self._classifier = None
        # Two cached schemas — one without provenance fields (fast default)
        # and one with them (richer output). Built lazily on first use.
        self._schema_core = None
        self._schema_provenance = None

    @property
    def model(self):
        """Lazy load the GLiNER2 extraction model."""
        if self._model is None:
            self._model = self.model_manager.load(self.model_id, adapter_id=self.adapter_id)
        return self._model

    @property
    def classifier(self):
        """Lazy load the BERT page-relevance classifier.

        Returns a HuggingFace text-classification pipeline. Only loaded the
        first time this property is accessed (i.e. when use_classifier=True).
        """
        if self._classifier is None:
            self._classifier = self.model_manager.load_classifier()
        return self._classifier

    @property
    def schema(self):
        """Lazy-build the core schema (no provenance fields).

        Kept as a property for backward compatibility with code that
        references extractor.schema directly.
        """
        return self._build_schema(extract_provenance=False)

    def _build_schema(self, extract_provenance: bool = False):
        """Build and cache the GLiNER2 schema.

        Args:
            extract_provenance: If True, include provenance fields
                (author, producer, publication_year, etc.) in the schema.
                The two variants are cached independently.

        Returns:
            Configured GLiNER2 schema object
        """
        if extract_provenance:
            if self._schema_provenance is None:
                builder = DatasetSchema(threshold=self.threshold)
                self._schema_provenance = builder.build(self.model, extract_provenance=True)
            return self._schema_provenance
        else:
            if self._schema_core is None:
                builder = DatasetSchema(threshold=self.threshold)
                self._schema_core = builder.build(self.model, extract_provenance=False)
            return self._schema_core

    # =========================================================================
    # Markdown-aware chunking helpers
    # =========================================================================

    def _extract_footnotes(self, text: str) -> Tuple[Dict[int, str], str]:
        """Extract footnote definitions from text, anchored by body [N] references.

        Detects three formats found in pymupdf4llm-extracted PDFs:
        - ``[N] text`` (closed bracket, 91 instances across WB + UNHCR docs)
        - ``[N text``  (open bracket,  362 instances)
        - ``N text``   (bare number, 6,848 instances) — only when matching [N] body ref

        Args:
            text: Input text (typically a full page)

        Returns:
            Tuple of (footnotes_dict, body_text) where footnotes_dict maps
            footnote number -> footnote text, and body_text is the text
            without the footnote section.
        """
        # Step 1: Find all [N] body references
        body_refs: Set[int] = set()
        for m in re.finditer(r"\[(\d+)\]", text):
            try:
                body_refs.add(int(m.group(1)))
            except ValueError:
                continue

        if not body_refs:
            return {}, text

        # Step 2: Find footnote definitions at the bottom of text
        footnotes: Dict[int, str] = {}
        lines = text.split("\n")
        footnote_start_line = None

        # Scan from bottom upward to find where footnotes begin
        for i in range(len(lines) - 1, -1, -1):
            ls = lines[i].strip()
            if not ls:
                continue

            # Match footnote def patterns
            m_closed = re.match(r"^\[(\d+)\]\s+(.*)", ls)
            m_open = re.match(r"^\[(\d+)\s+(.*)", ls)
            m_bare = re.match(r'^(\d{1,3})\s+([A-Z"\'\(].*|http.*|www.*)', ls)

            matched_num = None
            matched_text = None

            if m_closed:
                matched_num = int(m_closed.group(1))
                matched_text = m_closed.group(2)
            elif m_open:
                matched_num = int(m_open.group(1))
                matched_text = m_open.group(2)
            elif m_bare:
                matched_num = int(m_bare.group(1))
                matched_text = m_bare.group(2)

            if matched_num is not None and matched_num in body_refs:
                footnotes[matched_num] = matched_text
                footnote_start_line = i
            elif matched_num is not None and matched_num < 200:
                # Might be a footnote for a ref we didn't detect; still mark position
                footnote_start_line = i
            elif footnote_start_line is not None:
                # We've moved past the footnote section into body text
                break

        if footnote_start_line is not None:
            body = "\n".join(lines[:footnote_start_line])
        else:
            body = text

        return footnotes, body

    def _append_footnotes_to_chunk(self, chunk_text: str, footnotes: Dict[int, str]) -> str:
        """Append referenced footnote definitions to a chunk for model context.

        Scans chunk for [N] body references and appends matching footnote
        definitions after a separator. The appended text is for model input
        only — character offsets still reference the original text.

        Args:
            chunk_text: The chunk text to enrich
            footnotes: Dict mapping footnote number -> footnote text

        Returns:
            Enriched chunk text with appended footnotes (if any match)
        """
        if not footnotes:
            return chunk_text

        # Find [N] refs in this chunk
        refs_in_chunk: Set[int] = set()
        for m in re.finditer(r"\[(\d+)\]", chunk_text):
            try:
                refs_in_chunk.add(int(m.group(1)))
            except ValueError:
                continue

        # Collect matching footnotes
        matching = {n: footnotes[n] for n in sorted(refs_in_chunk) if n in footnotes}
        if not matching:
            return chunk_text

        # Append footnotes after separator
        footnote_section = "\n\n---\n" + "\n".join(f"[{n}] {text}" for n, text in matching.items())
        return chunk_text + footnote_section

    def _detect_table_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """Detect table regions in markdown text.

        Tables are identified by the presence of ``|---|`` separator rows.
        Each table region extends from the first ``|`` row before the separator
        to the last consecutive ``|`` row after it.

        Args:
            text: Input text

        Returns:
            List of (start_char, end_char) tuples for each table region
        """
        lines = text.split("\n")
        tables: List[Tuple[int, int]] = []

        char_pos = 0
        line_starts = []
        for line in lines:
            line_starts.append(char_pos)
            char_pos += len(line) + 1  # +1 for \n

        i = 0
        while i < len(lines):
            # Look for separator row |---|
            if re.match(r"^\s*\|[\-\s\|]+\|", lines[i]):
                # Found separator — find table extent
                # Look backward for first table row
                table_start = i
                for j in range(i - 1, -1, -1):
                    if "|" in lines[j]:
                        table_start = j
                    else:
                        break

                # Look forward for last table row
                table_end = i
                for j in range(i + 1, len(lines)):
                    if "|" in lines[j]:
                        table_end = j
                    else:
                        break

                start_char = line_starts[table_start]
                end_char = line_starts[table_end] + len(lines[table_end])
                tables.append((start_char, end_char))
                i = table_end + 1
            else:
                i += 1

        return tables

    def _find_split_point(
        self,
        text: str,
        target_pos: int,
        window: int = 200,
        table_boundaries: Optional[List[Tuple[int, int]]] = None,
    ) -> int:
        """Find the best split point near a target position.

        Prefers markdown-aware boundaries in priority order:
        1. Bold header (``\\n**``) or markdown header (``\\n#``)
        2. Paragraph break (``\\n\\n``)
        3. Table boundary (before/after a complete table)
        4. Line break (``\\n``)
        5. Fallback to target position

        Args:
            text: Full input text
            target_pos: Desired split position (character index)
            window: How far to search backward from target_pos
            table_boundaries: Pre-computed table regions to avoid splitting

        Returns:
            Best split position (character index)
        """
        search_start = max(0, target_pos - window)
        search_region = text[search_start:target_pos]

        # Priority 1: bold header or markdown header
        for pattern in ["\n**", "\n#"]:
            idx = search_region.rfind(pattern)
            if idx != -1:
                return search_start + idx + 1  # split just before the header line

        # Priority 1.5: Note boundary — keep notes attached to preceding context
        # Don't split between content and its _Note_: or Note: block
        for note_pat in ["\n_Note", "\nNote:"]:
            idx = search_region.rfind(note_pat)
            if idx != -1:
                return search_start + idx + 1  # split just before the note

        # Also check: if the target is inside a note block, don't split it
        # Look ahead to see if we're in a note
        lookahead = text[target_pos : min(target_pos + 100, len(text))]
        if re.match(r"^[^\n]*(?:_Note|Note:)", lookahead):
            # We're near a note — find the note start and split before it
            note_match = re.search(r"\n(?:_Note|Note:)", text[search_start : target_pos + 100])
            if note_match:
                note_pos = search_start + note_match.start()
                if note_pos > search_start:
                    return note_pos + 1

        # Priority 2: paragraph break
        idx = search_region.rfind("\n\n")
        if idx != -1:
            return search_start + idx + 2  # split after the blank line

        # Priority 3: avoid splitting inside a table
        if table_boundaries:
            for t_start, t_end in table_boundaries:
                if t_start < target_pos < t_end:
                    # We're inside a table — split before or after it
                    if t_start > search_start:
                        return t_start
                    elif t_end < len(text):
                        return t_end + 1

        # Priority 4: line break
        idx = search_region.rfind("\n")
        if idx != -1:
            return search_start + idx + 1

        # Priority 5: fallback
        return target_pos

    def _chunk_text(self, text: str, max_tokens: int = 200, overlap: int = 50) -> List[tuple]:
        """Split text into overlapping chunks with markdown-aware boundaries.

        Uses token counting to determine approximate split points, then snaps
        each split to the nearest markdown structural boundary (headers,
        paragraph breaks, table edges). Footnote definitions referenced in
        each chunk are appended for model context.

        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk (default: 200, aligned with
                training data token distribution. DeBERTa-v3 supports up to 512
                but shorter chunks match finetuning data better.)
            overlap: Number of tokens to overlap between chunks

        Returns:
            List of tuples (chunk_text, char_offset) where char_offset is the
            starting character position of the chunk in the original text.
            Note: chunk_text may include appended footnotes that extend beyond
            the offset range (for model context only).
        """
        from gliner2.processor import WhitespaceTokenSplitter

        splitter = WhitespaceTokenSplitter()
        tokens = list(splitter(text, lower=False))

        # If text is short enough, return as-is
        if len(tokens) <= max_tokens:
            return [(text, 0)]

        # Pre-compute footnotes and table boundaries
        footnotes, _ = self._extract_footnotes(text)
        table_boundaries = self._detect_table_boundaries(text)

        chunks = []
        start_token_idx = 0

        while start_token_idx < len(tokens):
            end_token_idx = min(start_token_idx + max_tokens, len(tokens))

            # Get character positions from tokens
            chunk_start_char = tokens[start_token_idx][1]
            raw_end_char = tokens[end_token_idx - 1][2] if end_token_idx > 0 else len(text)

            # Snap end position to markdown-aware boundary (unless last chunk)
            if end_token_idx < len(tokens):
                chunk_end_char = self._find_split_point(
                    text,
                    raw_end_char,
                    window=200,
                    table_boundaries=table_boundaries,
                )
                # Don't let snapping pull us behind the chunk start
                if chunk_end_char <= chunk_start_char:
                    chunk_end_char = raw_end_char
            else:
                chunk_end_char = raw_end_char

            chunk_text = text[chunk_start_char:chunk_end_char]

            # NOTE: Footnote enrichment was intentionally removed.
            # Appending footnote text after the body corrupted character-level
            # spans: the model returned offsets relative to the enriched text
            # but chunk_start_char only covers the body, so any entity detected
            # inside the footnote region produced a wrong absolute position after
            # _adjust_entity_indices. Since span accuracy for the highlight UI is
            # critical, we pass only the body text to the model.
            chunks.append((chunk_text, chunk_start_char))

            # Guarantee minimum forward progress: advance by at least
            # (max_tokens - overlap) tokens, then optionally use the
            # snapped position if it's further ahead
            min_next_token = start_token_idx + max_tokens - overlap
            if min_next_token >= len(tokens):
                break

            # Find token closest to snapped chunk_end_char
            snapped_token_idx = end_token_idx
            for t_idx in range(min_next_token, len(tokens)):
                if tokens[t_idx][1] >= chunk_end_char:
                    snapped_token_idx = t_idx
                    break

            # Use whichever makes more progress, but at least min_next_token
            start_token_idx = max(min_next_token, snapped_token_idx - overlap)

        return chunks

    @staticmethod
    def _get_entities(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entity list from a result dict.

        Supports both legacy ``dataset_mention`` and v21-diversity
        ``data_mention`` structure names.
        """
        if not isinstance(result, dict):
            return []
        return (
            result.get("dataset_mention")
            or result.get("data_mention")
            or result.get("entities")
            or []
        )

    @staticmethod
    def _get_name_field(entity: Dict[str, Any]):
        """Get the name field from an entity, supporting both legacy and v21 names."""
        return entity.get("dataset_name") or entity.get("mention_name")

    def _deduplicate_entities(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter null-name entities and deduplicate same-span entities.

        Removes entities where dataset_name/mention_name is None, then groups
        remaining entities by (name_text, start, end) and keeps the one with
        the most specific taxonomy (named > descriptive > vague), breaking ties
        with the highest confidence.

        Args:
            datasets: List of extracted entity dicts

        Returns:
            Filtered and deduplicated list
        """
        # Step 1: Filter out null name entities
        named = []
        for ds in datasets:
            name = self._get_name_field(ds)
            if name is None:
                continue
            if isinstance(name, dict):
                if name.get("text") is None:
                    continue
            named.append(ds)

        # Priority map for taxonomic specificity
        spec_priority = {"named": 3, "descriptive": 2, "vague": 1}

        # Step 2: Deduplicate same-span entities
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for ds in named:
            name = self._get_name_field(ds)
            if isinstance(name, dict):
                key = (name.get("text"), name.get("start"), name.get("end"))
                conf = name.get("confidence", 0)
            else:
                key = (name, None, None)
                conf = 0

            spec_field = ds.get("specificity_tag", {})
            spec_tag = spec_field.get("text") if isinstance(spec_field, dict) else spec_field
            spec_score = spec_priority.get(spec_tag, 0)

            if key not in grouped:
                grouped[key] = ds
            else:
                existing = grouped[key]
                ex_name = self._get_name_field(existing)
                ex_conf = ex_name.get("confidence", 0) if isinstance(ex_name, dict) else 0

                ex_spec_field = existing.get("specificity_tag", {})
                ex_spec_tag = (
                    ex_spec_field.get("text") if isinstance(ex_spec_field, dict) else ex_spec_field
                )
                ex_spec_score = spec_priority.get(ex_spec_tag, 0)

                if spec_score > ex_spec_score or (spec_score == ex_spec_score and conf > ex_conf):
                    grouped[key] = ds

        # Step 3: Filter overlapping spans (NMS)
        def get_sort_key(ds):
            name_f = self._get_name_field(ds)
            conf = name_f.get("confidence", 0.0) if isinstance(name_f, dict) else 0.0
            text_val = name_f.get("text", "") if isinstance(name_f, dict) else str(name_f or "")

            spec_f = ds.get("specificity_tag", {})
            spec_t = spec_f.get("text") if isinstance(spec_f, dict) else spec_f
            spec_s = spec_priority.get(spec_t, 0)

            return (spec_s, conf, len(text_val))

        sorted_entities = sorted(list(grouped.values()), key=get_sort_key, reverse=True)

        accepted = []
        for ds in sorted_entities:
            name_f = self._get_name_field(ds)
            if not isinstance(name_f, dict):
                accepted.append(ds)
                continue

            start = name_f.get("start")
            end = name_f.get("end")

            if start is None or end is None:
                accepted.append(ds)
                continue

            overlaps = False
            for acc in accepted:
                acc_name = self._get_name_field(acc)
                if isinstance(acc_name, dict):
                    acc_start = acc_name.get("start")
                    acc_end = acc_name.get("end")
                    if acc_start is not None and acc_end is not None:
                        # Check character span overlap
                        if not (end <= acc_start or start >= acc_end):
                            overlaps = True
                            break

            if not overlaps:
                accepted.append(ds)

        # Return accepted entities sorted by start index to maintain original order
        def get_start_idx(ds):
            name_f = self._get_name_field(ds)
            if isinstance(name_f, dict):
                return name_f.get("start") or 0
            return 0

        return sorted(accepted, key=get_start_idx)

    def _merge_chunk_results(self, chunk_results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from multiple chunks, removing duplicates.

        Args:
            chunk_results_list: List of result dicts from each chunk

        Returns:
            Merged result dict with deduplicated entities
        """
        if not chunk_results_list:
            return {"data_mention": []}

        # Collect all entities from all chunks
        all_entities = []
        seen_positions = set()  # (text, start, end) exact dedup
        seen_texts = {}  # text -> (index, confidence) for text-based dedup

        for chunk_result in chunk_results_list:
            if isinstance(chunk_result, dict):
                entities = self._get_entities(chunk_result)

                for entity in entities:
                    if isinstance(entity, dict):
                        # Create a hash for deduplication based on name and position
                        name_field = self._get_name_field(entity)
                        if isinstance(name_field, dict):
                            name_text = name_field.get("text")
                            name_start = name_field.get("start")
                            name_end = name_field.get("end")
                            name_conf = name_field.get("confidence", 0.0)
                        else:
                            name_text = name_field
                            name_start = None
                            name_end = None
                            name_conf = 0.0

                        # Exact position dedup
                        pos_key = (name_text, name_start, name_end)
                        if pos_key in seen_positions and not all(v is None for v in pos_key):
                            continue

                        # Text-based dedup: same mention text from
                        # overlapping chunks gets different adjusted
                        # positions.  Keep the higher-confidence value but
                        # preserve the FIRST occurrence's spans (they are
                        # already adjusted to the correct page offset).
                        # Replacing the whole entity would overwrite the spans
                        # with those from the later chunk's offset, causing
                        # highlights to land on the wrong word.
                        if name_text and name_text in seen_texts:
                            prev_idx, prev_conf = seen_texts[name_text]
                            if name_conf > prev_conf:
                                # Update confidence on the stored entity in-place
                                stored = all_entities[prev_idx]
                                stored_name = self._get_name_field(stored)
                                if isinstance(stored_name, dict):
                                    stored_name["confidence"] = name_conf
                                seen_texts[name_text] = (prev_idx, name_conf)
                            # Either way, skip adding a new entry
                            seen_positions.add(pos_key)
                            continue

                        idx = len(all_entities)
                        all_entities.append(entity)
                        seen_positions.add(pos_key)
                        if name_text:
                            seen_texts[name_text] = (idx, name_conf)

        return {"data_mention": all_entities}

    def _adjust_entity_indices(self, entity: Dict[str, Any], offset: int, seen_ids: set) -> None:
        """Adjust start/end indices in entity fields by adding the chunk offset.

        Args:
            entity: Entity dict with potential start/end fields
            offset: Character offset to add to all indices
            seen_ids: Set of dictionary IDs already adjusted
        """
        # Fields that may contain start/end indices (spans from GLiNER2).
        # Covers both legacy (dataset_name, data_type) and v21-diversity
        # (mention_name) field names, plus all provenance fields.
        fields_with_indices = [
            "dataset_name",
            "mention_name",
            "description",
            "acronym",
            "author",
            "producer",
            "publication_year",
            "reference_year",
            "reference_population",
            "data_type",
            "geography",
        ]

        for field in fields_with_indices:
            if field in entity and isinstance(entity[field], dict):
                field_data = entity[field]
                if id(field_data) in seen_ids:
                    continue
                seen_ids.add(id(field_data))
                if "start" in field_data and isinstance(field_data["start"], int):
                    field_data["start"] += offset
                if "end" in field_data and isinstance(field_data["end"], int):
                    field_data["end"] += offset

    def extract_from_text(
        self,
        text: str,
        include_confidence: bool = False,
        custom_schema: Optional[Any] = None,
        exclude_non_datasets: bool = True,
        dataset_threshold: Optional[float] = None,
        max_tokens: int = 200,
        enable_chunking: bool = True,
        use_classifier: bool = False,
        model_id: Optional[str] = None,
        apply_heuristics: bool = False,
        normalize_text: bool = True,
        extract_provenance: bool = False,
        verbose: bool = False,
        _page_label: Optional[str] = None,
        exclude_na_usage: bool = False,
    ) -> Dict[str, Any]:
        """Extract dataset mentions from text.

        Args:
            text: Input text to extract from
            include_confidence: Whether to include confidence scores
            custom_schema: Custom schema to use instead of default
            exclude_non_datasets: If True, filter out entries tagged as
                'non-dataset'. Safety net for pre-v2 models; v2-trained
                models never output this tag.
            dataset_threshold: Optional confidence threshold for mention_name field
            max_tokens: Maximum tokens per chunk (default: 200, aligned with
                training data token distribution)
            enable_chunking: Whether to split long text into chunks (default: True)
            use_classifier: Whether to use two-stage pre-filtering classifier
                (is_english then BERT) before running GLiNER2 (default: False)
            model_id: Optional model ID override for this call (unused by default extractor)
            apply_heuristics: If True, apply heuristic filters to remove likely
                false positives such as table/figure labels (default: False)
            normalize_text: If True, normalize input text before extraction by
                fixing hyphenated line breaks and collapsing excessive whitespace.
                Useful for pymupdf4llm markdown outputs (default: False)
            extract_provenance: If True, also extract provenance fields
                (author, producer, publication_year, reference_year,
                reference_population, geography, description, acronym).
                Increases inference latency. Defaults to False.
            verbose: If True, print skip messages and progress to stdout (default: False)
            exclude_na_usage: If True, drop mentions where is_used could not be
                determined (value is None, empty, or 'na'). Precision-oriented;
                may suppress genuinely ambiguous vague mentions. Default: False.
            _page_label: Optional label for the current page/chunk used in verbose
                logging (e.g. "page 5"). Internal parameter for extract_from_document.

        Returns:
            Dict with 'input_text' and 'datasets' keys containing the original text
            and list of extracted dataset mentions with indices relative to original text.
            When use_classifier=True and the page is skipped, also contains 'skip_reason':
            either 'non_english' or 'no_data'.
        """
        # Optionally normalize input text (fix OCR artifacts)
        if normalize_text:
            text = self._normalize_input_text(text)

        schema = (
            custom_schema if custom_schema is not None else self._build_schema(extract_provenance)
        )

        # Pre-filter: two-stage gate when use_classifier=True.
        #   Stage 1: cheap stopword heuristic — skip non-English pages immediately.
        #   Stage 2: BERT classifier — skip pages the model predicts have NO_DATA.
        # Both stages are skipped when use_classifier=False (default).
        if use_classifier:
            from ..utils.document_parser import DocumentParser

            if not DocumentParser.is_english(text):
                if verbose:
                    label = _page_label or "chunk"
                    preview = text.strip()[:80].replace("\n", " ")
                    logger.debug("SKIP %s (non-English) | preview: %r", label, preview)
                    print(f"   SKIP {label} (non-English) | preview: {preview!r}")
                return {"input_text": text, "datasets": [], "skip_reason": "non_english"}

            result = self.classifier(text)
            if result[0]["label"] == "NO_DATA":
                if verbose:
                    label = _page_label or "chunk"
                    score = result[0]["score"]
                    logger.debug("SKIP %s (NO_DATA, conf=%.2f)", label, score)
                    print(f"   SKIP {label} (NO_DATA, conf={score:.2f})")
                return {"input_text": text, "datasets": [], "skip_reason": "no_data"}

        # Chunk text if it exceeds token limit (returns list of (chunk_text, offset) tuples)
        if enable_chunking:
            chunks_with_offsets = self._chunk_text(text, max_tokens=max_tokens, overlap=50)
        else:
            chunks_with_offsets = [(text, 0)]

        if len(chunks_with_offsets) == 1:
            # Text is short enough, process directly (offset is 0)
            chunk_text, _ = chunks_with_offsets[0]
            results = schema.extract_with_classification(
                chunk_text,
                self.model,
                include_confidence=include_confidence,
                include_spans=True,
            )
        else:
            # Process each chunk and merge results
            chunk_results = []
            for chunk_text, chunk_offset in chunks_with_offsets:
                chunk_result = schema.extract_with_classification(
                    chunk_text,
                    self.model,
                    include_confidence=include_confidence,
                    include_spans=True,
                )

                # Adjust indices in this chunk's results
                if isinstance(chunk_result, dict):
                    entities = self._get_entities(chunk_result)
                    seen_ids = set()
                    for entity in entities:
                        if isinstance(entity, dict):
                            self._adjust_entity_indices(entity, chunk_offset, seen_ids)

                chunk_results.append(chunk_result)

            # Merge results from all chunks
            results = self._merge_chunk_results(chunk_results)

        # Extract dataset list from results
        if isinstance(results, dict):
            datasets = self._get_entities(results)
        else:
            datasets = results if isinstance(results, list) else []

        # Apply dataset_threshold filter if specified
        if dataset_threshold is not None:
            filtered = []
            for ds in datasets:
                name = self._get_name_field(ds)
                conf = name.get("confidence", 1.0) if isinstance(name, dict) else 1.0
                if conf >= dataset_threshold:
                    filtered.append(ds)
            datasets = filtered

        # Optionally filter out non-dataset tags
        if exclude_non_datasets:
            datasets = [
                ds
                for ds in datasets
                if not (
                    # Check v21-diversity field name
                    isinstance(ds.get("specificity_tag"), dict)
                    and ds["specificity_tag"].get("text") == "non-dataset"
                    or ds.get("specificity_tag") == "non-dataset"
                    # Check legacy field name
                    or isinstance(ds.get("dataset_tag"), dict)
                    and ds["dataset_tag"].get("text") == "non-dataset"
                    or ds.get("dataset_tag") == "non-dataset"
                )
            ]

        # Optionally drop mentions where is_used is indeterminate (na / None / "")
        if exclude_na_usage:

            def _is_used_val(ds):
                v = ds.get("is_used")
                return v.get("text") if isinstance(v, dict) else str(v) if v else ""

            datasets = [ds for ds in datasets if _is_used_val(ds) not in ("", "na")]

        # Deduplicate: filter null names and keep highest-confidence per span
        datasets = self._deduplicate_entities(datasets)

        # Add clean_text field (normalized whitespace for display) and validate acronyms
        for ds in datasets:
            self._clean_entity_text(ds)
            acro_field = ds.get("acronym")
            if isinstance(acro_field, dict) and acro_field.get("text"):
                name_field = self._get_name_field(ds)
                name_text = (
                    name_field.get("text", "")
                    if isinstance(name_field, dict)
                    else str(name_field or "")
                )
                if name_text:
                    acro_text = acro_field.get("text", "")
                    if acro_text.lower() in name_text.lower() or not self._is_valid_acronym(
                        acro_text, name_text
                    ):
                        acro_field["text"] = ""
                        acro_field["clean_text"] = ""
                        acro_field["start"] = None
                        acro_field["end"] = None

        # Optionally apply heuristic filters
        if apply_heuristics:
            datasets = self._apply_heuristic_filters(datasets, text=text)

        return {"input_text": text, "datasets": datasets}

    def _clean_entity_text(self, entity: Dict[str, Any]) -> None:
        """Add clean_text field to entity text fields.

        Normalizes whitespace (collapses newlines and multiple spaces) for
        display purposes. Original text, start, and end fields are preserved
        to maintain valid index spans against the source markdown.

        Args:
            entity: Entity dict to add clean_text fields to (modified in-place)
        """
        text_fields = [
            "dataset_name",
            "mention_name",
            "description",
            "acronym",
            "producer",
            "author",
            "geography",
            "reference_population",
        ]
        for field in text_fields:
            val = entity.get(field)
            if isinstance(val, dict) and "text" in val and isinstance(val["text"], str):
                # Collapse newlines and multiple spaces, strip edges
                cleaned = re.sub(r"\s+", " ", val["text"]).strip()
                val["clean_text"] = cleaned

    def _is_valid_acronym(self, acronym: str, name: str) -> bool:
        """Validate if acronym is potentially correct for a given dataset name.

        Checks if the acronym (excluding non-alphanumeric characters) is a substring
        or a subsequence of the dataset name (case-insensitive).
        """
        if not acronym:
            return True
        acro_clean = re.sub(r"[^a-zA-Z0-9]", "", acronym).lower()
        name_clean = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
        if not acro_clean:
            return True
        if acro_clean in name_clean:
            return True
        # Subsequence check: letters of acronym must appear in name in same relative order
        it = iter(name_clean)
        return all(char in it for char in acro_clean)

    @staticmethod
    def _is_table_figure_ref(name: str) -> bool:
        """Check if name is a single or compound table/figure/panel/annex label."""
        # Split by typical separators like "and", "or", "&", ",", ";"
        parts = re.split(r"\b(?:and|or|&)\b|[,;]", name, flags=re.IGNORECASE)
        if not parts:
            return False
        for part in parts:
            part_clean = part.strip()
            if not part_clean:
                continue
            if not _TABLE_FIGURE_PREFIX_RE.match(part_clean):
                return False
        return True

    @staticmethod
    def _is_all_caps_document_title(name: str) -> bool:
        """Check if name is a long all-caps document title/header."""
        if not name.isupper():
            return False
        if len(name) < 25 or " " not in name:
            return False
        return True

    def _is_personal_name(self, name: str) -> bool:
        """Check if a candidate name is a personal name using probablepeople and structural heuristics.

        Uses a dataset keyword pre-filter to protect actual datasets.
        """
        # 1. Pre-filter: if text contains any dataset keywords, it is not a personal name
        dataset_keywords = {
            "survey",
            "surveys",
            "census",
            "censuses",
            "database",
            "databases",
            "db",
            "registry",
            "registries",
            "indicator",
            "indicators",
            "index",
            "indices",
            "microdata",
            "data",
            "dataset",
            "datasets",
            "study",
            "studies",
            "report",
            "reports",
            "map",
            "maps",
            "statistics",
            "stats",
            "profile",
            "profiles",
            "panel",
            "panels",
            "round",
            "rounds",
            "assessment",
            "assessments",
        }
        name_lower = name.lower()
        if any(kw in name_lower for kw in dataset_keywords):
            return False

        # 2. Clean name from parenthetical annotations (e.g. "Konstantin Fastovets (UNHCR)" -> "Konstantin Fastovets")
        clean_name = re.sub(r"\s*[\(\[\{].*?[\)\]\}]\s*", " ", name).strip()
        if not clean_name:
            return False

        # 3. Check for single-word all-caps acronyms (e.g. "OECD", "MSNA", "UNHCR")
        if clean_name.isupper() and " " not in clean_name:
            return False

        try:
            import probablepeople as pp

            # Tag the cleaned name
            _, name_type = pp.tag(clean_name)
            if name_type == "Person":
                return True

            # Edge case handling: Konstantin Fastovets is classified as Corporation by pp.tag
            # If the parser tagged it as Corporation, but it consists only of 2-3 capitalized words
            # and doesn't contain common corporate/institutional keywords, it is likely a Person name.
            if name_type == "Corporation":
                words = clean_name.split()
                if 2 <= len(words) <= 3 and all(w[0].isupper() for w in words if w.isalpha()):
                    corporate_keywords = {
                        "corp",
                        "corporation",
                        "inc",
                        "incorporated",
                        "llc",
                        "ltd",
                        "limited",
                        "co",
                        "company",
                        "association",
                        "institute",
                        "university",
                        "dept",
                        "department",
                        "agency",
                        "commission",
                        "ministry",
                        "office",
                        "bank",
                        "unhcr",
                        "unicef",
                        "undp",
                        "iom",
                        "world",
                        "center",
                        "centre",
                        "group",
                        "foundation",
                        "union",
                        "organization",
                        "organisation",
                    }
                    if not any(w.lower() in corporate_keywords for w in words):
                        return True

        except Exception:
            # Fallback for parsing failures or RepeatedLabelError
            # Check basic structure: 2-3 words, all capitalized, alphabetic
            words = clean_name.split()
            if 2 <= len(words) <= 3 and all(w.isalpha() and w[0].isupper() for w in words):
                non_name_keywords = {
                    "corp",
                    "corporation",
                    "inc",
                    "llc",
                    "ltd",
                    "co",
                    "company",
                    "unhcr",
                    "unicef",
                    "undp",
                    "iom",
                    "bank",
                    "ministry",
                    "agency",
                    "office",
                    "survey",
                    "census",
                    "dataset",
                    "indicator",
                    "index",
                    "world",
                    "center",
                    "centre",
                    "group",
                    "foundation",
                    "union",
                    "organization",
                    "organisation",
                }
                if not any(w.lower() in non_name_keywords for w in words):
                    return True

        return False

    def _apply_heuristic_filters(
        self, datasets: List[Dict[str, Any]], text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter out likely false positive dataset mentions.

        Removes entities that match common non-dataset patterns like
        table/figure labels, single-word common English terms, or markdown
        headers / table columns.

        Args:
            datasets: List of dataset mention dicts
            text: Original raw input text (optional, used to analyze markdown context)

        Returns:
            Filtered list with likely false positives removed
        """
        filtered = []
        for ds in datasets:
            name_field = self._get_name_field(ds) or {}
            if isinstance(name_field, dict):
                name_text = name_field.get("clean_text") or name_field.get("text", "")
                start = name_field.get("start")
                end = name_field.get("end")
            else:
                name_text = str(name_field or "")
                start = None
                end = None

            name_text = name_text.strip()
            if not name_text:
                continue

            # Skip academic/citation patterns (e.g. Sohst et al., 2024)
            if _CITATION_RE.match(name_text):
                continue

            # Skip table/figure/panel/annex/appendix labels (including complex/compound ones)
            if self._is_table_figure_ref(name_text):
                continue

            # Skip document name patterns and citations
            if _DOCUMENT_SUFFIX_RE.search(name_text) or _DOCUMENT_NO_RE.search(name_text):
                continue

            # Skip long all-caps cover page titles / headers without dataset keywords
            if self._is_all_caps_document_title(name_text):
                continue

            # Skip personal names of individuals
            if self._is_personal_name(name_text):
                continue

            # Skip single-word common noise
            if name_text.lower() in _SINGLE_WORD_NOISE:
                continue

            # Markdown header / table column formatting context filter
            if text and start is not None and end is not None:
                # 1. Check if the line containing the span starts with '#' (markdown header)
                line_start = text.rfind("\n", 0, start) + 1
                line_end = text.find("\n", end)
                if line_end == -1:
                    line_end = len(text)
                line_text = text[line_start:line_end].strip()

                is_header = line_text.startswith("#")

                # 2. Check surrounding non-whitespace characters for bold formatting
                before = []
                i = start - 1
                while i >= 0 and len(before) < 4:
                    if not text[i].isspace():
                        before.append(text[i])
                    i -= 1
                before_str = "".join(reversed(before))

                after = []
                j = end
                while j < len(text) and len(after) < 4:
                    if not text[j].isspace():
                        after.append(text[j])
                    j += 1
                after_str = "".join(after)

                is_bold = (
                    "**" in before_str or "**" in after_str or "*" in before_str or "*" in after_str
                )

                # If the span is within a header or bold formatting block
                if is_header or is_bold:
                    # Reject all CAPS column headers / table keywords
                    if name_text.isupper():
                        if " " in name_text or len(name_text) > 5:
                            continue  # Reject!

                    header_noise_patterns = [
                        r"sample size",
                        r"number of observations",
                        r"country",
                        r"countries",
                        r"p-value",
                        r"std\.? error",
                        r"standard error",
                        r"t-statistic",
                        r"coefficient",
                        r"mean",
                        r"median",
                        r"std\.? dev",
                        r"total population",
                        r"variable",
                        r"year",
                        r"obs\.?",
                    ]
                    rejected = False
                    for pattern in header_noise_patterns:
                        if re.search(r"\b" + pattern + r"\b", name_text, re.IGNORECASE):
                            rejected = True
                            break
                    if rejected:
                        continue  # Reject!

            filtered.append(ds)
        return filtered

    def extract_from_document(
        self,
        source: Union[str, Path],
        include_confidence: bool = True,
        custom_schema: Optional[Any] = None,
        n_pages: int = 1,
        include_metadata: bool = True,
        exclude_non_datasets: bool = True,
        dataset_threshold: Optional[float] = None,
        max_tokens: int = 200,
        enable_chunking: bool = True,
        apply_heuristics: bool = False,
        use_classifier: bool = True,
        normalize_text: bool = True,
        skip_references: bool = False,
        extract_provenance: bool = False,
        verbose: bool = False,
        pages: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract dataset mentions from a PDF document.

        Args:
            source: Path to PDF file or URL
            include_confidence: Whether to include confidence scores
            custom_schema: Custom schema to use instead of default
            n_pages: Number of pages per chunk (default: 1 for page-by-page processing)
            include_metadata: Whether to include source document and page text metadata
            exclude_non_datasets: If True, filter out entries tagged as
                'non-dataset'. Safety net for pre-v2 models.
            dataset_threshold: Optional confidence threshold for mention_name field
            max_tokens: Maximum tokens per chunk (default: 200, aligned with
                training data token distribution)
            enable_chunking: Whether to split long text into chunks (default: True)
            apply_heuristics: If True, apply heuristic filters to remove likely
                false positives such as table/figure labels (default: False)
            use_classifier: If True, skip chunks that fail the English pre-filter
                before running GLiNER2 extraction (default: False)
            normalize_text: If True, normalize each page's text before extraction
                by fixing hyphenated line breaks and collapsing excessive
                whitespace. Useful for pymupdf4llm markdown outputs (default: False)
            skip_references: If True, skip pages after a references/appendix section
                is detected in the second half of the document (default: False)
            extract_provenance: If True, also extract provenance fields
                (author, producer, publication_year, reference_year,
                reference_population, geography, description, acronym).
                Increases inference latency. Defaults to False.
            verbose: If True, print logging when references are detected and pages
                are skipped (default: False)
            pages: Optional list of 0-indexed page numbers to include. If None,
                   processes all pages.

        Returns:
            List of extracted dataset mentions with metadata including page numbers,
            source document, and page text (if include_metadata=True)
        """
        # Convert source to string for metadata
        source_str = str(source)

        # Load PDF in chunks with page tracking
        chunks = DocumentParser.load_pdf_chunks(
            source_str,
            n_pages=n_pages,
            skip_references=skip_references,
            verbose=verbose,
            pages=pages,
        )

        # Extract from each chunk and aggregate results
        all_results = []
        skipped_classifier = []
        if verbose:
            logger.debug(
                "Processing %d chunk(s) with use_classifier=%s", len(chunks), use_classifier
            )
            print(f"\n   Processing {len(chunks)} chunk(s) with use_classifier={use_classifier}")

        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            chunk_pages = chunk["pages"]
            page_label = f"page {chunk_pages[0] + 1}" if chunk_pages else "chunk"

            # Extract from this chunk (returns dict with 'input_text' and 'datasets')
            extraction_result = self.extract_from_text(
                chunk_text,
                include_confidence=include_confidence,
                custom_schema=custom_schema,
                exclude_non_datasets=exclude_non_datasets,
                dataset_threshold=dataset_threshold,
                max_tokens=max_tokens,
                enable_chunking=enable_chunking,
                apply_heuristics=apply_heuristics,
                use_classifier=use_classifier,
                normalize_text=normalize_text,
                extract_provenance=extract_provenance,
                verbose=verbose,
                _page_label=page_label,
            )

            input_text = extraction_result["input_text"]
            datasets_extracted = extraction_result["datasets"]

            # Track classifier-skipped pages for summary.
            # extract_from_text embeds a skip_reason key when it short-circuits.
            skip_reason = extraction_result.get("skip_reason")
            classifier_skipped = skip_reason is not None
            if classifier_skipped:
                skipped_classifier.extend(chunk_pages)

            document_metadata = {"source": source_str, "pages": chunk_pages}
            all_results.append(
                {
                    "page": chunk_pages[0] if chunk_pages else None,
                    "chunk": i,
                    "input_text": input_text,
                    "datasets": datasets_extracted,
                    "classifier_skipped": classifier_skipped,
                    "skip_reason": skip_reason,
                    "document": document_metadata,
                }
            )

        if verbose:
            total_pages = sum(len(c["pages"]) for c in chunks)
            logger.debug(
                "Extraction complete: %d pages, %d classifier-skipped",
                total_pages,
                len(skipped_classifier),
            )
            print("\n   Extraction complete:")
            print(f"   Pages in chunks: {total_pages}")
            print(f"   Classifier-skipped pages: {len(skipped_classifier)}")
            if skipped_classifier:
                pages_display = [p + 1 for p in sorted(skipped_classifier)]
                print(f"   Skipped page numbers: {pages_display}")

        return all_results

    def extract_batch(
        self,
        texts: List[str],
        include_confidence: bool = True,
        custom_schema: Optional[Any] = None,
        use_classifier: bool = False,
        apply_heuristics: bool = False,
        exclude_non_datasets: bool = True,
        extract_provenance: bool = False,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract dataset mentions from multiple texts.

        Args:
            texts: List of input texts
            include_confidence: Whether to include confidence scores
            custom_schema: Custom schema to use instead of default
            use_classifier: Whether to use two-stage pre-filtering classifier (default: False)
            apply_heuristics: If True, apply heuristic filters (default: False)
            exclude_non_datasets: If True, filter out non-dataset tagged entries (default: True)
            verbose: If True, print skip messages (default: False)

        Returns:
            List of dicts, each containing 'input_text' and 'datasets' for each input text
        """
        results = []
        for text in texts:
            result = self.extract_from_text(
                text,
                include_confidence=include_confidence,
                custom_schema=custom_schema,
                use_classifier=use_classifier,
                apply_heuristics=apply_heuristics,
                exclude_non_datasets=exclude_non_datasets,
                extract_provenance=extract_provenance,
                verbose=verbose,
            )
            results.append(result)
        return results
