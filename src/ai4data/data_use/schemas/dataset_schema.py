"""Dataset mention extraction schema (DatasetSchema).

Single canonical schema supporting the three-model swarm pipeline:
  Call 1:  entity extraction (fine-tuned adapter, named/descriptive/vague).
  Call 1b: relation extraction (fine-tuned adapter, 5 types).
  Call 2:  classification (usage, typology, purpose_action).

See ``DatasetExtractor.extract_from_text`` for the production pipeline.
"""

import re
from typing import Any, Dict, List

# Confidence threshold below which Pass 2 model typology is overridden by map_typology.
# Set to 0.0 to disable the override (rely on model only).
_TYPOLOGY_CONFIDENCE_THRESHOLD = 0.85

# Confidence threshold below which the purpose_action classifier output falls
# back to ``contextual_reference`` (the safe default for impact assessment — a
# mislabeled contextual mention costs less than a mislabeled policy_informing
# mention). Set to 0.0 to disable the fallback (always trust the model).
_PURPOSE_ACTION_CONFIDENCE_THRESHOLD = 0.5

# Well-known survey/assessment program acronyms that map_typology should recognise.
# Ordered from most specific to avoid false prefix matches.
_KNOWN_SURVEY_ACRONYMS = frozenset(
    {
        "msna",  # Multi-Sector Needs Assessment
        "hno",  # Humanitarian Needs Overview
        "dhs",  # Demographic and Health Survey
        "mics",  # Multiple Indicator Cluster Survey
        "lsms",  # Living Standards Measurement Study
        "hies",  # Household Income and Expenditure Survey
        "fies",  # Food Insecurity Experience Scale
        "erss",  # Emergency Response Supplemental Survey
        "seis",  # Socio-Economic Impact Survey (context-specific)
        "pma",  # Performance Monitoring for Action
        "afrobarometer",
        "eurobarometer",
        "gallup",
    }
)


def map_typology(text: str) -> str:
    val = text.strip().lower()
    # Check known survey program acronyms (word-boundary safe)
    for acronym in _KNOWN_SURVEY_ACRONYMS:
        if re.search(r"\b" + re.escape(acronym) + r"\b", val):
            return "survey"
    if "survey" in val or "needs assessment" in val or "household survey" in val:
        return "survey"
    if "census" in val:
        return "census"
    if "database" in val or val == "db":
        return "database"
    if "admin" in val or "regist" in val or "system" in val or "record" in val or "platform" in val:
        return "administrative"
    if "indicat" in val or "index" in val or "indices" in val:
        return "indicator"
    if "geo" in val or "gis" in val or "map" in val or "spatial" in val:
        return "geospatial"
    if "micro" in val:
        return "microdata"
    if "report" in val or "document" in val or "paper" in val or "brief" in val:
        return "report"
    if "estimat" in val:
        return "estimates"
    return "other"


VALID_TYPOLOGIES = frozenset(
    {
        "survey",
        "census",
        "database",
        "administrative",
        "indicator",
        "geospatial",
        "microdata",
        "report",
        "estimates",
        "other",
    }
)

# Impact-assessment action labels — what the cited dataset enabled in this document.
# Compact, verb-driven taxonomy derived from unsupervised clustering + GPT
# mining + relabel of 1,587 WB/UNHCR purpose descriptions (see
# ``impact_assessment/outputs/action_taxonomy.md`` for full definitions and
# ``impact_assessment/mining/action_verb_analysis.md`` for methodology).
#
# Kept to 5 labels — within the GLiNER2 classification sweet spot (<=7).
PURPOSE_ACTION_LABELS = (
    "quantitative_analysis",  # estimation / measurement / statistical analysis
    "needs_evaluation",  # assessing needs / evaluating programs / profiling
    "policy_informing",  # informing policy / strategy / recommendations
    "monitoring",  # ongoing tracking / surveillance / trend observation
    "contextual_reference",  # background / supporting evidence / benchmark
)


class DatasetSchema:
    """Schema definitions for the three-model swarm pipeline.

    Provides schema builders for:
      - ``_get_entity_schema`` — Call 1: named_data, descriptive_data, vague_data
      - ``_get_relation_schema`` — Call 1b: named_data + 4 tail types + 5 relations
      - ``_get_classification_schema`` — Call 2: usage, typology, purpose_action

    Also provides shared helpers (NMS, span matching, validation) used by
    ``DatasetExtractor.extract_from_text``.

    The legacy ``extract_with_classification`` method is kept for backward
    compatibility with direct ``custom_schema`` callers.
    """

    DEFAULT_THRESHOLD = 0.3
    LABEL_PREFIX = ""

    # Entity + relation schema field definitions (reused across calls)
    _ENTITY_DEFS = {
        "named_data": "A proper name or well-known acronym for a data source or dataset",
        "descriptive_data": "A described data reference with enough detail to identify a dataset but no formal name",
        "vague_data": "A generic or loosely specified reference to data with minimal identifying detail",
        "acronym": "The acronym or abbreviation if any",
        "organization": "The organization or entity that produced or published the data",
        "year": "A 4-digit year like 2021 or 2022",
        "geography": ("The country, region, or geographic area the data covers"),
    }
    _RELATION_DEFS = {
        "has_acronym": "The acronym of the dataset",
        "has_organization": "The organization of the dataset",
        "has_timeframe": "The timeframe of the dataset",
        "has_geography": "The country or geographic coverage area of the dataset",
        # Impact-assessment relations — see impact_assessment/README.md.
        # Both producer and user link to the shared ``organization`` entity
        # type; they differ only in relation type so GLiNER2 keeps a single
        # organization detector and disambiguates by relation.
        "used_by": (
            "The organization or entity that is using or citing the dataset "
            "in this document (distinct from has_organization, which captures "
            "the data producer). Example: in 'Poland's Strategy 2026 draws on "
            "UNHCR data', used_by links the UNHCR-data mention to 'Poland'."
        ),
    }

    # Per-relation confidence thresholds for filtering.
    # Sparse relation types (has_acronym, has_timeframe, has_geography)
    # get higher thresholds to suppress over-generation from synthetic-only data.
    _RELATION_THRESHOLDS = {
        "has_acronym": 0.5,
        "has_organization": 0.3,
        "has_timeframe": 0.5,
        "has_geography": 0.5,
        "used_by": 0.3,
    }

    # Maps relation types to output field names
    _FACTUAL_RELATIONS = {
        "has_acronym": "acronym",
        "has_organization": "producer",
        "has_timeframe": "reference_year",
        "has_geography": "geography",
        "used_by": "user",  # impact-assessment: who cited / used the dataset here
    }

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._pass1_schema = None
        self._entity_schema = None  # entity-only schema for Call 1
        self._relation_schema = None  # relation-extraction schema (Call 1b)
        self._classification_schema = None

    # ── Schema builders (lazy, cached) ────────────────────────────────────

    def _get_pass1_schema(self, model):
        if self._pass1_schema is None:
            s = model.create_schema()
            s.entities(self._ENTITY_DEFS)
            s.relations(self._RELATION_DEFS)
            self._pass1_schema = s
        return self._pass1_schema

    def _get_entity_schema(self, model):
        """Entity-only schema for Call 1.

        Contains only the three mention-name entity types
        (``named_data``, ``descriptive_data``, ``vague_data``) with no
        metadata entities and no relations.  This gives the model the broadest
        possible net to detect all data mentions across a full chunk without
        the relation overhead — keeping Call 1 fast and high-recall.
        """
        if self._entity_schema is None:
            s = model.create_schema()
            s.entities(
                {
                    "named_data": self._ENTITY_DEFS["named_data"],
                    "descriptive_data": self._ENTITY_DEFS["descriptive_data"],
                    "vague_data": self._ENTITY_DEFS["vague_data"],
                }
            )
            self._entity_schema = s
        return self._entity_schema

    def _get_relation_schema(self, model):
        """Relation-extraction schema for Call 1b.

        Contains entities the relation model needs to detect
        (``named_data`` as relation heads, plus ``organization``,
        ``acronym``, ``year``, ``geography`` as tails) and all five
        relation types.  This schema runs on the fine‑tuned relation
        adapter; the extracted relations override the zero‑shot
        results from Call 1.
        """
        if self._relation_schema is None:
            s = model.create_schema()
            s.entities(
                {
                    "named_data": self._ENTITY_DEFS["named_data"],
                    "organization": self._ENTITY_DEFS["organization"],
                    "acronym": self._ENTITY_DEFS["acronym"],
                    "year": self._ENTITY_DEFS["year"],
                    "geography": self._ENTITY_DEFS["geography"],
                }
            )
            s.relations(self._RELATION_DEFS)
            self._relation_schema = s
        return self._relation_schema

    def _get_classification_schema(self, model):
        """Classification schema for Call 2.

        Contains three classification tasks:
          - ``usage``: primary / supporting / background
          - ``typology``: survey / census / database / administrative / indicator /
            geospatial / microdata / report / estimates / other
          - ``purpose_action``: 5 impact-assessment labels

        Run on deduplicated sentence contexts after Call 1 and Call 1b.
        """
        if self._classification_schema is None:
            s = model.create_schema()
            s.classification("usage", ["primary", "supporting", "background"], multi_label=False)
            s.classification(
                "typology",
                [
                    "survey",
                    "census",
                    "database",
                    "administrative",
                    "indicator",
                    "geospatial",
                    "microdata",
                    "report",
                    "estimates",
                    "other",
                ],
                multi_label=False,
            )
            # Impact-assessment action classification — what the cited dataset
            # enabled in this document. Five compact, verb-driven labels
            # (see ``impact_assessment/outputs/action_taxonomy.md``).
            s.classification(
                "purpose_action",
                list(PURPOSE_ACTION_LABELS),
                multi_label=False,
            )
            self._classification_schema = s
        return self._classification_schema

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_context(text, start, end, window_sentences=1):
        """Find the sentence(s) containing the span [start, end], expanding
        to ``window_sentences`` sentences before and after.

        With ``window_sentences=1`` (default), returns a single sentence —
        equivalent to ``_get_sentence_context``. Use ``window_sentences≥2``
        for paragraph-level context.

        Returns:
            (context_text, context_offset)
        """
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        boundaries = [0]
        for m in re.finditer(r"(?<=[.!?])\s+|\n", text):
            boundaries.append(m.end())
        boundaries.append(len(text))

        # Find which sentence the span falls within
        found_idx = 0
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= start < boundaries[i + 1]:
                found_idx = i
                break

        # Expand to window_sentences before and after
        before = max(0, found_idx - (window_sentences - 1))
        after = min(len(boundaries) - 1, found_idx + window_sentences)
        # Ensure end of mention is within the window
        while after < len(boundaries) - 1 and boundaries[after] < end:
            after += 1

        ctx_start = boundaries[before]
        ctx_end = boundaries[after]
        raw = text[ctx_start:ctx_end]
        leading = len(raw) - len(raw.lstrip())
        return raw.strip(), ctx_start + leading

    @staticmethod
    def _get_sentence_context(text, start, end):
        """Find the sentence containing the span [start, end].

        Returns:
            (sentence_text, sentence_offset) — the stripped sentence and the
            character offset of its first non-whitespace character within
            ``text``. Callers should use ``sentence_offset`` to translate
            between sentence-local and absolute character spans instead of
            ``text.find(sentence_text)``, which returns the wrong match when
            the same sentence text appears earlier in the document.
        """
        # Clamp start and end to text bounds to prevent IndexError on out-of-bounds mock spans
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        boundaries = [0]
        # Match sentence-ending punctuation followed by whitespace, or a newline
        for m in re.finditer(r"(?<=[.!?])\s+|\n", text):
            boundaries.append(m.end())
        boundaries.append(len(text))

        for i in range(len(boundaries) - 1):
            s_start = boundaries[i]
            s_end = boundaries[i + 1]
            if s_start <= start < s_end:
                actual_end = s_end
                while actual_end < len(text) and actual_end < end:
                    i += 1
                    actual_end = boundaries[i + 1]
                raw = text[s_start:actual_end]
                leading = len(raw) - len(raw.lstrip())
                return raw.strip(), s_start + leading

        raw = text
        leading = len(raw) - len(raw.lstrip())
        return raw.strip(), leading

    @staticmethod
    def _nms_name_spans(name_entities):
        """Non-maximum suppression for overlapping name entity spans.

        When the model detects the same mention twice (e.g. with different
        usage labels), we keep only the highest-confidence span.  Any span
        that character-overlaps with an already-accepted higher-confidence
        span is suppressed.
        """
        sorted_ents = sorted(
            name_entities,
            key=lambda e: -e.get("confidence", 0),
        )
        accepted = []
        for ent in sorted_ents:
            es, ee = ent["start"], ent["end"]
            overlaps = any(not (ee <= a["start"] or es >= a["end"]) for a in accepted)
            if not overlaps:
                accepted.append(ent)
        return accepted

    @staticmethod
    def _is_valid_relation(rel_type, head_text, tail_text):
        """Post-hoc validation: reject semantically impossible relations.

        Model over-generates metadata relations.  These rules reject
        predictions that are structurally invalid regardless of confidence.
        """
        h = head_text.strip().lower()
        t = tail_text.strip().lower()
        if not t:
            return False

        if rel_type == "has_acronym":
            # Must be a plausible acronym for the head text.
            # Case 1: acronym is substring of head name
            if t in h:
                return True
            # Case 2: backronym matches first letters of head words
            head_words = h.split()
            if len(head_words) >= 2 and len(t) >= 2:
                backronym = "".join(w[0] for w in head_words)
                if t == backronym.lower():
                    return True
            # Case 3: known well‑survey acronyms (the only ones the model
            # was trained on — anything else is likely spurious)
            _known_acrs = frozenset(
                {
                    "dhs",
                    "mics",
                    "lsms",
                    "wdi",
                    "em-dat",
                    "acled",
                    "ilostat",
                    "fao",
                    "gts",
                    "pisa",
                }
            )
            if t in _known_acrs:
                return True
            return False

        if rel_type == "has_timeframe":
            # Must parse as a year (4 digits) or year range
            import re

            if re.match(r"^\d{4}$", t) or re.match(r"^\d{4}\s*[-–]\s*\d{4}$", t):
                return True
            return False

        return True  # has_organization, used_by, has_geography: trust model

    # Generic terms the model sometimes predicts as organizations.
    # These are never valid org names — filter them out.
    _NON_ORG_TERMS = frozenset(
        {
            "we",
            "the authors",
            "researchers",
            "the research team",
            "the study",
            "this paper",
            "this report",
            "the analysis",
            "this",
            "it",
            "they",
            "them",
            "these",
            "those",
            "the data",
            "the survey",
            "the dataset",
            "the information",
        }
    )

    @staticmethod
    def _is_valid_org(tail_text: str) -> bool:
        """Reject obviously non-organization tails."""
        return tail_text.strip().lower() not in DatasetSchema._NON_ORG_TERMS

    @staticmethod
    def _is_self_link(head_text, tail_text):
        """Check if a relation tail is a self-link to the head.

        Rejects cases like:
          - head="Ghana population census", tail="Ghana population census"
          - head="2010 Ghana population census", tail="Ghana population census"
          - head="DHS data from Africa", tail="DHS" (when used as producer)
        """
        h = head_text.strip().lower()
        t = tail_text.strip().lower()
        if not t:
            return True
        # Exact match
        if h == t:
            return True
        # Tail is a substantial substring of head (>50% overlap)
        if t in h and len(t) / len(h) > 0.5:
            return True
        return False

    @staticmethod
    def _find_best_relation(relations, rel_type, head_start, head_end, head_text=None):
        """Find highest-confidence relation tail for a given head span.

        Accepts relation heads that are either an exact match OR a sub-span
        (fully contained within) the mention's character boundaries. This covers
        cases where the model correctly predicts a relation but links it to a
        sub-phrase of the full mention name (e.g. head="Republic of Moldova"
        inside mention="Data from the Republic of Moldova").

        If head_text is provided and the relation type is organization-linked
        (``has_organization``, ``used_by``), self-links are rejected (tail text
        equals head text or is a substantial substring).
        """
        best, best_conf = None, -1
        for r in relations.get(rel_type, []):
            h_s = r["head"]["start"]
            h_e = r["head"]["end"]
            # Accept exact match OR head span fully contained within the mention span
            exact_match = h_s == head_start and h_e == head_end
            sub_span_match = h_s >= head_start and h_e <= head_end
            if not (exact_match or sub_span_match):
                continue
            tail = r["tail"]
            tc = tail["confidence"]
            # Self-link filter for producer / user relations
            if (
                head_text
                and rel_type in ("has_organization", "used_by")
                and DatasetSchema._is_self_link(head_text, tail.get("text", ""))
            ):
                continue
            if tc > best_conf:
                best_conf = tc
                best = tail
        return best

    @staticmethod
    def _best_entity(entity_list):
        """Pick the highest-confidence entity from a list."""
        if not entity_list:
            return None
        return max(
            entity_list,
            key=lambda s: s.get("confidence", 0) if isinstance(s, dict) else 0,
        )

    # Words that, when they appear as the final token of a name span, indicate
    # the span was cut at a chunk boundary and the real name continues beyond.
    _TRAILING_STOPWORDS = frozenset(
        {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "nor",
            "for",
            "of",
            "in",
            "on",
            "at",
            "to",
            "by",
            "with",
            "from",
            "into",
            "onto",
            "upon",
            "as",
            "than",
        }
    )

    @classmethod
    def _is_truncated_name(cls, name_text: str) -> bool:
        """Return True if the name ends with a stopword, indicating chunk truncation.

        Examples that should be rejected:
          - "OECD indicators for"  (preposition)
          - "findings and"         (conjunction)
          - "data from"            (preposition)
        """
        last_word = (
            name_text.strip().rstrip(".,;:").split()[-1].lower() if name_text.strip() else ""
        )
        return last_word in cls._TRAILING_STOPWORDS

    # ── Main entry point ──────────────────────────────────────────────────

    def extract_with_classification(
        self,
        text: str,
        model,
        include_confidence: bool = True,
        include_spans: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run 2-pass hybrid inference on a chunk of text.

        Note: This is the legacy single-chunk entry point kept for direct
        ``custom_schema`` callers. The production path inlines the
        three-model swarm pipeline directly in
        ``DatasetExtractor.extract_from_text``, which does not call this
        method. Prefer the inlined pipeline for new code; this method exists
        for backward compatibility and direct custom-schema use.
        """
        clean_text = text.strip()
        prefix_len = len(self.LABEL_PREFIX)
        prefix_offset = prefix_len + (1 if prefix_len > 0 else 0)

        prefixed = f"{self.LABEL_PREFIX} {clean_text}" if prefix_len > 0 else clean_text

        # ── Pass 1: Entity + Relation extraction ─────────────────────────
        pass1_schema = self._get_pass1_schema(model)
        raw = model.extract(
            prefixed,
            pass1_schema,
            threshold=self.threshold,
            include_confidence=True,
            include_spans=True,
        )

        entities = raw.get("entities", {})
        relations = raw.get("relation_extraction", {})

        named_ents = entities.get("named_data", [])
        desc_ents = entities.get("descriptive_data", [])
        vague_ents = entities.get("vague_data", [])

        for ne in named_ents:
            ne["specificity_tag"] = "named"
        for de in desc_ents:
            de["specificity_tag"] = "descriptive"
        for ve in vague_ents:
            ve["specificity_tag"] = "vague"

        name_entities = named_ents + desc_ents + vague_ents

        if not name_entities:
            return []

        # NMS: deduplicate overlapping name spans
        name_entities = self._nms_name_spans(name_entities)

        # Build initial records from Pass 1
        records = []
        needs_fallback = []  # indices of records needing classification

        for idx, ne in enumerate(name_entities):
            adj_start = max(0, ne["start"] - prefix_offset)
            adj_end = max(0, ne["end"] - prefix_offset)
            confidence = float(ne.get("confidence", 1.0))

            name_text = ne["text"].strip()
            if not name_text or len(name_text) <= 2:
                continue
            # Drop names ending with a stopword -- these are chunk-boundary truncations
            if self._is_truncated_name(name_text):
                continue

            rec = {
                "mention_name": {
                    "text": name_text,
                    "confidence": confidence,
                    "start": adj_start,
                    "end": adj_end,
                }
            }

            # Factual relations with per-relation confidence thresholds
            for rel_type, field_name in self._FACTUAL_RELATIONS.items():
                rel_thresh = self._RELATION_THRESHOLDS.get(rel_type, self.threshold)
                matched = self._find_best_relation(
                    relations,
                    rel_type,
                    ne["start"],
                    ne["end"],
                    head_text=name_text,
                )
                if matched and float(matched.get("confidence", 0.0)) >= rel_thresh:
                    if not DatasetSchema._is_valid_relation(
                        rel_type, name_text, matched.get("text", "")
                    ):
                        matched = None
                if matched:
                    text_val = matched["text"]
                    # Only set if not already set, or if this match is higher confidence
                    existing = rec.get(field_name)
                    if not existing or existing.get("confidence", 0.0) < float(
                        matched["confidence"]
                    ):
                        rec[field_name] = {
                            "text": text_val,
                            "confidence": float(matched["confidence"]),
                            "start": max(0, matched["start"] - prefix_offset),
                            "end": max(0, matched["end"] - prefix_offset),
                        }
                else:
                    # Only set default if not already set by another relation
                    if field_name not in rec:
                        rec[field_name] = {
                            "text": "",
                            "confidence": 0.0,
                            "start": adj_start,
                            "end": adj_end,
                        }

            # Classification relations (may be missing)
            spec_match = {"text": ne["specificity_tag"], "confidence": confidence}

            rec["_spec_match"] = spec_match
            rec["_usage_match"] = None
            rec["_orig_start"] = adj_start
            rec["_orig_end"] = adj_end

            needs_fallback.append(len(records))
            records.append(rec)

        # ── Pass 2: Fallback classification ──────────────────────────────
        if needs_fallback:
            fallback_schema = self._get_classification_schema(model)

            # Build sentence contexts and deduplicate to avoid redundant model
            # calls for mentions in the same sentence (common in overlapping chunks).
            sentence_cache: dict = {}  # sentence_text -> fb result (filled after batch_extract)
            ordered_unique: list = []  # unique sentences in insertion order

            rec_to_sentence = {}  # rec_idx -> sentence_text
            for rec_idx in needs_fallback:
                rec = records[rec_idx]
                sentence, _sentence_offset = self._get_sentence_context(
                    clean_text,
                    rec["_orig_start"],
                    rec["_orig_end"],
                )
                rec_to_sentence[rec_idx] = sentence
                if sentence not in sentence_cache:
                    sentence_cache[sentence] = None  # placeholder
                    ordered_unique.append(sentence)

            fb_results = model.batch_extract(
                ordered_unique,
                fallback_schema,
                threshold=0.1,
                include_confidence=True,
            )

            # Populate cache with actual results
            for sent, fb in zip(ordered_unique, fb_results):
                sentence_cache[sent] = fb

            for rec_idx in needs_fallback:
                fb = sentence_cache.get(rec_to_sentence[rec_idx], {}) or {}
                rec = records[rec_idx]

                usage_info = fb.get("usage")
                if usage_info and isinstance(usage_info, dict):
                    rec["_usage_match"] = {
                        "text": usage_info.get("label", "primary"),
                        "confidence": float(usage_info.get("confidence", 0.0)),
                    }

                # Resolve typology tag via native classification
                typology_info = fb.get("typology")
                text_val = "other"
                conf_val = 0.0
                if typology_info and isinstance(typology_info, dict):
                    text_val = typology_info.get("label", "other")
                    conf_val = float(typology_info.get("confidence", 0.0))

                # If classification is "other" OR model confidence is below threshold,
                # fall back to keyword matching on the mention name. This corrects
                # low-confidence misclassifications (e.g. model predicts "census" for MSNA data).
                if text_val == "other" or conf_val < _TYPOLOGY_CONFIDENCE_THRESHOLD:
                    mapped_val = map_typology(rec["mention_name"]["text"])
                    if mapped_val != "other":
                        text_val = mapped_val
                        conf_val = 0.0  # signal: keyword-derived, no model confidence

                rec["typology_tag"] = {
                    "text": text_val,
                    "confidence": conf_val,
                    "start": rec["mention_name"]["start"],
                    "end": rec["mention_name"]["end"],
                }

                # Resolve purpose_action (impact-assessment field).
                # Falls back to ``contextual_reference`` when the classifier
                # is missing or below the safe-default threshold — a
                # mislabeled contextual mention costs less than a mislabeled
                # high-impact label like ``policy_informing``.
                purpose_info = fb.get("purpose_action")
                purpose_label = "contextual_reference"
                purpose_conf = 0.0
                if purpose_info and isinstance(purpose_info, dict):
                    purpose_label = purpose_info.get("label", "contextual_reference")
                    purpose_conf = float(purpose_info.get("confidence", 0.0))
                    if purpose_conf < _PURPOSE_ACTION_CONFIDENCE_THRESHOLD:
                        purpose_label = "contextual_reference"
                        purpose_conf = 0.0
                rec["purpose_action"] = {
                    "text": purpose_label,
                    "confidence": purpose_conf,
                    "start": rec["mention_name"]["start"],
                    "end": rec["mention_name"]["end"],
                }

        # ── Finalize records ─────────────────────────────────────────────
        results = []
        for rec in records:
            spec_match = rec.pop("_spec_match", None)
            usage_match = rec.pop("_usage_match", None)
            orig_start = rec.pop("_orig_start")
            orig_end = rec.pop("_orig_end")
            confidence = rec["mention_name"]["confidence"]

            # Specificity
            if spec_match and isinstance(spec_match, dict):
                spec_text = spec_match.get("text", "named")
                spec_conf = float(spec_match.get("confidence", confidence))
            else:
                spec_text = "named"
                spec_conf = 0.0

            rec["specificity_tag"] = {
                "text": spec_text,
                "confidence": spec_conf,
                "start": orig_start,
                "end": orig_end,
            }

            # Usage
            if usage_match and isinstance(usage_match, dict):
                usage_text = usage_match.get("text", "primary")
                usage_conf = float(usage_match.get("confidence", confidence))
            else:
                usage_text = "primary"
                usage_conf = 0.0

            rec["usage_context"] = {
                "text": usage_text,
                "confidence": usage_conf,
                "start": orig_start,
                "end": orig_end,
            }

            # Derived fields
            usage_lower = usage_text.strip().lower()
            is_used_val = "False" if usage_lower == "background" else "True"
            rec["is_used"] = {
                "text": is_used_val,
                "confidence": usage_conf,
                "start": orig_start,
                "end": orig_end,
            }

            results.append(rec)

        return results
