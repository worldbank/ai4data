"""Dataset data mention extraction schema for GLiNER2 two-pass hybrid inference.

This schema implements the 2-pass extraction strategy required by the
ai4data/datause-extraction adapter:
1. `extract()` for mention entities (named, descriptive, vague).
2. `batch_extract()` on the context window of each extracted mention for
   classification metadata (typology_tag, is_used, usage_context).
"""

from typing import Any, Dict, List

# ==============================================================================
# PRESERVED: Original 2-pass implementation (Option C)
# ==============================================================================
# class DatasetSchemaV2:
#     """Schema implementation using 2-pass inference (Option C)."""
#
#     DEFAULT_THRESHOLD = 0.3
#     CONTEXT_WINDOW = 350
#
#     ENTITY_DESCRIPTIONS = {
#         "named_mention":       "A proper name or well-known acronym for a data source...",
#         "descriptive_mention": "A described data reference with enough detail to identify a dataset but no formal name.",
#         "vague_mention":       "A generic or loosely specified reference to data with minimal identifying detail.",
#     }
#
#     CLASSIFICATION_LABELS = {
#         "typology_tag": [
#             "survey", "census", "administrative", "database",
#             "indicator", "geospatial", "microdata", "report", "other"
#         ],
#         "is_used": ["True", "False"],
#         "usage_context": ["primary", "supporting", "background"],
#     }
#
#     # Specificity priority for NMS tie-breaking: higher wins.
#     _SPEC_PRIORITY = {"named": 3, "descriptive": 2, "vague": 1}
#
#     def __init__(self, threshold: float = DEFAULT_THRESHOLD):
#         self.threshold = threshold
#         self._entity_schema = None
#         self._cls_schema = None
#
#     def _nms_spans(self, spans: list) -> list:
#         """Deduplicate pass-1 entity spans.
#
#         Handles two cases with a single overlap check:
#           - Exact duplicates: same (start, end) — the model sometimes emits the
#             same character region as both 'named' and 'descriptive'.
#           - Overlapping spans: any partial character overlap is suppressed.
#
#         Sort order: specificity priority (named > descriptive > vague) desc,
#         then confidence desc. The highest-priority span is accepted first;
#         any later span that overlaps with an already-accepted span is dropped.
#         """
#         spans = sorted(
#             spans,
#             key=lambda s: (
#                 -self._SPEC_PRIORITY.get(s.get("specificity_tag", ""), 0),
#                 -s.get("confidence", 0),
#             ),
#         )
#         accepted: list = []
#         for sp in spans:
#             cs, ce = sp["start"], sp["end"]
#             overlaps = any(
#                 not (ce <= a["start"] or cs >= a["end"])
#                 for a in accepted
#             )
#             if not overlaps:
#                 accepted.append(sp)
#         return accepted
#
#     def _build_entity_schema(self, model) -> Any:
#         # Pass 1: Extract flat entities
#         return model.create_schema().entities(self.ENTITY_DESCRIPTIONS)
#
#     def _build_classification_schema(self, model) -> Any:
#         # Pass 2: Classify context windows
#         schema = model.create_schema()
#         for field, labels in self.CLASSIFICATION_LABELS.items():
#             schema = schema.classification(field, labels=labels)
#         return schema
#
#     def extract_with_classification(
#         self,
#         text: str,
#         model,
#         include_confidence: bool = True,
#         include_spans: bool = True,
#     ) -> List[Dict[str, Any]]:
#         """Run 2-pass batched inference on a chunk of text."""
#
#         if self._entity_schema is None:
#             self._entity_schema = self._build_entity_schema(model)
#         if self._cls_schema is None:
#             self._cls_schema = self._build_classification_schema(model)
#
#         # Pass 1: entity extraction
#         p1_res = model.extract(
#             text,
#             self._entity_schema,
#             include_confidence=True,
#             include_spans=True,
#             threshold=self.threshold
#         )
#
#         # Handle dict or list return type from GLiNER extract
#         raw_entities = p1_res.get("entities", p1_res) if isinstance(p1_res, dict) else p1_res
#
#         # Flatten into a single list of span dictionaries
#         entities = []
#         if isinstance(raw_entities, dict):
#             for label, spans in raw_entities.items():
#                 if isinstance(spans, list):
#                     for sp in spans:
#                         sp["label"] = label
#                         sp["specificity_tag"] = label.replace("_mention", "")
#                         entities.append(sp)
#         elif isinstance(raw_entities, list):
#             entities = raw_entities
#
#         if not entities:
#             return []
#
#         # Deduplicate: keep highest-specificity span for exact and overlapping positions.
#         entities = self._nms_spans(entities)
#
#         # Pass 2: context window classification via batch_extract
#         contexts = []
#         for sp in entities:
#             c_start = max(0, sp["start"] - self.CONTEXT_WINDOW)
#             c_end = min(len(text), sp["end"] + self.CONTEXT_WINDOW)
#             contexts.append(text[c_start:c_end])
#
#         batch_cls = model.batch_extract(contexts, self._cls_schema, threshold=self.threshold)
#
#         results = []
#         for sp, cls in zip(entities, batch_cls):
#             # Parse specificity from pre-tagged field (set during pass-1 flattening)
#             # or fall back to entity label (e.g. named_mention -> named)
#             label = sp.get("label", "")
#             specificity = sp.get("specificity_tag") or (
#                 label.split("_")[0] if "_" in label else label
#             )
#
#             # Map to standard downstream output format
#             # Using mention_name per user confirmation
#             res = {
#                 "mention_name": {
#                     "text": sp["text"],
#                     "confidence": sp.get("confidence", 1.0),
#                     "start": sp["start"],
#                     "end": sp["end"]
#                 },
#                 "specificity_tag": {
#                     "text": specificity,
#                     "confidence": sp.get("confidence", 1.0),
#                     "start": sp["start"],
#                     "end": sp["end"]
#                 }
#             }
#
#             # Merge pass 2 classification fields
#             for field in self.CLASSIFICATION_LABELS.keys():
#                 cls_val = cls.get(field, "")
#                 if isinstance(cls_val, dict):
#                     # batch_extract might return dicts with confidence for classification fields
#                     res[field] = cls_val
#                 else:
#                     # fallback
#                     res[field] = {
#                         "text": str(cls_val),
#                         "confidence": 1.0,
#                         "start": sp["start"],
#                         "end": sp["end"]
#                     }
#
#             results.append(res)
#
#         return results


def map_typology(text: str) -> str:
    val = text.strip().lower()
    if "survey" in val:
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


class DatasetSchemaV2:
    """Schema implementation using V4/V5 model single-pass JSON schema."""

    DEFAULT_THRESHOLD = 0.3
    LABEL_PREFIX = (
        "specificity: named | descriptive | vague usage: primary | supporting | background |"
    )

    SCHEMA = {
        "data_mention": [
            "name::str::The exact full name of the data source or dataset",
            "acronym::str::The acronym or abbreviation if any",
            "specificity::str::Whether this mention is named, descriptive, or vague",
            "usage::str::Whether this is primary, supporting, or background data",
            "datatype::str::The type of data verbatim from text such as survey, report, census, program, system, or assessment",
            "producer::str::The organization or entity that produced or published the data",
            "timeframe::str::The year or time period of the data such as 2019 or 2019 to 2020",
        ]
    }

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def extract_with_classification(
        self,
        text: str,
        model,
        include_confidence: bool = True,
        include_spans: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run single-pass inference on a chunk of text using V4/V5 schema."""
        clean_text = text.strip()
        prefix_len = len(self.LABEL_PREFIX)

        prefixed = f"{self.LABEL_PREFIX} {clean_text}"

        # Run single-pass extract_json
        result = model.extract_json(
            prefixed,
            self.SCHEMA,
            threshold=self.threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
        )

        raw_mentions = result.get("data_mention", [])
        if isinstance(raw_mentions, dict):
            raw_mentions = [raw_mentions] if raw_mentions else []

        def _get_str(field) -> str:
            if isinstance(field, dict):
                return field.get("text", field.get("selected", "")) or ""
            return str(field or "")

        results = []
        for m in raw_mentions:
            name_field = m.get("name", "")
            if isinstance(name_field, dict):
                name = name_field.get("text", "")
                confidence = float(name_field.get("confidence", 1.0))
                start = name_field.get("start", None)
                end = name_field.get("end", None)
            else:
                name = str(name_field or "")
                confidence = 1.0
                start = None
                end = None

            if not name or len(name.strip()) <= 1:
                continue

            # Offset adjustment: subtract len(LABEL_PREFIX) + 1 to align with original text
            offset = prefix_len + 1
            if start is not None:
                start = max(0, start - offset)
            if end is not None:
                end = max(0, end - offset)

            # Helper to map a field to dict format
            def _map_field(field_val, default_val="", is_typology=False):
                text_val = default_val
                field_conf = confidence
                field_start = start
                field_end = end

                if isinstance(field_val, dict):
                    text_val = field_val.get("text", field_val.get("selected", "")) or default_val
                    field_conf = float(field_val.get("confidence", confidence))
                    if "start" in field_val:
                        field_start = max(0, field_val["start"] - offset)
                    if "end" in field_val:
                        field_end = max(0, field_val["end"] - offset)
                elif field_val is not None:
                    text_val = str(field_val).strip()

                if is_typology:
                    text_val = map_typology(text_val)

                return {
                    "text": text_val,
                    "confidence": field_conf,
                    "start": field_start,
                    "end": field_end,
                }

            usage_context_dict = _map_field(m.get("usage"), "primary")
            usage_str = usage_context_dict["text"].strip().lower()
            is_used_val = "True"
            if usage_str == "background":
                is_used_val = "False"
            elif usage_str in ("primary", "supporting"):
                is_used_val = "True"
            else:
                is_used_val = "True"

            res = {
                "mention_name": {
                    "text": name.strip(),
                    "confidence": confidence,
                    "start": start,
                    "end": end,
                },
                "specificity_tag": _map_field(m.get("specificity"), "named"),
                "usage_context": usage_context_dict,
                "typology_tag": _map_field(m.get("datatype"), "other", is_typology=True),
                "acronym": _map_field(m.get("acronym")),
                "producer": _map_field(m.get("producer")),
                "reference_year": _map_field(m.get("timeframe")),
                "is_used": {
                    "text": is_used_val,
                    "confidence": usage_context_dict["confidence"],
                    "start": usage_context_dict["start"],
                    "end": usage_context_dict["end"],
                },
                "geography": {"text": "", "confidence": confidence, "start": start, "end": end},
            }
            results.append(res)

        return results


class DatasetSchemaV3:
    """Schema using entity+relation extraction (Pass 1) with per-mention
    classification fallback (Pass 2).

    Pass 1 runs full-text entity+relation extraction with ``LABEL_PREFIX``
    prepended. This extracts dataset name spans, factual metadata
    (acronym, producer, timeframe, datatype) via relations, and
    classification (specificity, usage) via ``has_specificity`` /
    ``has_usage`` relations when the relation head fires.

    Pass 2 only runs for mentions that did **not** receive a classification
    relation in Pass 1. For each such mention a context window is sliced,
    the prefix is prepended, and ``specificity`` / ``usage`` are extracted
    as entities. Because each window contains a single mention the
    classification is unambiguous.

    This avoids the ``extract_json`` count-prediction bottleneck while
    maintaining 100 % classification coverage.
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
        "timeframe": ("The year or time period of the data such as 2019 or 2019 to 2020"),
        "geography": ("The country, region, or geographic area the data covers"),
    }
    _RELATION_DEFS = {
        "has_acronym": "The acronym of the dataset",
        "has_organization": "The organization of the dataset",
        "has_timeframe": "The timeframe of the dataset",
        "has_geography": "The country or geographic coverage area of the dataset",
    }

    # Maps relation types to output field names
    _FACTUAL_RELATIONS = {
        "has_acronym": "acronym",
        "has_organization": "producer",
        "has_timeframe": "reference_year",
        "has_geography": "geography",
    }

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._pass1_schema = None
        self._fallback_schema = None

    # ── Schema builders (lazy, cached) ────────────────────────────────────

    def _get_pass1_schema(self, model):
        if self._pass1_schema is None:
            s = model.create_schema()
            s.entities(self._ENTITY_DEFS)
            s.relations(self._RELATION_DEFS)
            self._pass1_schema = s
        return self._pass1_schema

    def _get_fallback_schema(self, model):
        if self._fallback_schema is None:
            s = model.create_schema()
            s.classification(
                "usage",
                ["primary", "supporting", "background"],
                multi_label=False
            )
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
                multi_label=False
            )
            self._fallback_schema = s
        return self._fallback_schema

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_sentence_context(text, start, end):
        """Find the sentence containing the span [start, end]."""
        # Clamp start and end to text bounds to prevent IndexError on out-of-bounds mock spans
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        s_start = start
        while s_start > 0:
            if text[s_start - 1] in {'.', '!', '?', '\n'}:
                break
            s_start -= 1
        
        s_end = end
        while s_end < len(text):
            if text[s_end] in {'.', '!', '?', '\n'}:
                s_end += 1
                break
            s_end += 1
            
        return text[s_start:s_end].strip()

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

        If head_text is provided and the relation type is ``has_producer``,
        self-links are rejected (tail text == head text or substantial
        substring).
        """
        best, best_conf = None, -1
        for r in relations.get(rel_type, []):
            if r["head"]["start"] == head_start and r["head"]["end"] == head_end:
                tail = r["tail"]
                tc = tail["confidence"]
                # Self-link filter for producer relations
                if (
                    head_text
                    and rel_type == "has_organization"
                    and DatasetSchemaV3._is_self_link(head_text, tail.get("text", ""))
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
        """Run 2-pass hybrid inference on a chunk of text."""
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

            # Factual relations
            for rel_type, field_name in self._FACTUAL_RELATIONS.items():
                matched = self._find_best_relation(
                    relations,
                    rel_type,
                    ne["start"],
                    ne["end"],
                    head_text=name_text,
                )
                if matched:
                    text_val = matched["text"]
                    # Only set if not already set, or if this match is higher confidence
                    existing = rec.get(field_name)
                    if not existing or existing.get("confidence", 0.0) < float(matched["confidence"]):
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
            fallback_schema = self._get_fallback_schema(model)
            contexts = []
            for rec_idx in needs_fallback:
                rec = records[rec_idx]
                sentence = self._get_sentence_context(
                    clean_text,
                    rec["_orig_start"],
                    rec["_orig_end"]
                )
                contexts.append(sentence)

            fb_results = model.batch_extract(
                contexts,
                fallback_schema,
                threshold=0.1,
                include_confidence=True,
            )

            for i, rec_idx in enumerate(needs_fallback):
                fb = fb_results[i] if i < len(fb_results) else {}
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

                # If classification is "other", try to fall back to keyword matching the mention name
                if text_val == "other":
                    mapped_val = map_typology(rec["mention_name"]["text"])
                    if mapped_val != "other":
                        text_val = mapped_val
                        conf_val = 0.0

                rec["typology_tag"] = {
                    "text": text_val,
                    "confidence": conf_val,
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


# Backward compatibility wrapper
class DatasetSchema:
    DEFAULT_THRESHOLD = 0.3

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._impl = DatasetSchemaV3(threshold=threshold)

    def set_threshold(self, field_name: str, threshold: float):
        self.threshold = threshold
        self._impl.threshold = threshold
        return self

    def build(self, model, extract_provenance: bool = False):
        """Deprecated. Always returns self to allow extractor to use extract_with_classification."""
        return self

    def extract_with_classification(self, text: str, model, *args, **kwargs) -> Dict[str, Any]:
        """Wrap the V3 implementation to return the dict expected by chunk merger."""
        entities = self._impl.extract_with_classification(
            text,
            model,
            include_confidence=kwargs.get("include_confidence", True),
            include_spans=kwargs.get("include_spans", True),
        )
        return {"data_mention": entities}
