import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

from rapidfuzz import fuzz

# Define relation fields we want to keep as metadata
# Only acronym is extracted from relations - other fields come directly from the schema
RELATION_META_FIELDS = ["acronym"]

# Note: These fields are descriptive metadata only - they're merged but don't affect clustering
FIELDS_TO_IGNORE_FOR_DEDUP = {"data type", "data description"}


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    return text.strip().rstrip(".,:;()[]{} ")


def as_score_list(score):
    """Convert score to list format."""
    if isinstance(score, list):
        return score
    if score is None:
        return [0]
    return [score]


def label_rank(label):
    """Return priority ranking for labels (lower is better)."""
    ranks = {"named": 0, "descriptive": 1, "vague": 2}
    return ranks.get(label, 99)


# ------------------------------
# Context Extraction
# ------------------------------


def extract_context_window(text: str, start: int, end: int, sentences: int = 1) -> str:
    """Extract ±N sentences around the mention.

    Args:
        text: Full text
        start: Start index of mention
        end: End index of mention
        sentences: Number of sentences before and after to include

    Returns:
        Context string with ~±1 sentence around the mention
    """
    # Simple sentence tokenization using common sentence endings
    sentence_pattern = r"[.!?]+[\s]+"

    # Find sentence boundaries
    sentence_ends = [0]  # Start of text
    for match in re.finditer(sentence_pattern, text):
        sentence_ends.append(match.end())
    sentence_ends.append(len(text))  # End of text

    # Find which sentence contains the mention
    mention_sentence_idx = 0
    for i, boundary in enumerate(sentence_ends[:-1]):
        if sentence_ends[i] <= start < sentence_ends[i + 1]:
            mention_sentence_idx = i
            break

    # Get ±N sentences
    context_start_idx = max(0, mention_sentence_idx - sentences)
    context_end_idx = min(len(sentence_ends) - 1, mention_sentence_idx + sentences + 1)

    context_start = sentence_ends[context_start_idx]
    context_end = sentence_ends[context_end_idx]

    return text[context_start:context_end].strip()


# ------------------------------
# Improved Fuzzy Matching
# ------------------------------


def are_fuzzy_duplicates(text1: str, text2: str, min_len_threshold: int = 10) -> bool:
    """Check if two texts are fuzzy duplicates using multiple metrics.

    Uses a multi-metric approach for robustness:
    - token_sort_ratio: Handles word order differences
    - token_set_ratio: Handles subset relationships
    - partial_ratio: Handles partial matches

    Args:
        text1, text2: Texts to compare
        min_len_threshold: Strings shorter than this use stricter threshold (95 vs 85)

    Returns:
        True if texts are likely duplicates
    """
    # Exact match after normalization
    if text1.lower() == text2.lower():
        return True

    # Use multiple similarity metrics
    token_sort = fuzz.token_sort_ratio(text1, text2)
    token_set = fuzz.token_set_ratio(text1, text2)
    partial = fuzz.partial_ratio(text1, text2)

    # Short strings need higher threshold to prevent false positives
    min_len = min(len(text1), len(text2))
    threshold = 95 if min_len < min_len_threshold else 85

    # Require high similarity on at least 2 metrics for robustness
    scores = [token_sort, token_set, partial]
    return sum(s >= threshold for s in scores) >= 2


# ------------------------------
# Acronym Validation
# ------------------------------


def is_likely_acronym_pair(short_text: str, long_text: str) -> bool:
    """Check if short_text could plausibly be an acronym of long_text.

    Args:
        short_text: Potential acronym
        long_text: Potential long form

    Returns:
        True if short_text could be an acronym of long_text
    """
    if len(short_text) > len(long_text):
        return False

    # Normalize
    short = short_text.strip().upper()
    long = long_text.strip().upper()

    # Extract initials from long form (filter common stop words)
    stop_words = {"OF", "THE", "AND", "FOR", "A", "AN", "IN", "ON", "AT", "TO", "BY"}
    words = [w for w in long.split() if w and w not in stop_words]

    if not words:
        return False

    # Check if short matches initials
    initials = "".join(w[0] for w in words)

    # Allow flexible matching (short can be subset or prefix of initials)
    return short in initials or initials.startswith(short) or short == initials


# ------------------------------
# Overlap Detection
# ------------------------------


def filter_overlapping_mentions(mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove overlapping mentions from the same source/page.

    When mentions overlap in their text spans, keep the one with:
    1. Higher confidence score
    2. Longer text (if scores are similar)
    3. Better label quality

    Args:
        mentions: List of mention dicts with start/end indices

    Returns:
        Filtered list with overlaps removed
    """
    if not mentions:
        return []

    # Group by (source, page)
    groups = defaultdict(list)
    for i, m in enumerate(mentions):
        # Convert page to tuple if it's a list (for hashing)
        page = m.get("page")
        if isinstance(page, list):
            page = tuple(page)
        key = (m.get("source"), page)
        groups[key].append((i, m))

    keep_indices = set()

    for group in groups.values():
        # Sort by start position
        sorted_group = sorted(group, key=lambda x: x[1].get("start", 0) or 0)

        # Track which indices in sorted_group to keep
        local_keep = []

        for idx, mention in sorted_group:
            # Check if this overlaps with any kept mention
            overlaps = False
            for kept_idx, kept_mention in local_keep:
                start1, end1 = mention.get("start") or 0, mention.get("end") or 0
                start2, end2 = kept_mention.get("start") or 0, kept_mention.get("end") or 0

                # Check overlap
                if not (end1 <= start2 or end2 <= start1):  # Overlapping
                    score1 = max(as_score_list(mention.get("score", 0)))
                    score2 = max(as_score_list(kept_mention.get("score", 0)))

                    # Current mention is better - replace the kept one
                    if score1 > score2 + 0.05:
                        local_keep.remove((kept_idx, kept_mention))
                        continue
                    elif abs(score1 - score2) <= 0.05:  # Similar scores
                        # Compare length
                        if len(mention["text"]) > len(kept_mention["text"]):
                            local_keep.remove((kept_idx, kept_mention))
                            continue
                        elif len(mention["text"]) == len(kept_mention["text"]):
                            # Compare label quality
                            if label_rank(mention.get("label", "")) < label_rank(
                                kept_mention.get("label", "")
                            ):
                                local_keep.remove((kept_idx, kept_mention))
                                continue

                    overlaps = True
                    break

            if not overlaps:
                local_keep.append((idx, mention))

        keep_indices.update(idx for idx, _ in local_keep)

    return [mentions[i] for i in sorted(keep_indices)]


# ------------------------------
# Extraction
# ------------------------------


def extract_mentions(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract raw dataset mentions with metadata from records.

    Args:
        records: List of dicts from extract_from_text/extract_from_document

    Returns:
        List of mention dicts
    """
    mentions = []
    for rec in records:
        datasets = rec.get("datasets")
        if not datasets:
            continue

        # Normalize: handle both dict and list
        if isinstance(datasets, dict):
            datasets = [datasets]

        for ds_entry in datasets:
            # Handle nested dataset_name structure
            dataset_name = ds_entry.get("dataset_name", "")
            if isinstance(dataset_name, dict):
                text = clean_text(dataset_name.get("text", ""))
                start = dataset_name.get("start")
                end = dataset_name.get("end")
                # Confidence might be in dataset_name dict when include_confidence=True
                confidence_from_name = dataset_name.get("confidence")
            else:
                text = clean_text(dataset_name)
                start = ds_entry.get("start")
                end = ds_entry.get("end")
                confidence_from_name = None

            if not text:
                continue

            # Extract dataset_tag (was label)
            dataset_tag_field = ds_entry.get("dataset_tag", ds_entry.get("label", ""))
            if isinstance(dataset_tag_field, dict):
                dataset_tag = dataset_tag_field.get("text", "")
            else:
                dataset_tag = dataset_tag_field

            # Extract confidence (was score) - check multiple sources
            # Priority: dataset_name.confidence > confidence field > score field
            if confidence_from_name is not None:
                confidence = confidence_from_name
            else:
                confidence_field = ds_entry.get("confidence", ds_entry.get("score", 0))
                if isinstance(confidence_field, dict):
                    confidence = confidence_field.get(
                        "confidence", confidence_field.get("score", 0)
                    )
                else:
                    confidence = confidence_field

            # Extract description
            description_field = ds_entry.get("description", "")
            if isinstance(description_field, dict):
                description = description_field.get("text", "")
            else:
                description = description_field

            # Extract other schema fields
            producer_field = ds_entry.get("producer", "")
            if isinstance(producer_field, dict):
                producer = producer_field.get("text", "")
            else:
                producer = producer_field

            author_field = ds_entry.get("author", "")
            if isinstance(author_field, dict):
                author = author_field.get("text", "")
            else:
                author = author_field

            geography_field = ds_entry.get("geography", "")
            if isinstance(geography_field, dict):
                geography = geography_field.get("text", "")
            else:
                geography = geography_field

            publication_year_field = ds_entry.get("publication_year", "")
            if isinstance(publication_year_field, dict):
                publication_year = publication_year_field.get("text", "")
            else:
                publication_year = publication_year_field

            reference_year_field = ds_entry.get("reference_year", "")
            if isinstance(reference_year_field, dict):
                reference_year = reference_year_field.get("text", "")
            else:
                reference_year = reference_year_field

            reference_population_field = ds_entry.get("reference_population", "")
            if isinstance(reference_population_field, dict):
                reference_population = reference_population_field.get("text", "")
            else:
                reference_population = reference_population_field

            is_used_field = ds_entry.get("is_used", "")
            if isinstance(is_used_field, dict):
                is_used = is_used_field.get("text", "")
            else:
                is_used = is_used_field

            usage_context_field = ds_entry.get("usage_context", "")
            if isinstance(usage_context_field, dict):
                usage_context = usage_context_field.get("text", "")
            else:
                usage_context = usage_context_field

            ds = {
                "text": text,
                "dataset_tag": dataset_tag,
                "confidence": confidence,
                "description": description,
                "producer": producer,
                "author": author,
                "geography": geography,
                "publication_year": publication_year,
                "reference_year": reference_year,
                "reference_population": reference_population,
                "is_used": is_used,
                "usage_context": usage_context,
                "start": start,
                "end": end,
                "source": rec.get("source"),
                "page": rec.get("page"),
                "raw_context": rec.get("text", ""),
            }

            # Extract relation fields (acronym, author, etc.)
            for field in RELATION_META_FIELDS:
                vals = []

                # Handle acronym from dataset_name structure
                if field == "acronym" and isinstance(dataset_name, dict):
                    acronym_list = dataset_name.get("acronym", [])
                    if isinstance(acronym_list, list):
                        for acr in acronym_list:
                            if isinstance(acr, dict):
                                acr_text = clean_text(acr.get("text", ""))
                            else:
                                acr_text = clean_text(acr)
                            if acr_text:
                                vals.append(acr_text)

                # Also check relations list
                for r in rec.get("relations", []):
                    if r.get("relation", "").lower().replace("_", " ") == field.lower():
                        if clean_text(r.get("source", "")) == text:
                            vals.append(clean_text(r["target"]))

                if vals:
                    ds[field] = list(set(vals))  # Remove duplicates

            mentions.append(ds)

    return mentions


# ------------------------------
# Clustering
# ------------------------------


def build_acronym_clusters(mentions: List[Dict[str, Any]]) -> List[List[int]]:
    """Cluster mentions by acronym ↔ longform relations with validation.

    Args:
        mentions: List of mention dicts

    Returns:
        List of clusters (each cluster is a list of indices)
    """
    n = len(mentions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    # Index by normalized text
    text2idx = defaultdict(list)
    for i, m in enumerate(mentions):
        text2idx[clean_text(m["text"])].append(i)

    # Cluster identical normalized texts
    for idxs in text2idx.values():
        if len(idxs) > 1:
            first = idxs[0]
            for j in idxs[1:]:
                union(first, j)

    # Cluster acronym relations with validation
    for i, m in enumerate(mentions):
        for acr in m.get("acronym", []):
            acr_clean = clean_text(acr)
            for j in text2idx.get(acr_clean, []):
                # Validate that this is a plausible acronym pair
                text_i = m["text"]
                text_j = mentions[j]["text"]

                if is_likely_acronym_pair(text_i, text_j) or is_likely_acronym_pair(text_j, text_i):
                    union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    return list(clusters.values())


def fuzzy_clusters(
    mentions: List[Dict[str, Any]], indices: List[int], threshold: int = 85
) -> List[List[int]]:
    """Cluster near-duplicate mentions using improved fuzzy matching.

    Args:
        mentions: List of all mentions
        indices: Indices to cluster
        threshold: Minimum similarity threshold (not directly used, adaptive in are_fuzzy_duplicates)

    Returns:
        List of clusters
    """
    used = set()
    out = []

    for i in indices:
        if i in used:
            continue
        grp = [i]
        used.add(i)
        ti = mentions[i]["text"]

        for j in indices:
            if j in used:
                continue
            tj = mentions[j]["text"]

            if are_fuzzy_duplicates(ti, tj):
                grp.append(j)
                used.add(j)

        out.append(grp)

    return out


# ------------------------------
# Merging
# ------------------------------


def choose_canonical(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick canonical mention from group by dataset_tag priority > length > confidence."""
    return sorted(
        group,
        key=lambda x: (
            label_rank(x.get("dataset_tag", "")),
            -len(x["text"]),
            -max(as_score_list(x.get("confidence", 0))),
        ),
    )[0]


def merge_cluster(cluster_idxs: List[int], mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a cluster of mentions into a single deduplicated entry.

    Args:
        cluster_idxs: Indices of mentions to merge
        mentions: List of all mentions

    Returns:
        Merged dict with canonical form and structured occurrences
    """
    group = [mentions[i] for i in cluster_idxs]
    canonical = choose_canonical(group)

    merged = {
        "dataset_name": canonical["text"],  # Schema field
        "dataset_tag": canonical.get("dataset_tag"),
        "confidence": as_score_list(canonical.get("confidence", 0)),
        "count": len(group),
        "form_counts": dict(Counter(m["text"] for m in group)),
    }

    # Merge description and other schema fields (take from canonical or merge unique)
    def merge_single_field(field: str):
        """Merge non-list fields - prefer canonical, fallback to first non-empty."""
        val = canonical.get(field)
        if val:
            return val
        # Try to find non-empty value from group
        for m in group:
            v = m.get(field)
            if v:
                return v
        return ""

    merged["description"] = merge_single_field("description")
    merged["producer"] = merge_single_field("producer")
    merged["author"] = merge_single_field("author")
    merged["geography"] = merge_single_field("geography")
    merged["publication_year"] = merge_single_field("publication_year")
    merged["reference_year"] = merge_single_field("reference_year")
    merged["reference_population"] = merge_single_field("reference_population")
    merged["is_used"] = merge_single_field("is_used")
    merged["usage_context"] = merge_single_field("usage_context")

    # Build structured occurrences with context
    occurrences = []
    for m in group:
        # Extract context window around this mention
        context = ""
        raw_text = m.get("raw_context", "")
        start_idx = m.get("start")
        end_idx = m.get("end")

        if raw_text and start_idx is not None and end_idx is not None:
            # The start/end indices should point to the actual text in raw_context
            # Extract context window around this mention
            context = extract_context_window(raw_text, start_idx, end_idx, sentences=1)

        occurrence = {
            "text": m["text"],
            "start": m.get("start"),
            "end": m.get("end"),
            "confidence": max(as_score_list(m.get("confidence", 0))),
            "dataset_tag": m.get("dataset_tag"),
            "description": m.get("description", ""),
            "author": m.get("author", ""),
            "geography": m.get("geography", ""),
            "source": m.get("source"),
            "page": m.get("page"),
            "context": context,
        }
        occurrences.append(occurrence)

    merged["occurrences"] = occurrences

    # Merge metadata fields
    def merge_field(field: str):
        vals = []
        for m in group:
            v = m.get(field)
            if not v:
                continue
            if isinstance(v, list):
                vals.extend(v)
            else:
                vals.append(v)
        return sorted(set(str(v) for v in vals), key=str) if vals else []

    # Always merge acronyms
    merged["acronym"] = merge_field("acronym")

    # Merge all relation metadata fields
    for f in RELATION_META_FIELDS:
        if f != "acronym":  # Already handled
            merged[f] = merge_field(f)

    # Merge page/source lists for summary
    merged["pages"] = merge_field("page")
    merged["sources"] = merge_field("source")

    return merged


# ------------------------------
# Cross-Cluster Validation
# ------------------------------


def merge_similar_clusters(
    merged_clusters: List[Dict[str, Any]], threshold: int = 90
) -> List[Dict[str, Any]]:
    """Merge clusters that likely refer to the same dataset.

    After initial clustering, check if different clusters have similar
    canonical texts and should be merged.

    Args:
        merged_clusters: List of already-merged cluster dicts
        threshold: Similarity threshold for merging (not directly used, adaptive)

    Returns:
        Further merged list
    """
    if len(merged_clusters) <= 1:
        return merged_clusters

    # Build similarity graph
    n = len(merged_clusters)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            text_i = merged_clusters[i].get("dataset_name", merged_clusters[i].get("text", ""))
            text_j = merged_clusters[j].get("dataset_name", merged_clusters[j].get("text", ""))

            # Use fuzzy matching or acronym check to check similarity
            if (
                are_fuzzy_duplicates(text_i, text_j)
                or is_likely_acronym_pair(text_i, text_j)
                or is_likely_acronym_pair(text_j, text_i)
            ):
                union(i, j)

    # Group by parent
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(merged_clusters[i])

    # Merge each group
    final = []
    for group in groups.values():
        if len(group) == 1:
            final.append(group[0])
        else:
            # Merge multiple clusters
            merged = merge_multiple_clusters(group)
            final.append(merged)

    return final


def merge_multiple_clusters(clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple cluster dicts into one.

    Args:
        clusters: List of cluster dicts to merge

    Returns:
        Single merged cluster dict
    """
    # Choose best canonical text
    canonical = max(
        clusters,
        key=lambda c: (
            -label_rank(c.get("dataset_tag", "")),
            len(c.get("dataset_name", c.get("text", ""))),
            max(as_score_list(c.get("confidence", 0))),
        ),
    )

    merged = {
        "dataset_name": canonical.get("dataset_name", canonical.get("text", "")),
        "dataset_tag": canonical.get("dataset_tag"),
        "confidence": canonical.get("confidence"),
        "count": sum(c.get("count", 0) for c in clusters),
        "description": canonical.get("description", ""),
        "producer": canonical.get("producer", ""),
        "author": canonical.get("author", ""),
        "geography": canonical.get("geography", ""),
        "publication_year": canonical.get("publication_year", ""),
        "reference_year": canonical.get("reference_year", ""),
        "reference_population": canonical.get("reference_population", ""),
        "is_used": canonical.get("is_used", ""),
        "usage_context": canonical.get("usage_context", ""),
    }

    # Merge form counts
    form_counts = Counter()
    for c in clusters:
        form_counts.update(c.get("form_counts", {}))
    merged["form_counts"] = dict(form_counts)

    # Merge occurrences
    all_occurrences = []
    for c in clusters:
        all_occurrences.extend(c.get("occurrences", []))
    merged["occurrences"] = all_occurrences

    # Merge metadata lists
    def merge_all(field: str):
        vals = []
        for c in clusters:
            v = c.get(field, [])
            if isinstance(v, list):
                vals.extend(v)
            elif v:
                vals.append(v)
        return sorted(set(str(v) for v in vals), key=str) if vals else []

    for field in ["acronym", "pages", "sources"]:
        merged[field] = merge_all(field)

    # Merge relation fields
    for f in RELATION_META_FIELDS:
        if f != "acronym":  # Already handled
            merged[f] = merge_all(f)

    return merged


# ------------------------------
# Main Pipeline
# ------------------------------


def deduplicate_pipeline(
    records: List[Dict[str, Any]], fuzzy_threshold: int = 85
) -> List[Dict[str, Any]]:
    """Deduplicate dataset mentions with improved clustering (legacy function).

    Note: This is the internal pipeline. Use deduplicate_extraction() for the public API.

    Args:
        records: List of extraction result dicts
        fuzzy_threshold: Minimum similarity for fuzzy matching (default: 85)

    Returns:
        List of deduplicated dataset dicts with structured occurrences
    """
    mentions = extract_mentions(records)

    if not mentions:
        return []

    # Step 0: Filter overlapping mentions from same source/page
    mentions = filter_overlapping_mentions(mentions)

    if not mentions:
        return []

    # Step 1: Build acronym clusters with validation
    acronym_cls = build_acronym_clusters(mentions)
    clustered_idxs = {i for cl in acronym_cls for i in cl}

    # Step 2: Fuzzy clusters for leftovers using improved matching
    leftover = [i for i in range(len(mentions)) if i not in clustered_idxs]
    fuzzy_cls = fuzzy_clusters(mentions, leftover, threshold=fuzzy_threshold) if leftover else []

    # Step 3: Merge clusters
    merged = [merge_cluster(cl, mentions) for cl in acronym_cls + fuzzy_cls]

    # Step 4: Cross-cluster validation (merge similar clusters)
    merged = merge_similar_clusters(merged, threshold=fuzzy_threshold)

    # Step 5: Sort by frequency
    merged.sort(key=lambda m: m.get("count", 0), reverse=True)

    return merged


# ------------------------------
# Public API
# ------------------------------


def deduplicate_extraction(
    extraction_result: Any,
    fuzzy_threshold: int = 85,
) -> Dict[str, Any]:
    """Deduplicate dataset mentions from extraction output.

    Automatically detects whether the input is from extract_from_text or
    extract_from_document and applies the appropriate deduplication strategy.

    Args:
        extraction_result: Output from extract_from_text or extract_from_document
            - From extract_from_text: {"input_text": str, "datasets": [...]}
            - From extract_from_document: [{"input_text": str, "datasets": [...], "document": {...}}, ...]
        fuzzy_threshold: Minimum similarity for fuzzy matching (default: 85)

    Returns:
        Dict with deduplicated datasets and metadata:
        - For text input: {"input_text": str, "datasets": List[deduplicated]}
        - For document input: {"source": str, "datasets": List[deduplicated],
                              "document_metadata": {...}}

    Examples:
        >>> # Text-level deduplication (auto-detected)
        >>> result = extractor.extract_from_text(text)
        >>> dedup = deduplicate_extraction(result)

        >>> # Document-level deduplication (auto-detected)
        >>> doc_results = extractor.extract_from_document("report.pdf")
        >>> dedup = deduplicate_extraction(doc_results)
    """
    # Auto-detect input type
    if (
        isinstance(extraction_result, dict)
        and "datasets" in extraction_result
        and "input_text" in extraction_result
    ):
        # Text-level input from extract_from_text
        level = "text"
    elif isinstance(extraction_result, list):
        # Document-level input from extract_from_document
        level = "document"
    else:
        # Try to infer from structure
        if isinstance(extraction_result, dict) and "datasets" in extraction_result:
            level = "text"
        else:
            raise ValueError(
                "Cannot auto-detect input type. Expected output from extract_from_text "
                "(dict with 'datasets' and 'input_text' keys) or extract_from_document "
                "(list of page results)."
            )

    if level == "text":
        # Handle extract_from_text output: {"input_text": str, "datasets": [...]}
        if not isinstance(extraction_result, dict) or "datasets" not in extraction_result:
            raise ValueError(
                "Invalid extraction_result format for level='text'. Expected dict with 'datasets' key."
            )

        datasets = extraction_result.get("datasets", [])
        if not datasets:
            return extraction_result

        # Convert to records format
        records = [
            {
                "text": extraction_result.get("input_text", ""),
                "source": "text_input",
                "page": None,
                "datasets": datasets,
            }
        ]

        # Run deduplication
        deduplicated = deduplicate_pipeline(records, fuzzy_threshold=fuzzy_threshold)

        return {"input_text": extraction_result.get("input_text", ""), "datasets": deduplicated}

    elif level == "document":
        # Handle extract_from_document output: List[{"input_text": str, "datasets": [...], "document": {...}}]
        if not isinstance(extraction_result, list):
            raise ValueError(
                "Invalid extraction_result format for level='document'. Expected list of page results."
            )

        if not extraction_result:
            return {"source": None, "datasets": [], "document_metadata": {}}

        # Flatten all results into records format
        records = []
        source = None
        all_pages = []

        for page_result in extraction_result:
            if not isinstance(page_result, dict):
                continue

            page_datasets = page_result.get("datasets", [])
            doc_meta = page_result.get("document", {})

            if source is None:
                source = doc_meta.get("source")

            pages = doc_meta.get("pages", [])
            all_pages.append(pages)

            # Create a record for each page's extraction
            if page_datasets:
                records.append(
                    {
                        "text": page_result.get("input_text", ""),
                        "source": doc_meta.get("source"),
                        "page": pages,
                        "datasets": page_datasets,
                    }
                )

        # Run deduplication across all pages
        deduplicated = deduplicate_pipeline(records, fuzzy_threshold=fuzzy_threshold)

        return {
            "source": source,
            "datasets": deduplicated,
            "document_metadata": {
                "total_pages": len(extraction_result),
                "pages_processed": all_pages,
            },
        }
