"""Dataset mention extractor."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..models.model_manager import ModelManager
from ..schemas.dataset_schema import DatasetSchema
from ..utils.document_parser import DocumentParser


class DatasetExtractor:
    """Extract dataset mentions from text or documents."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        threshold: float = 0.5,
        cache_dir: Optional[str] = None,
        classifier_model: Optional[str] = None,
    ):
        """Initialize dataset extractor.

        Args:
            model_id: HuggingFace model ID or path to local model.
                     If None, uses default model.
            threshold: Default confidence threshold for extraction
            cache_dir: Directory to cache models
            classifier_model: HuggingFace model ID for pre-filtering classifier.
                            If None, uses default: "ai4data-use/bert-base-uncased-data-use"
        """
        self.model_manager = ModelManager(cache_dir=cache_dir)
        self.model_id = model_id
        self.threshold = threshold
        self.classifier_model = classifier_model
        self._model = None
        self._schema = None
        self._classifier = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy load and return the model."""
        if self._model is None:
            self._model = self.model_manager.load(self.model_id)
        return self._model

    @property
    def schema(self):
        """Lazy load and return the schema."""
        if self._schema is None:
            self._schema = DatasetSchema(threshold=self.threshold).build(self.model)
        return self._schema

    @property
    def classifier(self):
        """Lazy load and return the classifier and tokenizer."""
        if self._classifier is None:
            self._classifier, self._tokenizer = self._load_classifier()
        return self._classifier, self._tokenizer

    def _load_classifier(self, model: str = None, device=None):
        """Load the classifier model and tokenizer using HF pipeline.

        Args:
            model: Model ID to use. If None, uses self.classifier_model or default.
            device: Device to use. If None, auto-detects (GPU if available, else CPU).

        Returns:
            Tuple of (pipeline, tokenizer)
        """
        import torch
        from transformers import AutoTokenizer, pipeline

        # Determine device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        # Determine model
        model_id = model or self.classifier_model or "ai4data-use/bert-base-uncased-data-use"

        # Load tokenizer and classifier
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        classifier = pipeline(
            "text-classification", model=model_id, tokenizer=tokenizer, device=device
        )

        return classifier, tokenizer

    def _should_process_chunks(self, chunks, classifier, tokenizer) -> bool:
        """Check if any chunk is classified as containing data.

        Args:
            chunks: List of text chunks to classify
            classifier: HuggingFace text classification pipeline
            tokenizer: Tokenizer for the classifier

        Returns:
            True if any chunk is classified as having data (label != "NO_DATA"),
            False otherwise.
        """
        for chunk in chunks:
            # Tokenize with truncation to handle long chunks
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)

            # Decode back to text (classifier expects raw string)
            truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            # Run classification
            classification = classifier(truncated_text)

            # Check if this chunk has data (assume HF pipeline output format)
            if classification and classification[0].get("label") != "NO_DATA":
                return True

        return False

    def _chunk_text(self, text: str, max_tokens: int = 200, overlap: int = 50) -> List[tuple]:
        """Split text into overlapping chunks to handle token limits.

        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk (default: 200 for safety with 512 limit)
            overlap: Number of tokens to overlap between chunks

        Returns:
            List of tuples (chunk_text, token_offset) where token_offset is the starting
            token position of the chunk in the original tokenized text
        """
        from gliner2.processor import WhitespaceTokenSplitter

        splitter = WhitespaceTokenSplitter()
        tokens = list(splitter(text, lower=False))

        # If text is short enough, return as-is with token offset 0
        if len(tokens) <= max_tokens:
            return [(text, 0)]

        chunks = []
        start_token_idx = 0

        while start_token_idx < len(tokens):
            end_token_idx = min(start_token_idx + max_tokens, len(tokens))

            # Get character positions for extracting chunk text
            chunk_start_char = tokens[start_token_idx][1]  # start position
            chunk_end_char = (
                tokens[end_token_idx - 1][2] if end_token_idx > 0 else len(text)
            )  # end position

            chunk_text = text[chunk_start_char:chunk_end_char]
            # Store chunk with its TOKEN offset (not character offset)
            chunks.append((chunk_text, start_token_idx))

            # Move to next chunk with overlap
            start_token_idx += max_tokens - overlap

        return chunks

    def _merge_chunk_results(self, chunk_results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from multiple chunks.

        Args:
            chunk_results_list: List of result dicts from each chunk

        Returns:
            Merged result dict with all entities (deduplication happens later)
        """
        if not chunk_results_list:
            return {"dataset_mention": []}

        # Collect all entities from all chunks
        all_entities = []

        for chunk_result in chunk_results_list:
            if isinstance(chunk_result, dict):
                entities = chunk_result.get("dataset_mention", chunk_result.get("entities", []))
                all_entities.extend(entities)

        return {"dataset_mention": all_entities}

    def _adjust_entity_indices(self, entity: Dict[str, Any], offset: int) -> None:
        """Adjust start/end indices in entity fields by adding the chunk offset.

        Args:
            entity: Entity dict with potential start/end fields
            offset: Character offset to add to all indices
        """
        # Fields that may contain start/end indices
        fields_with_indices = [
            "dataset_name",
            "description",
            "acronym",
            "author",
            "producer",
            "geography",
            "publication_year",
            "reference_year",
            "reference_population",
            "data_type",
        ]

        for field in fields_with_indices:
            if field in entity and isinstance(entity[field], dict):
                field_data = entity[field]
                if "start" in field_data and isinstance(field_data["start"], int):
                    field_data["start"] += offset
                if "end" in field_data and isinstance(field_data["end"], int):
                    field_data["end"] += offset

    def _is_empty_dataset(self, dataset: Dict[str, Any]) -> bool:
        """Check if a dataset should be filtered out.

        A dataset is considered empty if:
        1. dataset_name is None or empty (required field), OR
        2. All extractive fields are None (excluding classification fields)

        Args:
            dataset: Dataset dictionary to check

        Returns:
            True if dataset should be filtered out, False otherwise
        """
        # Check if dataset_name is None or empty (this is a required field)
        dataset_name = dataset.get("dataset_name")
        if dataset_name is None:
            return True
        if isinstance(dataset_name, dict):
            if dataset_name.get("text") is None or not dataset_name.get("text", "").strip():
                return True
        elif isinstance(dataset_name, str):
            if not dataset_name.strip():
                return True

        # Additionally check if all other extractive fields are None
        # Fields to check (excluding classification fields like dataset_tag, is_used, usage_context)
        extractive_fields = [
            "description",
            "data_type",
            "acronym",
            "author",
            "producer",
            "geography",
            "publication_year",
            "reference_year",
            "reference_population",
        ]

        all_empty = True
        for field in extractive_fields:
            value = dataset.get(field)
            # If the field exists and is not None (could be a string or dict with text)
            if value is not None:
                # Handle dict format (with text, start, end)
                if isinstance(value, dict):
                    if value.get("text") is not None and value.get("text", "").strip():
                        all_empty = False
                        break
                # Handle string format
                elif isinstance(value, str) and value.strip():
                    all_empty = False
                    break

        return all_empty

    def _deduplicate_datasets(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate datasets based on start/end indices.

        Deduplication strategy:
        1. First pass: exact match on (dataset_name text, start index, end index)
        2. Second pass: merge entries with overlapping index ranges, prefer longer name

        Args:
            datasets: List of dataset dictionaries

        Returns:
            Deduplicated list of datasets
        """
        if not datasets:
            return []

        # --- First pass: exact match deduplication ---
        seen_keys = {}
        exact_deduplicated = []

        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue

            # Extract dataset_name and indices for deduplication key
            dataset_name = dataset.get("dataset_name")
            if isinstance(dataset_name, dict):
                name_text = dataset_name.get("text")
                name_start = dataset_name.get("start")
                name_end = dataset_name.get("end")
            else:
                name_text = dataset_name
                name_start = None
                name_end = None

            # Create deduplication key
            key = (name_text, name_start, name_end)

            # Skip if all key components are None
            if all(v is None for v in key):
                exact_deduplicated.append(dataset)
                continue

            # Check if we've seen this key before
            if key in seen_keys:
                # Keep the dataset with more non-None fields
                existing_idx = seen_keys[key]
                existing_dataset = exact_deduplicated[existing_idx]

                # Count non-None fields in both datasets
                current_count = sum(1 for v in dataset.values() if v is not None and v != "")
                existing_count = sum(
                    1 for v in existing_dataset.values() if v is not None and v != ""
                )

                # Replace if current has more fields
                if current_count > existing_count:
                    exact_deduplicated[existing_idx] = dataset
            else:
                # New key, add to results
                seen_keys[key] = len(exact_deduplicated)
                exact_deduplicated.append(dataset)

        # --- Second pass: overlapping indices deduplication ---
        # For entries with valid indices, merge those with overlapping ranges
        return self._merge_overlapping_datasets(exact_deduplicated)

    def _merge_overlapping_datasets(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge datasets with overlapping index ranges.

        When indices overlap, prefer the entry with the longer dataset name
        (more complete extraction). If same length, prefer more populated fields.

        Args:
            datasets: List of dataset dictionaries (already exact-deduplicated)

        Returns:
            List with overlapping entries merged
        """
        if not datasets:
            return []

        # Extract indexed info for comparison
        indexed_items = []
        no_index_items = []

        for dataset in datasets:
            dataset_name = dataset.get("dataset_name")
            if isinstance(dataset_name, dict):
                name_text = dataset_name.get("text", "")
                name_start = dataset_name.get("start")
                name_end = dataset_name.get("end")
            else:
                name_text = dataset_name or ""
                name_start = None
                name_end = None

            if name_start is not None and name_end is not None:
                indexed_items.append(
                    {
                        "name_text": name_text,
                        "start": name_start,
                        "end": name_end,
                        "dataset": dataset,
                    }
                )
            else:
                no_index_items.append(dataset)

        # Sort by start index for efficient overlap detection
        indexed_items.sort(key=lambda x: x["start"])

        # Merge overlapping entries
        merged = []
        for item in indexed_items:
            start = item["start"]
            end = item["end"]
            name_text = item["name_text"]
            dataset = item["dataset"]

            # Check for overlap with existing entries
            overlap_idx = None
            for i, existing in enumerate(merged):
                ex_start = existing["start"]
                ex_end = existing["end"]

                # Overlap: ranges intersect if start <= ex_end and end >= ex_start
                if start <= ex_end and end >= ex_start:
                    overlap_idx = i
                    break

            if overlap_idx is not None:
                existing = merged[overlap_idx]
                # Prefer longer name (more complete extraction)
                if len(name_text) > len(existing["name_text"]):
                    merged[overlap_idx] = item
                elif len(name_text) == len(existing["name_text"]):
                    # Same length: prefer more populated fields
                    current_count = sum(1 for v in dataset.values() if v is not None and v != "")
                    existing_count = sum(
                        1 for v in existing["dataset"].values() if v is not None and v != ""
                    )
                    if current_count > existing_count:
                        merged[overlap_idx] = item
            else:
                merged.append(item)

        # Combine results: merged items + items without indices
        result = [item["dataset"] for item in merged]
        result.extend(no_index_items)

        return result

    def extract_from_text(
        self,
        text: str,
        include_confidence: bool = False,
        custom_schema: Optional[Any] = None,
        exclude_non_datasets: bool = True,
        dataset_threshold: Optional[float] = None,
        max_tokens: int = 200,
        model_id: Optional[str] = None,
        enable_chunking: bool = True,
        use_classifier: bool = False,
    ) -> Dict[str, Any]:
        """Extract dataset mentions from text.

        Args:
            text: Input text to extract from
            include_confidence: Whether to include confidence scores
            custom_schema: Optional custom schema to use instead of default
            exclude_non_datasets: If True, filter out datasets with dataset_tag="non-dataset"
            dataset_threshold: Optional confidence threshold for dataset_name field (0.0-1.0).
                             If provided, overrides the default threshold for dataset extraction.
            max_tokens: Maximum tokens per chunk for long texts (default: 200).
                       Lower values create smaller chunks, higher values process more text at once.
            model_id: Optional model ID to use for this specific extraction.
                     If provided, overrides the instance's model.
            enable_chunking: Whether to split long text into chunks (default: True).
                            If False, processes entire text at once.
            use_classifier: Whether to use pre-filtering classifier to check if text contains
                           dataset mentions before extraction (default: False).

        Returns:
            Dict with 'input_text' and 'datasets' keys containing the original text
            and list of extracted dataset mentions with indices relative to original text.
            Datasets with all None values (excluding classification fields) are automatically
            filtered out.
        """
        # Determine model to use
        if model_id is not None:
            model = self.model_manager.load(model_id)
        else:
            model = self.model

        # Build schema with custom threshold if specified
        if dataset_threshold is not None and custom_schema is None:
            from ..schemas.dataset_schema import DatasetSchema

            # Use the resolved model for schema building
            custom_schema = (
                DatasetSchema().set_threshold("dataset_name", dataset_threshold).build(model)
            )

        # If no custom schema provided and no threshold override, verify schema against model
        if custom_schema is None:
            if model_id is not None:
                # We need to rebuild schema for the specific model if utilizing a different model
                from ..schemas.dataset_schema import DatasetSchema

                schema = DatasetSchema(threshold=self.threshold).build(model)
            else:
                schema = self.schema
        else:
            schema = custom_schema

        # Chunk text determines how we process
        if enable_chunking:
            chunks_with_offsets = self._chunk_text(text, max_tokens=max_tokens, overlap=50)
        else:
            # Process entire text as one chunk
            chunks_with_offsets = [(text, 0)]

        # Pre-filtering: check if any chunk contains dataset mentions
        if use_classifier:
            classifier, tokenizer = self.classifier
            chunks_text = [chunk[0] for chunk in chunks_with_offsets]

            if not self._should_process_chunks(chunks_text, classifier, tokenizer):
                # No dataset mentions detected, return empty result
                return {"input_text": text, "datasets": [], "prefiltered": True}

        if len(chunks_with_offsets) == 1:
            # Text is short enough, process directly (offset is 0)
            chunk_text, _ = chunks_with_offsets[0]
            results = model.extract(chunk_text, schema, include_confidence=include_confidence)
        else:
            # Process each chunk and merge results
            chunk_results = []
            for chunk_text, chunk_offset in chunks_with_offsets:
                chunk_result = model.extract(
                    chunk_text, schema, include_confidence=include_confidence
                )

                # Adjust indices in this chunk's results
                if isinstance(chunk_result, dict):
                    entities = chunk_result.get("dataset_mention", chunk_result.get("entities", []))
                    for entity in entities:
                        if isinstance(entity, dict):
                            self._adjust_entity_indices(entity, chunk_offset)

                chunk_results.append(chunk_result)

            # Merge results from all chunks
            results = self._merge_chunk_results(chunk_results)

        # Extract dataset list from results
        if isinstance(results, dict):
            datasets = results.get("dataset_mention", results.get("entities", []))
        else:
            datasets = results if isinstance(results, list) else []

        # Post-processing: filter datasets based on rules
        filtered_datasets = []
        for dataset in datasets:
            if isinstance(dataset, dict):
                # Rule 1: Always remove if dataset_name is None or empty
                dataset_name = dataset.get("dataset_name")
                if dataset_name is None:
                    continue
                if isinstance(dataset_name, dict):
                    if dataset_name.get("text") is None or not dataset_name.get("text", "").strip():
                        continue
                elif isinstance(dataset_name, str):
                    if not dataset_name.strip():
                        continue

                # Rule 2: Check non-dataset tag (only if dataset_name exists)
                dataset_tag = dataset.get("dataset_tag")
                # Handle both string and dict formats
                if isinstance(dataset_tag, dict):
                    tag_value = dataset_tag.get("text")
                else:
                    tag_value = dataset_tag

                # Skip non-datasets if flag is set
                if exclude_non_datasets and tag_value == "non-dataset":
                    continue

                filtered_datasets.append(dataset)

        # Deduplicate datasets based on start/end indices
        filtered_datasets = self._deduplicate_datasets(filtered_datasets)

        # Return in the requested format
        result = {"input_text": text, "datasets": filtered_datasets}

        # Add prefiltered metadata if classifier was used
        if use_classifier:
            result["prefiltered"] = False  # Classifier checked but extraction proceeded

        return result

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
        use_classifier: bool = True,
        skip_references: bool = True,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract dataset mentions from a PDF document.

        Args:
            source: Path to PDF file or URL
            include_confidence: Whether to include confidence scores
            custom_schema: Optional custom schema to use instead of default
            n_pages: Number of pages per chunk (default: 1 for page-by-page processing)
            include_metadata: Whether to include source document and page text metadata
            exclude_non_datasets: If True, filter out datasets with dataset_tag="non-dataset"
            dataset_threshold: Optional confidence threshold for dataset_name field (0.0-1.0).
                             If provided, overrides the default threshold for dataset extraction.
            max_tokens: Maximum tokens per chunk for long texts (default: 200).
            use_classifier: Whether to use pre-filtering classifier to check if text contains
                           dataset mentions before extraction (default: True for documents).
            skip_references: If True, automatically skip pages in references/appendix sections
                           after the halfway point of the document (default: True).
            verbose: If True, print logging messages when references are detected and skipped.

        Returns:
            List of extracted dataset mentions with metadata including page numbers,
            source document, and page text (if include_metadata=True).
            Datasets with all None values (excluding classification fields) are automatically
            filtered out.
        """
        # Convert source to string for metadata
        source_str = str(source)

        # Load PDF in chunks with page tracking
        chunks = DocumentParser.load_pdf_chunks(
            source_str, n_pages=n_pages, skip_references=skip_references, verbose=verbose
        )

        # Extract from each chunk and aggregate results
        all_results = []
        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_pages = chunk["pages"]

            # Extract from this chunk (returns dict with 'input_text' and 'datasets')
            extraction_result = self.extract_from_text(
                chunk_text,
                include_confidence,
                custom_schema,
                exclude_non_datasets,
                dataset_threshold,
                max_tokens,
                use_classifier=use_classifier,
            )

            input_text = extraction_result["input_text"]
            datasets_extracted = extraction_result["datasets"]
            document_metadata = {"source": source_str, "pages": chunk_pages}
            all_results.append(
                {
                    "input_text": input_text,
                    "datasets": datasets_extracted,
                    "document": document_metadata,
                }
            )

        return all_results

    def extract_batch(
        self, texts: List[str], include_confidence: bool = True, custom_schema: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Extract dataset mentions from multiple texts.

        Args:
            texts: List of input texts
            include_confidence: Whether to include confidence scores
            custom_schema: Custom schema to use instead of default

        Returns:
            List of dicts, each containing 'input_text' and 'datasets' for each input text
        """
        results = []
        for text in texts:
            result = self.extract_from_text(text, include_confidence, custom_schema)
            results.append(result)
        return results
