"""Main orchestration class for data dictionary augmentation.

``DataDictionaryAugmentor`` implements a five-step pipeline:
1. Load: ingest variables from CSV, JSON, dict, or NADA catalog
2. Embed: encode variable labels/descriptions with sentence-transformers
3. Cluster: group semantically similar variables
4. Generate: call an LLM for each cluster to curate a DDI variable group
5. Export: write the augmented dictionary to disk or return as a DataFrame

The LLM backend is provider-agnostic via ``litellm``: set ``model`` to any
litellm-supported model string (e.g., ``"claude-sonnet-4-6"``,
``"gpt-4o-mini"``, ``"gemini/gemini-2.0-flash"``).

Install requirements: ``uv pip install ai4data[metadata]``
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from .adapters import (
    ConfigurableDictionaryAdapter,
    DictionaryAdapter,
    NADACatalogAdapter,
)
from .clustering import (
    DEFAULT_MAX_CLUSTER_TOKENS,
    DEFAULT_N_CLUSTERS_RANGE,
    DEFAULT_SVD_THRESHOLD,
    build_cluster_map,
    cluster_variables,
    split_clusters_for_token_budget,
    reduce_dimensions,
)
from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingEncoder
from .prompts import (
    SYSTEM_PROMPT,
    get_json_object_format,
    get_variable_group_response_format,
    render_user_prompt,
)
from .qa import (
    QA_SYSTEM_PROMPT,
    get_qa_response_format,
    render_qa_user_prompt,
)
from .schemas import (
    AugmentedDictionary,
    DictionaryVariable,
    VariableGroup,
    VariableGroupAssignment,
    VariableGroupCurationResult,
    VariableGroupQAResult,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass
class _ClusterProcessResult:
    cluster_id: int
    group: VariableGroup
    assignments: List[VariableGroupAssignment]
    n_qa_passed: int = 0
    n_qa_failed: int = 0
    n_qa_skipped: int = 0


class DataDictionaryAugmentor:
    """LLM-powered data dictionary augmentation.

    Generates DDI-style variable groups for microdata or administrative data
    dictionary variables using semantic clustering and LLM-elicited curation.

    Parameters
    ----------
    model : str
        LiteLLM model string for curation. Defaults to ``"claude-sonnet-4-6"``.
        Other examples: ``"gpt-4o-mini"``, ``"gemini/gemini-2.0-flash"``.
    qa_model : str, optional
        LiteLLM model string for self-consistency QA. Defaults to ``model``.
    enable_qa : bool
        When True, run a QA agent after each successful curation call.
    embedding_model : str
        SentenceTransformer model name. Defaults to ``"BAAI/bge-small-en-v1.5"``.
    n_clusters : int, optional
        Number of clusters. If not given, estimated automatically via silhouette.
    max_cluster_tokens : int
        Maximum token budget per LLM cluster prompt.
    temperature : float
        LLM temperature. Use 0.0 for deterministic outputs.
    random_state : int
        Random seed for clustering and silhouette estimation.
    max_concurrency : int
        Maximum number of clusters to process in parallel during LLM
        generation. Defaults to ``1`` (sequential). Increase to reduce
        wall-clock time; watch provider rate limits.
    device : str, optional
        Embedding inference device (``"cuda"``, ``"mps"``, ``"cpu"``).
        Auto-detected if not specified.

    Examples
    --------
    One-call convenience:

    >>> augmentor = DataDictionaryAugmentor()
    >>> result = augmentor.augment("variables.csv")
    >>> augmentor.export("augmented.json")

    Step-by-step with custom adapter:

    >>> from ai4data.metadata.augmentation import ConfigurableDictionaryAdapter
    >>> adapter = ConfigurableDictionaryAdapter({"variable_name": "name", "label": "labl"})
    >>> result = (
    ...     DataDictionaryAugmentor()
    ...     .load("dictionary.csv", adapter=adapter)
    ...     .embed(show_progress_bar=True)
    ...     .cluster(n_clusters=10)
    ...     .generate_variable_groups()
    ... )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        qa_model: Optional[str] = None,
        enable_qa: bool = True,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        n_clusters: Optional[int] = None,
        max_cluster_tokens: int = DEFAULT_MAX_CLUSTER_TOKENS,
        temperature: float = 0.0,
        random_state: int = 42,
        max_concurrency: int = 1,
        device: Optional[str] = None,
    ):
        self.model = model
        self.qa_model = qa_model or model
        self.enable_qa = enable_qa
        self.n_clusters = n_clusters
        self.max_cluster_tokens = max_cluster_tokens
        self.temperature = temperature
        self.random_state = random_state
        self.max_concurrency = max_concurrency

        self._encoder = EmbeddingEncoder(embedding_model, device=device)
        self._variables: Optional[List[DictionaryVariable]] = None
        self._embeddings: Optional[Any] = None   # np.ndarray
        self._cluster_labels: Optional[Any] = None  # np.ndarray
        self._cluster_map: Optional[Dict[int, List[DictionaryVariable]]] = None
        self._result: Optional[AugmentedDictionary] = None

    # ----- Step 1: Load ----- #

    def load(
        self,
        source: Union[
            List[DictionaryVariable],
            List[Dict[str, Any]],
            str,
            Path,
        ],
        *,
        adapter: Optional[DictionaryAdapter] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        dataset_id: Optional[str] = None,
    ) -> "DataDictionaryAugmentor":
        """Load variables from various sources.

        Parameters
        ----------
        source : list, str, or Path
            - ``List[DictionaryVariable]``: used directly.
            - ``List[dict]``: converted via adapter or default mapping.
            - ``str`` or ``Path``: loaded from file (.csv or .json).
        adapter : DictionaryAdapter, optional
            Custom adapter. If not given, a default adapter is constructed.
        column_mapping : dict, optional
            Column mapping for ``ConfigurableDictionaryAdapter``. Ignored if
            ``adapter`` is provided.
        dataset_id : str, optional
            Dataset identifier stored in the output metadata.

        Returns
        -------
        DataDictionaryAugmentor
            Self, for method chaining.
        """
        self._dataset_id = dataset_id
        self._result = None  # reset any prior result

        if isinstance(source, list):
            if not source:
                self._variables = []
                return self
            if isinstance(source[0], DictionaryVariable):
                self._variables = list(source)
                return self
            # List of dicts
            _adapter = adapter or ConfigurableDictionaryAdapter(
                column_mapping or None
            )
            self._variables = _adapter.from_records(source)
            return self

        # File-based loading
        path = Path(source)
        if adapter is None:
            if path.suffix.lower() == ".json":
                # Detect NADA vs plain JSON by trying NADACatalogAdapter first
                try:
                    _adapter_inst = NADACatalogAdapter(validate_output=False)
                    vars_ = _adapter_inst.load_json(path)
                    if vars_:
                        self._variables = vars_
                        return self
                except Exception:
                    pass
                # Fall back to plain JSON adapter
                _adapter = ConfigurableDictionaryAdapter(
                    column_mapping or None, validate_output=False
                )
                self._variables = _adapter.load_json(path)
            else:
                _adapter = ConfigurableDictionaryAdapter(
                    column_mapping or None, validate_output=False
                )
                self._variables = _adapter.load_csv(path)
        else:
            if path.suffix.lower() == ".json":
                self._variables = adapter.load_json(path)
            else:
                self._variables = adapter.load_csv(path)

        return self

    # ----- Step 2: Embed ----- #

    def embed(
        self,
        *,
        show_progress_bar: bool = False,
    ) -> "DataDictionaryAugmentor":
        """Compute embeddings for the loaded variables.

        Parameters
        ----------
        show_progress_bar : bool
            Display a tqdm progress bar during encoding.

        Returns
        -------
        DataDictionaryAugmentor
            Self, for method chaining.
        """
        self._require_loaded()
        self._embeddings = self._encoder.encode_variables(
            self._variables, show_progress_bar=show_progress_bar
        )
        # Apply dimensionality reduction for large dictionaries
        self._embeddings = reduce_dimensions(
            self._embeddings,
            threshold=DEFAULT_SVD_THRESHOLD,
            random_state=self.random_state,
        )
        self._cluster_labels = None  # reset downstream state
        self._result = None
        return self

    # ----- Step 3: Cluster ----- #

    def cluster(
        self,
        n_clusters: Optional[int] = None,
    ) -> "DataDictionaryAugmentor":
        """Cluster the embedded variables into semantic groups.

        Parameters
        ----------
        n_clusters : int, optional
            Override the instance-level ``n_clusters`` setting.

        Returns
        -------
        DataDictionaryAugmentor
            Self, for method chaining.
        """
        self._require_embedded()
        k = n_clusters or self.n_clusters
        labels = cluster_variables(
            self._embeddings,
            n_clusters=k,
            n_range=DEFAULT_N_CLUSTERS_RANGE,
            random_state=self.random_state,
        )
        # Enforce token budget per cluster
        labels = split_clusters_for_token_budget(
            labels,
            self._variables,
            max_tokens_per_cluster=self.max_cluster_tokens,
        )
        self._cluster_labels = labels
        self._cluster_map = build_cluster_map(labels, self._variables)
        self._result = None
        return self

    # ----- Step 4: Generate variable groups ----- #

    def generate_variable_groups(
        self,
        *,
        show_progress_bar: bool = False,
        max_concurrency: Optional[int] = None,
    ) -> AugmentedDictionary:
        """Call the LLM for each cluster and assemble the augmented dictionary.

        Makes one curation LLM call per cluster, optionally followed by a
        self-consistency QA call. Failures in individual clusters are logged
        and handled with a fallback ``VariableGroup`` so that partial results
        are always returned.

        Parameters
        ----------
        show_progress_bar : bool
            Display a tqdm progress bar during LLM calls per cluster.
        max_concurrency : int, optional
            Override the instance-level ``max_concurrency`` for this run.
            When greater than 1, clusters are processed in parallel via a
            thread pool.

        Returns
        -------
        AugmentedDictionary
            The complete augmented dictionary with variable groups and assignments.
        """
        self._require_clustered()

        concurrency = (
            max_concurrency if max_concurrency is not None else self.max_concurrency
        )
        items = list(self._cluster_map.items())
        results = self._process_clusters(
            items,
            concurrency=concurrency,
            show_progress_bar=show_progress_bar,
        )

        variable_groups = [r.group for r in results]
        variable_assignments: List[VariableGroupAssignment] = []
        for result in results:
            variable_assignments.extend(result.assignments)

        n_qa_passed = sum(r.n_qa_passed for r in results)
        n_qa_failed = sum(r.n_qa_failed for r in results)
        n_qa_skipped = sum(r.n_qa_skipped for r in results)

        metadata: Dict[str, Any] = {
            "model": self.model,
            "embedding_model": self._encoder.model_name,
            "n_variables": len(self._variables),
            "n_clusters": len(self._cluster_map),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "enable_qa": self.enable_qa,
            "max_concurrency": concurrency,
        }
        if self.enable_qa:
            metadata.update(
                {
                    "qa_model": self.qa_model,
                    "n_qa_passed": n_qa_passed,
                    "n_qa_failed": n_qa_failed,
                    "n_qa_skipped": n_qa_skipped,
                }
            )

        self._result = AugmentedDictionary(
            dataset_id=getattr(self, "_dataset_id", None),
            variable_groups=variable_groups,
            variable_assignments=variable_assignments,
            metadata=metadata,
        )
        return self._result

    # ----- One-call convenience ----- #

    def augment(
        self,
        source: Union[List[DictionaryVariable], List[Dict], str, Path],
        *,
        adapter: Optional[DictionaryAdapter] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        n_clusters: Optional[int] = None,
        dataset_id: Optional[str] = None,
        show_progress_bar: bool = False,
        max_concurrency: Optional[int] = None,
    ) -> AugmentedDictionary:
        """Load → embed → cluster → generate variable groups in one call.

        Parameters
        ----------
        source : list, str, or Path
            Variable source (see ``load()``).
        adapter : DictionaryAdapter, optional
            Custom adapter (see ``load()``).
        column_mapping : dict, optional
            Column mapping for CSV loading (see ``load()``).
        n_clusters : int, optional
            Override cluster count.
        dataset_id : str, optional
            Dataset identifier for output metadata.
        show_progress_bar : bool
            Show tqdm progress bars during embedding and variable group generation.
        max_concurrency : int, optional
            Override parallel LLM concurrency for variable group generation
            (see ``generate_variable_groups()``).

        Returns
        -------
        AugmentedDictionary
        """
        return (
            self.load(
                source,
                adapter=adapter,
                column_mapping=column_mapping,
                dataset_id=dataset_id,
            )
            .embed(show_progress_bar=show_progress_bar)
            .cluster(n_clusters=n_clusters)
            .generate_variable_groups(
                show_progress_bar=show_progress_bar,
                max_concurrency=max_concurrency,
            )
        )

    # ----- Export ----- #

    def export(
        self,
        path: Union[str, Path],
        *,
        indent: int = 2,
    ) -> Path:
        """Write the augmented dictionary to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        indent : int
            JSON indentation level.

        Returns
        -------
        Path
            The resolved output path.
        """
        self._require_result()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(
                json.loads(self._result.model_dump_json()),
                fh,
                indent=indent,
                ensure_ascii=False,
            )
        logger.info("Augmented dictionary written to %s", out)
        return out

    def to_dataframe(self):
        """Return variable assignments as a flat pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per curated variable with columns: ``variable_name``,
            ``vgid``, ``label``, ``cluster_id``.
        """
        self._require_result()
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc
        rows = [
            {
                "variable_name": a.variable_name,
                "vgid": a.vgid,
                "label": a.label,
                "cluster_id": a.cluster_id,
            }
            for a in self._result.variable_assignments
        ]
        return pd.DataFrame(rows)

    # ----- Internal cluster processing ----- #

    def _process_cluster(
        self,
        cluster_id: int,
        vars_: List[DictionaryVariable],
    ) -> _ClusterProcessResult:
        """Run curation (and optional QA) for a single cluster."""
        n_qa_passed = 0
        n_qa_failed = 0
        n_qa_skipped = 0

        curation = self._call_llm_for_cluster(cluster_id, vars_)

        if curation is not None:
            qa_result = (
                self._call_qa_for_curation(cluster_id, vars_, curation)
                if self.enable_qa
                else None
            )
            if qa_result is not None:
                if qa_result.is_self_consistent:
                    n_qa_passed = 1
                else:
                    n_qa_failed = 1
                group = VariableGroup.from_curation(
                    curation,
                    cluster_id=cluster_id,
                    qa=qa_result,
                )
            elif self.enable_qa:
                n_qa_skipped = 1
                group = VariableGroup.from_curation(
                    curation,
                    cluster_id=cluster_id,
                    qa_error="QA call failed",
                )
            else:
                group = VariableGroup.from_curation(
                    curation,
                    cluster_id=cluster_id,
                )
            assigned_names = curation.variables
        else:
            all_names = [v.variable_name for v in vars_]
            group = VariableGroup.uncategorized_fallback(
                cluster_id=cluster_id,
                variable_names=all_names,
            )
            assigned_names = all_names

        assignments = [
            VariableGroupAssignment(
                variable_name=name,
                vgid=group.vgid,
                label=group.label,
                cluster_id=cluster_id,
            )
            for name in assigned_names
        ]

        return _ClusterProcessResult(
            cluster_id=cluster_id,
            group=group,
            assignments=assignments,
            n_qa_passed=n_qa_passed,
            n_qa_failed=n_qa_failed,
            n_qa_skipped=n_qa_skipped,
        )

    def _process_clusters(
        self,
        items: List[tuple[int, List[DictionaryVariable]]],
        *,
        concurrency: int,
        show_progress_bar: bool,
    ) -> List[_ClusterProcessResult]:
        """Process all clusters sequentially or in parallel."""
        if concurrency <= 1:
            if show_progress_bar:
                from tqdm import tqdm

                results: List[_ClusterProcessResult] = []
                with tqdm(
                    items,
                    total=len(items),
                    desc="Variable groups",
                ) as pbar:
                    for cluster_id, vars_ in pbar:
                        result = self._process_cluster(cluster_id, vars_)
                        results.append(result)
                        pbar.set_postfix(
                            cluster=result.cluster_id,
                            label=result.group.label[:30],
                        )
                return results

            return [
                self._process_cluster(cluster_id, vars_)
                for cluster_id, vars_ in items
            ]

        if show_progress_bar:
            from tqdm import tqdm

            results: List[Optional[_ClusterProcessResult]] = [None] * len(items)
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_index = {
                    executor.submit(self._process_cluster, cid, vars_): i
                    for i, (cid, vars_) in enumerate(items)
                }
                with tqdm(total=len(items), desc="Variable groups") as pbar:
                    for future in as_completed(future_to_index):
                        result = future.result()
                        results[future_to_index[future]] = result
                        pbar.set_postfix(
                            cluster=result.cluster_id,
                            label=result.group.label[:30],
                        )
                        pbar.update(1)
            return cast(List[_ClusterProcessResult], results)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            return list(
                executor.map(
                    lambda kv: self._process_cluster(kv[0], kv[1]),
                    items,
                )
            )

    # ----- Internal LLM calls ----- #

    def _litellm_json_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Dict,
        cluster_id: int,
        call_label: str,
    ) -> Optional[str]:
        """Run a litellm completion and return raw JSON content, or None."""
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "litellm is required for LLM calls. "
                "Install with: uv pip install ai4data[metadata]"
            ) from exc

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=self.temperature,
            )
        except Exception:
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    response_format=get_json_object_format(),
                    temperature=self.temperature,
                )
            except Exception as exc2:
                logger.warning(
                    "%s call failed for cluster %d: %s",
                    call_label,
                    cluster_id,
                    exc2,
                )
                return None

        try:
            return response.choices[0].message.content
        except Exception as exc:
            logger.warning(
                "%s response missing content for cluster %d: %s",
                call_label,
                cluster_id,
                exc,
            )
            return None

    def _call_llm_for_cluster(
        self,
        cluster_id: int,
        variables: List[DictionaryVariable],
    ) -> Optional[VariableGroupCurationResult]:
        """Make one litellm completion call for a single cluster."""
        candidate_names = {v.variable_name for v in variables}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(variables)},
        ]

        content = self._litellm_json_completion(
            model=self.model,
            messages=messages,
            response_format=get_variable_group_response_format(),
            cluster_id=cluster_id,
            call_label="Curation",
        )
        if content is None:
            return None

        try:
            return VariableGroupCurationResult.from_llm_response(
                content,
                candidate_names=candidate_names,
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse curation response for cluster %d: %s\nContent: %s",
                cluster_id,
                exc,
                content[:300],
            )
            return None

    def _call_qa_for_curation(
        self,
        cluster_id: int,
        variables: List[DictionaryVariable],
        curation: VariableGroupCurationResult,
    ) -> Optional[VariableGroupQAResult]:
        """Run the self-consistency QA agent on a curation result."""
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": render_qa_user_prompt(variables, curation),
            },
        ]

        content = self._litellm_json_completion(
            model=self.qa_model,
            messages=messages,
            response_format=get_qa_response_format(),
            cluster_id=cluster_id,
            call_label="QA",
        )
        if content is None:
            return None

        try:
            return VariableGroupQAResult.model_validate_json(content)
        except Exception as exc:
            logger.warning(
                "Failed to parse QA response for cluster %d: %s\nContent: %s",
                cluster_id,
                exc,
                content[:300],
            )
            return None

    # ----- State guards ----- #

    def _require_loaded(self) -> None:
        if self._variables is None:
            raise RuntimeError(
                "No variables loaded. Call .load() before .embed()."
            )

    def _require_embedded(self) -> None:
        self._require_loaded()
        if self._embeddings is None:
            raise RuntimeError(
                "Embeddings not computed. Call .embed() before .cluster()."
            )

    def _require_clustered(self) -> None:
        self._require_embedded()
        if self._cluster_labels is None or self._cluster_map is None:
            raise RuntimeError(
                "Clustering not done. Call .cluster() before "
                ".generate_variable_groups()."
            )

    def _require_result(self) -> None:
        if self._result is None:
            raise RuntimeError(
                "No result available. Call .generate_variable_groups() or "
                ".augment() first."
            )

    # ----- Properties ----- #

    @property
    def variables(self) -> Optional[List[DictionaryVariable]]:
        """Loaded variables, or None if not yet loaded."""
        return self._variables

    @property
    def result(self) -> Optional[AugmentedDictionary]:
        """Augmented dictionary result, or None if not yet generated."""
        return self._result

    @property
    def n_variables(self) -> int:
        """Number of loaded variables."""
        return len(self._variables) if self._variables else 0
