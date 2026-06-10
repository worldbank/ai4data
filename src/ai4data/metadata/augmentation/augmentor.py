"""Main orchestration class for data dictionary augmentation.

``DataDictionaryAugmentor`` implements a five-step pipeline:
1. Load: ingest variables from CSV, JSON, dict, or NADA catalog
2. Embed: encode variable labels/descriptions with sentence-transformers
3. Cluster: group semantically similar variables
4. Generate: call an LLM for each cluster to produce a theme name + description
5. Export: write the augmented dictionary to disk or return as a DataFrame

The LLM backend is provider-agnostic via ``litellm``: set ``model`` to any
litellm-supported model string (e.g., ``"claude-haiku-4-5-20251001"``,
``"gpt-4o-mini"``, ``"gemini/gemini-2.0-flash"``).

Install requirements: ``uv pip install ai4data[metadata]``
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    get_theme_response_format,
    render_user_prompt,
)
from .schemas import (
    AugmentedDictionary,
    DictionaryVariable,
    Theme,
    ThemeAssignment,
    ThemeGenerationResult,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class DataDictionaryAugmentor:
    """LLM-powered data dictionary augmentation.

    Generates thematic structure for microdata or administrative data dictionary
    variables using semantic clustering and LLM-elicited theme names.

    Parameters
    ----------
    model : str
        LiteLLM model string. Defaults to ``"claude-haiku-4-5-20251001"``.
        Other examples: ``"gpt-4o-mini"``, ``"gemini/gemini-2.0-flash"``.
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
    ...     .generate_themes()
    ... )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        n_clusters: Optional[int] = None,
        max_cluster_tokens: int = DEFAULT_MAX_CLUSTER_TOKENS,
        temperature: float = 0.0,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.model = model
        self.n_clusters = n_clusters
        self.max_cluster_tokens = max_cluster_tokens
        self.temperature = temperature
        self.random_state = random_state

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

    # ----- Step 4: Generate themes ----- #

    def generate_themes(self) -> AugmentedDictionary:
        """Call the LLM for each cluster and assemble the augmented dictionary.

        Makes one LLM call per cluster. Failures in individual clusters are
        logged and skipped (returning a ``Theme`` with name "Uncategorized" for
        affected variables) so that partial results are always returned.

        Returns
        -------
        AugmentedDictionary
            The complete augmented dictionary with all themes and assignments.
        """
        self._require_clustered()

        themes: List[Theme] = []
        variable_assignments: List[ThemeAssignment] = []
        cluster_to_theme: Dict[int, str] = {}

        for cluster_id, vars_ in self._cluster_map.items():
            result = self._call_llm_for_cluster(cluster_id, vars_)
            if result is not None:
                theme = Theme(
                    theme_name=result.theme_name,
                    description=result.description,
                    example_variables=result.example_variables[:5],
                )
                theme_name = result.theme_name
            else:
                # Graceful fallback for failed LLM calls
                theme = Theme(
                    theme_name="Uncategorized",
                    description="Theme generation failed for this cluster.",
                    example_variables=[vars_[0].variable_name] if vars_ else [],
                )
                theme_name = "Uncategorized"

            themes.append(theme)
            cluster_to_theme[cluster_id] = theme_name

        # Build per-variable assignments
        for idx, var in enumerate(self._variables):
            cid = int(self._cluster_labels[idx])
            variable_assignments.append(
                ThemeAssignment(
                    variable_name=var.variable_name,
                    theme_name=cluster_to_theme.get(cid, "Uncategorized"),
                    cluster_id=cid,
                )
            )

        self._result = AugmentedDictionary(
            dataset_id=getattr(self, "_dataset_id", None),
            themes=themes,
            variable_assignments=variable_assignments,
            metadata={
                "model": self.model,
                "embedding_model": self._encoder.model_name,
                "n_variables": len(self._variables),
                "n_clusters": len(self._cluster_map),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
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
    ) -> AugmentedDictionary:
        """Load → embed → cluster → generate themes in one call.

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
            Show embedding progress bar.

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
            .generate_themes()
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
            One row per variable with columns: ``variable_name``,
            ``theme_name``, ``cluster_id``.
        """
        self._require_result()
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc
        rows = [
            {
                "variable_name": a.variable_name,
                "theme_name": a.theme_name,
                "cluster_id": a.cluster_id,
            }
            for a in self._result.variable_assignments
        ]
        return pd.DataFrame(rows)

    # ----- Internal LLM call ----- #

    def _call_llm_for_cluster(
        self,
        cluster_id: int,
        variables: List[DictionaryVariable],
    ) -> Optional[ThemeGenerationResult]:
        """Make one litellm completion call for a single cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster identifier (used for logging).
        variables : list of DictionaryVariable
            Variables in this cluster.

        Returns
        -------
        ThemeGenerationResult or None
            Parsed result, or None if the call failed.
        """
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "litellm is required for LLM calls. "
                "Install with: uv pip install ai4data[metadata]"
            ) from exc

        user_prompt = render_user_prompt(variables)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Try structured output first; fall back to json_object mode
        response_format = get_theme_response_format()
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=self.temperature,
            )
        except Exception:
            # Some providers don't support strict JSON schema; retry with json_object
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    response_format=get_json_object_format(),
                    temperature=self.temperature,
                )
            except Exception as exc2:
                logger.warning(
                    "LLM call failed for cluster %d (%d variables): %s",
                    cluster_id,
                    len(variables),
                    exc2,
                )
                return None

        try:
            content = response.choices[0].message.content
            return ThemeGenerationResult.model_validate_json(content)
        except Exception as exc:
            logger.warning(
                "Failed to parse LLM response for cluster %d: %s\nContent: %s",
                cluster_id,
                exc,
                response.choices[0].message.content[:300] if response else "",
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
                "Clustering not done. Call .cluster() before .generate_themes()."
            )

    def _require_result(self) -> None:
        if self._result is None:
            raise RuntimeError(
                "No result available. Call .generate_themes() or .augment() first."
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
