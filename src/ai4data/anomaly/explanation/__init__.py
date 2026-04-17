"""ai4data.anomaly.explanation - Anomaly detection and explanation in timeseries data.

This module provides tools for detecting anomalies in timeseries indicators
and generating LLM-based explanations. It requires optional dependencies:

    uv pip install ai4data[anomaly]

Example usage:
    from ai4data.anomaly.explanation import (
        ScorecardWideAdapter,
        extract_anomaly_contexts,
        parse_batch_output,
    )
    from ai4data.anomaly.explanation.prompts import (
        SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE,
        get_anomaly_response_format,
    )

    adapter = ScorecardWideAdapter()
    df = adapter.load("wide.csv", "anomalies.csv")
"""

__version__ = "0.1.0"

from . import adapters, arbiter, context, explainers, output_parser, prompts, schemas
from .adapters import (
    CANONICAL_COLUMNS,
    ConfigurableAdapter,
    ScorecardWideAdapter,
    adapter_from_config,
    load_csv_filtered,
)
from .legacy_custom_id import (
    new_compact_id_from_legacy_parts,
    parse_legacy_nosearch_custom_id,
    write_custom_id_map_from_legacy_batch_output,
)
from .mapping_suggest import (
    suggest_column_mapping,
    suggest_column_mapping_with_llm,
)
from .context import extract_anomaly_contexts
from .arbiter import (
    build_arbiter_payload,
    group_explanations_by_context_with_providers,
    harmonize_explanations,
)
from .explainers import list_explainers, register_explainer
from .batch_builder import (
    CUSTOM_ID_MAP_SUFFIX,
    build_batch_file,
    compact_custom_id,
    list_batch_providers,
)
from .batch_runner import (
    download_batch_output,
    run_batch,
    submit_batch,
    wait_for_batch,
)
from .output_parser import parse_batch_output
from .review_output import (
    export_for_review,
    export_for_review_with_explainers,
    to_review_format,
    to_review_format_with_explainers,
)
from .schemas import (
    Anomaly,
    AnomalyExplanation,
    Classification,
    EvidenceSource,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "ConfigurableAdapter",
    "ScorecardWideAdapter",
    "adapter_from_config",
    "load_csv_filtered",
    "suggest_column_mapping",
    "suggest_column_mapping_with_llm",
    "extract_anomaly_contexts",
    "build_batch_file",
    "compact_custom_id",
    "CUSTOM_ID_MAP_SUFFIX",
    "list_batch_providers",
    "parse_legacy_nosearch_custom_id",
    "new_compact_id_from_legacy_parts",
    "write_custom_id_map_from_legacy_batch_output",
    "submit_batch",
    "wait_for_batch",
    "download_batch_output",
    "run_batch",
    "parse_batch_output",
    "to_review_format",
    "to_review_format_with_explainers",
    "export_for_review",
    "export_for_review_with_explainers",
    "build_arbiter_payload",
    "group_explanations_by_context_with_providers",
    "harmonize_explanations",
    "register_explainer",
    "list_explainers",
    "Anomaly",
    "AnomalyExplanation",
    "Classification",
    "EvidenceSource",
    "adapters",
    "arbiter",
    "context",
    "explainers",
    "output_parser",
    "prompts",
    "schemas",
]
