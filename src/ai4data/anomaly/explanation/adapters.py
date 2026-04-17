"""Data adapters for converting legacy formats to canonical timeseries anomaly format.

The canonical format uses these column names:
- indicator_id, indicator_name
- geography_id, geography_name
- period, value, is_imputed
- anomaly_score, outlier_count
- freq (optional)

Use adapter_from_config(mapping) for custom column mappings, or ScorecardWideAdapter
for the default World Bank Scorecard format.
"""

from pathlib import Path
from typing import Any, Collection, Dict, Optional, Protocol

import pandas as pd


# Default Scorecard column mapping (wide format + anomaly scores)
SCORECARD_COLUMN_MAPPING = {
    "indicator_id": "INDICATOR",
    "indicator_name": "INDICATOR_LABEL",
    "geography_id": "REF_AREA",
    "geography_name": "REF_AREA_LABEL",
    "period": "YEAR",
    "value": "VALUE",
    "is_imputed": "Imputed",
    "anomaly_score": "absZscore",
    "outlier_count": "outlier_indicator_total",
    "freq": "FREQ",
}

# Canonical column names (required for pipeline)
REQUIRED_CANONICAL_COLUMNS = [
    "indicator_id",
    "indicator_name",
    "geography_id",
    "geography_name",
    "period",
    "value",
    "is_imputed",
    "anomaly_score",
    "outlier_count",
]
OPTIONAL_CANONICAL_COLUMNS = ["freq"]
CANONICAL_COLUMNS = REQUIRED_CANONICAL_COLUMNS + OPTIONAL_CANONICAL_COLUMNS


def _detect_year_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that represent years (numeric, parseable as int)."""
    year_cols = []
    for col in df.columns:
        try:
            int(col)
            year_cols.append(col)
        except (ValueError, TypeError):
            pass
    return year_cols


def _build_reverse_mapping(
    mapping: Dict[str, str],
    columns: Collection[str],
) -> Dict[str, str]:
    """Build source_col -> canonical_col mapping for columns that exist."""
    return {
        v: k
        for k, v in mapping.items()
        if v in columns and k in CANONICAL_COLUMNS
    }


def _rename_to_canonical(
    df: pd.DataFrame,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    """Rename source columns to canonical names."""
    reverse = _build_reverse_mapping(mapping, df.columns)
    return df.rename(columns=reverse)


def _ensure_imputed_bool(df: pd.DataFrame, imputed_col: str) -> pd.DataFrame:
    """Ensure is_imputed column is boolean."""
    if imputed_col in df.columns:
        df = df.copy()
        df[imputed_col] = df[imputed_col].fillna(False).astype(bool)
    return df


class AnomalyAdapter(Protocol):
    """Protocol for anomaly data adapters."""

    def load_excel(self, path: str | Path) -> pd.DataFrame:
        """Load from Excel file (long format)."""
        ...

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """Load from CSV file (long format)."""
        ...

    def load_wide(
        self,
        wide_path: str | Path,
        anomaly_path: str | Path,
    ) -> pd.DataFrame:
        """Load from wide-format CSV + anomaly scores CSV."""
        ...


class ConfigurableAdapter:
    """Adapter that converts source data to canonical format using a column mapping.

    Supports Excel and CSV files (already in long format) and wide + anomaly
    CSV pairs (melt + merge).
    """

    def __init__(
        self,
        mapping: Dict[str, str],
        *,
        validate_output: bool = False,
    ):
        """Initialize with column mapping.

        Parameters
        ----------
        mapping : dict
            Maps canonical column names to source column names.
            Keys: indicator_id, indicator_name, geography_id, geography_name,
            period, value, is_imputed, anomaly_score, outlier_count.
        validate_output : bool
            If True, raise if output is missing required canonical columns.
        """
        self.mapping = mapping.copy()
        self.validate_output = validate_output

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optionally validate that required columns exist."""
        if self.validate_output:
            missing = set(REQUIRED_CANONICAL_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Adapter output missing required columns: {sorted(missing)}"
                )
        return df

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename source columns to canonical names and optionally validate."""
        df = _rename_to_canonical(df, self.mapping)
        imputed = self.mapping.get("is_imputed")
        if imputed and imputed in df.columns:
            df = _ensure_imputed_bool(df, imputed)
        return self._validate(df)

    def load_excel(self, path: str | Path) -> pd.DataFrame:
        """Load from Excel file and rename to canonical columns."""
        df = pd.read_excel(path)
        return self._finalize(df)

    def load_csv(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Load from CSV file and rename to canonical columns."""
        df = pd.read_csv(path, **kwargs)
        return self._finalize(df)

    def load_wide(
        self,
        wide_path: str | Path,
        anomaly_path: str | Path,
    ) -> pd.DataFrame:
        """Load wide-format CSV and anomaly CSV, merge, filter to anomalous series."""
        wide_df = pd.read_csv(wide_path)
        raw_df = pd.read_csv(anomaly_path)

        # Derive absZscore from Zscore if needed
        if "Zscore" in raw_df.columns and "absZscore" not in raw_df.columns:
            raw_df = raw_df.copy()
            raw_df["absZscore"] = raw_df["Zscore"].abs()

        # Melt wide to long
        year_cols = _detect_year_columns(wide_df)
        non_year_cols = [c for c in wide_df.columns if c not in year_cols]
        period_col = self.mapping.get("period", "YEAR")
        value_col = self.mapping.get("value", "VALUE")

        long_df = pd.melt(
            wide_df,
            id_vars=non_year_cols,
            value_vars=year_cols,
            var_name=period_col,
            value_name=value_col,
        )
        long_df[period_col] = pd.to_numeric(
            long_df[period_col], errors="coerce"
        ).astype("Int64")

        # Merge long series with anomaly metadata
        common_cols = [c for c in raw_df.columns if c in long_df.columns]
        merged_df = long_df.merge(raw_df, on=common_cols, how="left")

        # Filter to (indicator, geography) pairs present in anomaly file
        ind_col = self.mapping["indicator_id"]
        geo_col = self.mapping["geography_id"]
        anomaly_keys = raw_df[[ind_col, geo_col]].drop_duplicates()
        result = anomaly_keys.merge(
            merged_df,
            on=[ind_col, geo_col],
            how="left",
        )

        # Ensure is_imputed is bool
        imputed_col = self.mapping.get("is_imputed", "Imputed")
        if imputed_col in result.columns:
            result = _ensure_imputed_bool(result, imputed_col)

        # Rename to canonical
        result = _rename_to_canonical(result, self.mapping)
        return self._validate(result)

    # Backward compatibility: dict-like access
    def __getitem__(self, key: str):
        """Support adapt['adapt_excel'](path) for backward compatibility."""
        aliases = {
            "adapt_excel": self.load_excel,
            "adapt_csv": self.load_csv,
            "adapt_wide": self.load_wide,
        }
        if key in aliases:
            return aliases[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in {"adapt_excel", "adapt_csv", "adapt_wide"}


def adapter_from_config(
    mapping: Dict[str, str],
    *,
    validate_output: bool = False,
) -> ConfigurableAdapter:
    """Create a configured adapter from a column mapping.

    Parameters
    ----------
    mapping : dict
        Maps canonical column names to source column names.
        Required keys: indicator_id, indicator_name, geography_id, geography_name,
        period, value, is_imputed, anomaly_score, outlier_count.
    validate_output : bool
        If True, raise when output lacks required canonical columns.

    Returns
    -------
    ConfigurableAdapter
        Adapter with load_excel, load_csv, load_wide methods.
        Also supports dict access: adapt["adapt_excel"](path).
    """
    return ConfigurableAdapter(mapping, validate_output=validate_output)


def load_csv_filtered(
    path: str | Path,
    mapping: Dict[str, str],
    *,
    indicator_id_pattern: str | None = None,
    chunksize: int | None = None,
    validate_output: bool = False,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load a long-format CSV, optionally filter by indicator id regex, then adapt.

    When ``chunksize`` is set, reads the file in chunks and drops rows before full
    materialization, which reduces memory use for large files when filtering to a
    subset of indicators.

    Parameters
    ----------
    path : str or Path
        CSV path.
    mapping : dict
        Canonical name → source column name (same as :class:`ConfigurableAdapter`).
    indicator_id_pattern : str, optional
        If set, keep only rows where the **source** indicator column matches this
        regex (via :meth:`pandas.Series.str.match`).
    chunksize : int, optional
        If set, pass to :func:`pandas.read_csv` and filter each chunk.
    validate_output : bool
        Passed through to :class:`ConfigurableAdapter`.
    **read_csv_kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        Canonical long-format data.
    """
    path = Path(path)
    adapter = ConfigurableAdapter(mapping, validate_output=validate_output)
    ind_src = mapping.get("indicator_id")
    if not ind_src:
        raise ValueError("mapping must include 'indicator_id'")

    if chunksize is not None and int(chunksize) < 1:
        raise ValueError("chunksize must be a positive integer")

    if chunksize:
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=int(chunksize), **read_csv_kwargs):
            if indicator_id_pattern is not None:
                if ind_src not in chunk.columns:
                    raise KeyError(
                        f"indicator_id source column {ind_src!r} not in CSV chunk columns"
                    )
                chunk = chunk.loc[
                    chunk[ind_src].astype(str).str.match(indicator_id_pattern, na=False)
                ]
            if not chunk.empty:
                parts.append(chunk)
        if parts:
            df = pd.concat(parts, ignore_index=True)
        else:
            df = pd.read_csv(path, nrows=0, **read_csv_kwargs)
    else:
        df = pd.read_csv(path, **read_csv_kwargs)
        if indicator_id_pattern is not None:
            if ind_src not in df.columns:
                raise KeyError(
                    f"indicator_id source column {ind_src!r} not in CSV columns"
                )
            df = df.loc[
                df[ind_src].astype(str).str.match(indicator_id_pattern, na=False)
            ].copy()

    return adapter._finalize(df)


class ScorecardWideAdapter:
    """Adapter for World Bank Scorecard wide-format + anomaly scores CSVs.

    Uses SCORECARD_COLUMN_MAPPING by default. For custom mappings, use
    adapter_from_config(mapping) instead.
    """

    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        *,
        validate_output: bool = False,
    ):
        """Initialize with optional custom column mapping.

        Parameters
        ----------
        column_mapping : dict, optional
            Override default Scorecard mapping. Keys are canonical names.
        validate_output : bool
            If True, raise when output lacks required columns.
        """
        self._adapter = ConfigurableAdapter(
            column_mapping or SCORECARD_COLUMN_MAPPING.copy(),
            validate_output=validate_output,
        )

    def load(
        self,
        wide_path: str | Path,
        anomaly_path: str | Path,
    ) -> pd.DataFrame:
        """Load and convert to canonical format.

        Parameters
        ----------
        wide_path : str or Path
            Path to wide-format CSV (metadata cols + year columns).
        anomaly_path : str or Path
            Path to anomaly scores CSV.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with canonical column names.
        """
        return self._adapter.load_wide(wide_path, anomaly_path)
