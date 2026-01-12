"""
Adapter module to convert dedup output format to harmonization input format.

This module provides utilities to bridge the gap between the dataset extraction
pipeline output (dedup format) and the harmonization pipeline input (S2ORC-like format).
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_dedup_file(file_path: str) -> Dict[str, Any]:
    """
    Load a single dedup JSON file.

    Parameters
    ----------
    file_path : str
        Path to the dedup JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dedup_files(folder_path: str, pattern: str = "**/*_dedup.json") -> List[Dict[str, Any]]:
    """
    Load all dedup JSON files from a folder structure.

    Parameters
    ----------
    folder_path : str
        Root folder path containing dedup files.
    pattern : str, optional
        Glob pattern to match dedup files (default: "**/*_dedup.json").

    Returns
    -------
    list of dict
        List of parsed JSON objects from all dedup files.
    """
    folder = Path(folder_path)
    dedup_files = []

    for file_path in folder.glob(pattern):
        try:
            data = load_dedup_file(str(file_path))
            dedup_files.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    return dedup_files


def extract_dataset_mentions(dedup_files: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract and aggregate dataset mentions from dedup files.

    Converts the dedup format to a format compatible with harmonization pipeline:
    - dataset_name -> datasets (for compatibility)
    - dataset_tag -> label
    - Aggregates counts across all occurrences
    - Extracts acronyms (picking the first non-empty one if multiple)

    Parameters
    ----------
    dedup_files : list of dict
        List of dedup JSON objects.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: datasets, label, count, acronym
        Aggregated by unique dataset name.
    """
    mentions = []

    for dedup_data in dedup_files:
        if "datasets" not in dedup_data:
            continue

        for dataset in dedup_data["datasets"]:
            # Extract basic fields
            dataset_name = dataset.get("dataset_name", "").strip()
            if not dataset_name:
                continue

            dataset_tag = dataset.get("dataset_tag", "named")
            count = dataset.get("count", 1)

            # Extract acronym - handle both list and string formats
            acronym_field = dataset.get("acronym", [])
            if isinstance(acronym_field, list):
                # Pick first non-empty acronym
                acronym = next((a for a in acronym_field if a and str(a).strip()), None)
            else:
                acronym = acronym_field if acronym_field and str(acronym_field).strip() else None

            mentions.append(
                {"datasets": dataset_name, "label": dataset_tag, "count": count, "acronym": acronym}
            )

    # Convert to DataFrame and aggregate
    if not mentions:
        return pd.DataFrame(columns=["datasets", "label", "count", "acronym"])

    df = pd.DataFrame(mentions)

    # Aggregate by dataset name and label, summing counts
    # For acronyms, take the first non-null value
    agg_dict = {
        "count": "sum",
        "acronym": lambda x: next((v for v in x if pd.notna(v) and str(v).strip()), None),
    }

    df_agg = df.groupby(["datasets", "label"], as_index=False).agg(agg_dict)

    return df_agg


def convert_dedup_to_harmonization_format(
    dedup_files: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert dedup format to harmonization input format (S2ORC-like).

    Parameters
    ----------
    dedup_files : list of dict
        List of dedup JSON objects.

    Returns
    -------
    dict
        Dictionary with 'data' key containing list of dataset mentions
        in S2ORC-like format: {text, label, count, acronym}
    """
    df = extract_dataset_mentions(dedup_files)

    # Convert DataFrame to list of dicts with S2ORC-like field names
    data = []
    for _, row in df.iterrows():
        data.append(
            {
                "text": row["datasets"],
                "label": row["label"],
                "count": int(row["count"]),
                "acronym": row["acronym"] if pd.notna(row["acronym"]) else None,
            }
        )

    return {"data": data}


def prepare_dedup_dataframe(
    folder_path: str, pattern: str = "**/*_dedup.json", named_only: bool = True
) -> pd.DataFrame:
    """
    Load dedup files and prepare a DataFrame ready for harmonization pipeline.

    This is a convenience function that combines loading and extraction.

    Parameters
    ----------
    folder_path : str
        Root folder path containing dedup files.
    pattern : str, optional
        Glob pattern to match dedup files (default: "**/*_dedup.json").
    named_only : bool, optional
        If True, only include datasets with label='named' (default: True).

    Returns
    -------
    pd.DataFrame
        DataFrame ready for harmonization with columns:
        datasets, label, count, acronym
    """
    dedup_files = load_dedup_files(folder_path, pattern)
    df = extract_dataset_mentions(dedup_files)

    if named_only and not df.empty:
        df = df[df["label"] == "named"].reset_index(drop=True)

    return df


def get_dedup_folder_structure(root_folder: str) -> List[str]:
    """
    Get all dedup folders in the expected structure.

    Expected structure:
        root_folder/
        ├── project_001/
        │   └── dedup/
        │       └── *.json
        ├── project_002/
        │   └── dedup/
        │       └── *.json

    Parameters
    ----------
    root_folder : str
        Root folder containing project subfolders with dedup directories.

    Returns
    -------
    list of str
        List of paths to dedup folders.
    """
    root = Path(root_folder)
    dedup_folders = []

    # Find all 'dedup' directories
    for dedup_path in root.glob("**/dedup"):
        if dedup_path.is_dir():
            dedup_folders.append(str(dedup_path))

    return sorted(dedup_folders)
