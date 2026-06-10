"""
enforce_metadata_schema.py
==========================

Post-process document metadata JSON files to ensure HuggingFace dataset
compatibility.  HuggingFace (via Apache Arrow) requires every JSON file in
a dataset to share an **identical schema** — same keys at every nesting
level, same types, and no empty ``[]`` arrays where other files have
typed items.

This script performs **deep recursive schema unification** across one or
more input directories, applying three classes of fixes:

1. **Key unification** — at *every* nesting level (not just ``additional``),
   scan all files to find the union of all keys and back-fill missing ones
   with type-appropriate defaults.
2. **Empty-list backfilling** — replace empty ``[]`` arrays with a
   single-item placeholder so Arrow infers a consistent element type
   (avoids ``list<null>`` vs ``list<struct<…>>`` or ``list<string>``).
3. **Date normalisation** — convert date-like strings to ISO 8601.

Multiple ``--input_dir`` paths can be provided to unify schemas **across**
dataset subsets (e.g. UNHCR + PRWP + Refugee).

Usage
-----
.. code-block:: bash

    # Single directory
    uv run python -m data_snapshot.metadata.enforce_metadata_schema \\
        --input_dir data/hf_metadata/prwp/ \\
        --output_dir data/hf_metadata_fixed/prwp/

    # Multiple directories — unify schema across subsets
    uv run python -m data_snapshot.metadata.enforce_metadata_schema \\
        --input_dir data/hf_metadata/unhcr/ \\
        --input_dir data/hf_metadata/prwp/ \\
        --input_dir data/hf_metadata/refugee/ \\
        --output_dir data/hf_metadata_fixed/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


# ------------------------------------------------------------------------------
# Date helpers
# ------------------------------------------------------------------------------

_SLASH_DATE_RE = re.compile(r"^(\d{4})/(\d{2})/(\d{2})$")
_SPACE_DATETIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})$")

_DATE_PATHS: list[tuple[str, ...]] = [
    ("metadata_information", "production_date"),
    ("document_description", "date_created"),
    ("document_description", "date_available"),
    ("document_description", "date_modified"),
    ("document_description", "date_published"),
]

_PROVENANCE_DATE_KEYS: list[str] = ["harvest_date", "date_stamp"]


def _normalize_date(value: Any) -> Any:
    """Normalize a date-like string to ISO 8601 format.

    Parameters
    ----------
    value : Any
        The value to normalise.

    Returns
    -------
    Any
        The normalised value, or the original if no transformation applied.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    text = value.strip()
    m = _SLASH_DATE_RE.match(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = _SPACE_DATETIME_RE.match(text)
    if m:
        return f"{m.group(1)}T{m.group(2)}Z"
    return value


def _normalize_dates_in_doc(doc: dict[str, Any]) -> None:
    """Normalize all known date fields in a metadata document in-place.

    Parameters
    ----------
    doc : dict[str, Any]
        Metadata document dict (mutated in-place).
    """
    for path in _DATE_PATHS:
        obj = doc
        for key in path[:-1]:
            obj = obj.get(key, {})
            if not isinstance(obj, dict):
                break
        else:
            leaf = path[-1]
            if leaf in obj:
                obj[leaf] = _normalize_date(obj[leaf])
    for prov in doc.get("provenance", []):
        od = prov.get("origin_description", {})
        if isinstance(od, dict):
            for key in _PROVENANCE_DATE_KEYS:
                if key in od:
                    od[key] = _normalize_date(od[key])


# ------------------------------------------------------------------------------
# Deep schema template
# ------------------------------------------------------------------------------

# Sentinel to mark a node as a dict-type schema (vs a leaf value).
_DICT_SCHEMA = "__dict_schema__"
# Sentinel to mark a list whose items are dicts.
_LIST_OF_DICTS = "__list_of_dicts__"
# Sentinel to mark a list whose items are scalars.
_LIST_OF_SCALARS = "__list_of_scalars__"


def _infer_default(value: Any) -> Any:
    """Return a type-appropriate default for a given sample value.

    Parameters
    ----------
    value : Any
        A sample value from the data.

    Returns
    -------
    Any
        An empty default of the same type.
    """
    if isinstance(value, str):
        return ""
    if isinstance(value, list):
        return []
    if isinstance(value, dict):
        return {}
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return 0.0
    return None


def build_deep_template(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a deep template capturing the union schema of all documents.

    Recursively walks all documents and builds a template dict that
    contains every key seen at every nesting level, with information
    about the expected type at each position.

    The template has the following structure:

    - For dict-valued keys: ``{_DICT_SCHEMA: True, "child_key": ...}``
    - For list-of-dict keys: ``{_LIST_OF_DICTS: True, "child_key": ...}``
    - For list-of-scalar keys: ``{_LIST_OF_SCALARS: sample_scalar}``
    - For leaf keys: the sample value itself

    Parameters
    ----------
    docs : list[dict[str, Any]]
        All loaded metadata documents.

    Returns
    -------
    dict[str, Any]
        The deep template.
    """
    template: dict[str, Any] = {}

    def _merge(tmpl: dict[str, Any], obj: dict[str, Any]) -> None:
        for key, value in obj.items():
            if isinstance(value, dict):
                # Dict-valued field — recurse
                if key not in tmpl:
                    tmpl[key] = {_DICT_SCHEMA: True}
                sub = tmpl[key]
                if isinstance(sub, dict) and sub.get(_DICT_SCHEMA):
                    _merge(sub, value)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    # List of dicts — merge all item keys.
                    # Upgrade from [] if this is first populated encounter.
                    if key not in tmpl or (
                        isinstance(tmpl[key], list) and not tmpl[key]
                    ):
                        tmpl[key] = {_LIST_OF_DICTS: True}
                    sub = tmpl[key]
                    if isinstance(sub, dict) and sub.get(_LIST_OF_DICTS):
                        for item in value:
                            if isinstance(item, dict):
                                _merge(sub, item)
                elif value:
                    # List of scalars — record a sample.
                    # Upgrade from [] if this is first populated encounter.
                    if key not in tmpl or (
                        isinstance(tmpl[key], list) and not tmpl[key]
                    ):
                        tmpl[key] = {_LIST_OF_SCALARS: value[0]}
                else:
                    # Empty list — record as unknown list if not seen before
                    if key not in tmpl:
                        tmpl[key] = []
            else:
                # Leaf value — record sample for type inference
                if key not in tmpl:
                    tmpl[key] = value

    for doc in docs:
        _merge(template, doc)

    return template


def apply_deep_template(doc: dict[str, Any], template: dict[str, Any]) -> None:
    """Apply the deep template to a document, filling in missing keys.

    Recursively walks the template and ensures the document has every
    key at every nesting level. Missing keys are added with
    type-appropriate defaults. Empty lists are replaced with typed
    placeholders where the template indicates a non-empty type.

    Parameters
    ----------
    doc : dict[str, Any]
        Metadata document dict (mutated in-place).
    template : dict[str, Any]
        The deep template from :func:`build_deep_template`.
    """
    for key, tmpl_value in template.items():
        # Skip sentinel keys
        if key in (_DICT_SCHEMA, _LIST_OF_DICTS, _LIST_OF_SCALARS):
            continue

        if isinstance(tmpl_value, dict):
            if tmpl_value.get(_DICT_SCHEMA):
                # Expected: a dict
                if key not in doc or not isinstance(doc[key], dict):
                    doc[key] = {}
                apply_deep_template(doc[key], tmpl_value)

            elif tmpl_value.get(_LIST_OF_DICTS):
                # Expected: a list of dicts
                if key not in doc:
                    doc[key] = []
                current = doc[key]
                if isinstance(current, list):
                    if len(current) == 0:
                        # Empty list → add placeholder struct
                        placeholder = _build_placeholder_struct(tmpl_value)
                        doc[key] = [placeholder]
                    else:
                        # Existing items — ensure they all have the
                        # full key set from the template
                        for item in current:
                            if isinstance(item, dict):
                                _backfill_struct_item(item, tmpl_value)

            elif _LIST_OF_SCALARS in tmpl_value:
                # Expected: a list of scalars
                if key not in doc:
                    doc[key] = []
                current = doc[key]
                if isinstance(current, list) and len(current) == 0:
                    sample = tmpl_value[_LIST_OF_SCALARS]
                    doc[key] = [_infer_default(sample)]

        elif isinstance(tmpl_value, list):
            # Template has [] — was never populated in any file.
            # Just ensure the key exists.
            if key not in doc:
                doc[key] = []

        else:
            # Leaf value — ensure key exists with typed default
            if key not in doc:
                doc[key] = _infer_default(tmpl_value)


def _build_placeholder_struct(tmpl: dict[str, Any]) -> dict[str, Any]:
    """Build a placeholder struct dict from a list-of-dicts template.

    Parameters
    ----------
    tmpl : dict[str, Any]
        Template for a list-of-dicts field.

    Returns
    -------
    dict[str, Any]
        A dict with all expected keys set to typed defaults.
    """
    result: dict[str, Any] = {}
    for key, value in tmpl.items():
        if key in (_DICT_SCHEMA, _LIST_OF_DICTS, _LIST_OF_SCALARS):
            continue
        if isinstance(value, dict):
            if value.get(_DICT_SCHEMA):
                result[key] = _build_placeholder_dict(value)
            elif value.get(_LIST_OF_DICTS):
                result[key] = [_build_placeholder_struct(value)]
            elif _LIST_OF_SCALARS in value:
                result[key] = [_infer_default(value[_LIST_OF_SCALARS])]
            else:
                result[key] = {}
        elif isinstance(value, list):
            result[key] = []
        else:
            result[key] = _infer_default(value)
    return result


def _build_placeholder_dict(tmpl: dict[str, Any]) -> dict[str, Any]:
    """Build a placeholder dict from a dict-schema template.

    Parameters
    ----------
    tmpl : dict[str, Any]
        Template for a dict-valued field.

    Returns
    -------
    dict[str, Any]
        A dict with all expected keys set to typed defaults.
    """
    result: dict[str, Any] = {}
    for key, value in tmpl.items():
        if key in (_DICT_SCHEMA, _LIST_OF_DICTS, _LIST_OF_SCALARS):
            continue
        if isinstance(value, dict):
            if value.get(_DICT_SCHEMA):
                result[key] = _build_placeholder_dict(value)
            elif value.get(_LIST_OF_DICTS):
                result[key] = [_build_placeholder_struct(value)]
            elif _LIST_OF_SCALARS in value:
                result[key] = [_infer_default(value[_LIST_OF_SCALARS])]
            else:
                result[key] = {}
        elif isinstance(value, list):
            result[key] = []
        else:
            result[key] = _infer_default(value)
    return result


def _backfill_struct_item(item: dict[str, Any], tmpl: dict[str, Any]) -> None:
    """Ensure a struct item has all keys from the template.

    Parameters
    ----------
    item : dict[str, Any]
        A dict item from a list (mutated in-place).
    tmpl : dict[str, Any]
        Template for the list-of-dicts field.
    """
    for key, value in tmpl.items():
        if key in (_DICT_SCHEMA, _LIST_OF_DICTS, _LIST_OF_SCALARS):
            continue
        if key not in item:
            if isinstance(value, dict):
                if value.get(_DICT_SCHEMA):
                    item[key] = _build_placeholder_dict(value)
                elif value.get(_LIST_OF_DICTS):
                    item[key] = [_build_placeholder_struct(value)]
                elif _LIST_OF_SCALARS in value:
                    item[key] = [_infer_default(value[_LIST_OF_SCALARS])]
                else:
                    item[key] = {}
            elif isinstance(value, list):
                item[key] = []
            else:
                item[key] = _infer_default(value)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def process_directories(
    input_dirs: list[str],
    output_dir: str,
) -> None:
    """Process metadata JSON files across one or more directories.

    Builds a unified deep schema template from *all* files across all
    input directories, then applies it to every file. Output files are
    written to ``output_dir``, preserving the input directory name as a
    subdirectory.

    Parameters
    ----------
    input_dirs : list[str]
        Paths to directories containing metadata ``*.json`` files.
    output_dir : str
        Path to the root output directory. Each input directory's files
        will be written to ``<output_dir>/<dirname>/``.
    """
    out_root = Path(output_dir)

    # --- Discover files ---
    all_file_groups: list[tuple[Path, list[Path]]] = []
    for d in input_dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"WARNING: skipping {p} (not a directory)")
            continue
        files = sorted(p.glob("*.json"))
        if files:
            all_file_groups.append((p, files))
        else:
            print(f"WARNING: no *.json files in {p}")

    if not all_file_groups:
        print("No files found. Exiting.")
        return

    total_files = sum(len(files) for _, files in all_file_groups)
    dir_names = [p.name for p, _ in all_file_groups]
    print(
        f"Scanning {total_files} files across {len(all_file_groups)} "
        f"directories: {dir_names}"
    )

    # --- Pass 1: Load all documents, normalise dates, build template ---
    all_docs: list[tuple[Path, Path, dict[str, Any]]] = []
    load_errors = 0
    for src_dir, files in all_file_groups:
        for filepath in tqdm(files, desc=f"Loading {src_dir.name}"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    doc = json.load(f)
                _normalize_dates_in_doc(doc)
                all_docs.append((src_dir, filepath, doc))
            except Exception as e:
                print(f"  ERROR loading {filepath.name}: {e}")
                load_errors += 1

    doc_dicts = [doc for _, _, doc in all_docs]
    template = build_deep_template(doc_dicts)

    # Count template stats
    def _count_keys(t: dict[str, Any]) -> int:
        count = 0
        for k, v in t.items():
            if k in (_DICT_SCHEMA, _LIST_OF_DICTS, _LIST_OF_SCALARS):
                continue
            count += 1
            if isinstance(v, dict):
                count += _count_keys(v)
        return count

    print(f"  Built deep template with {_count_keys(template)} total keys")

    # --- Pass 2: Apply template, write output ---
    write_errors = 0
    for src_dir, filepath, doc in tqdm(all_docs, desc="Writing"):
        try:
            apply_deep_template(doc, template)
            dest_dir = out_root / src_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_file = dest_dir / filepath.name
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ERROR writing {filepath.name}: {e}")
            write_errors += 1

    ok = len(all_docs) - write_errors
    print(
        f"\nDone: {ok} processed, "
        f"{load_errors} load errors, {write_errors} write errors"
    )
    print(f"Output written to: {out_root}")


def main() -> None:
    """Parse CLI arguments and run the metadata enforcement pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Enforce uniform metadata schema for HuggingFace compatibility. "
            "Performs deep recursive schema unification across one or more "
            "directories: key backfilling, empty-list typing, and date "
            "normalisation."
        ),
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        action="append",
        dest="input_dirs",
        help=(
            "Directory containing metadata *.json files. "
            "Can be specified multiple times for cross-subset unification."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory for fixed files.",
    )
    args = parser.parse_args()
    process_directories(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
