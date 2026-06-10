"""
unhcr_to_schema.py
==================

Convert UNHCR/ReliefWeb scraped metadata JSON files to the Document Metadata
Schema defined in the Metadata Standards for Improved Data Discoverability
(Documents section).

The UNHCR metadata files are produced by scraping ReliefWeb report pages.
Key names come directly from HTML ``<dt>`` elements and may appear in
singular or plural forms (e.g., ``Theme`` / ``Themes``).  This script
normalizes those variants before mapping to the target schema.

Usage
-----
.. code-block:: bash

    uv run python -m data_snapshot.metadata.unhcr_to_schema \
        --input_dir path/to/metadata_dir/ \
        --output_dir path/to/output_dir/
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from countrycode import countrycode
from tqdm.auto import tqdm


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

RELIEFWEB_BASE_URL = "https://reliefweb.int"

# Language name -> ISO 639-2 code
LANGUAGE_CODES: dict[str, str] = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "russian": "ru",
    "portuguese": "pt",
    "japanese": "ja",
    "german": "de",
    "italian": "it",
    "dutch": "nl",
    "korean": "ko",
    "turkish": "tr",
    "polish": "pl",
    "romanian": "ro",
    "thai": "th",
    "ukrainian": "uk",
    "hungarian": "hu",
    "czech": "cs",
    "swedish": "sv",
    "norwegian": "no",
    "finnish": "fi",
    "danish": "da",
    "greek": "el",
    "hebrew": "he",
    "persian": "fa",
    "dari": "prs",
    "pashto": "ps",
    "urdu": "ur",
    "hindi": "hi",
    "bengali": "bn",
    "burmese": "my",
    "somali": "so",
    "swahili": "sw",
    "amharic": "am",
    "tigrinya": "ti",
    "hausa": "ha",
}

# Special country name -> ISO3 mappings not handled by countrycode
COUNTRY_SPECIAL_CASES: dict[str, str] = {
    "World": "WLD",
    "occupied Palestinian territory": "PSE",
    "the Republic of North Macedonia": "MKD",
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _clean_text(text: str | None) -> str:
    """Normalize whitespace and strip a string.

    Parameters
    ----------
    text : str | None
        Raw text value.

    Returns
    -------
    str
        Cleaned text, or empty string if *text* is ``None``.
    """
    if text is None or text == "":
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _merge_list_keys(doc: dict[str, Any], *keys: str) -> list[str]:
    """Merge values from multiple keys into a single deduplicated list.

    Handles the singular/plural key variants produced by the ReliefWeb
    scraper (e.g., ``Theme`` and ``Themes``).

    Parameters
    ----------
    doc : dict[str, Any]
        Source UNHCR metadata document.
    *keys : str
        One or more key names to merge (e.g., ``"Theme"``, ``"Themes"``).

    Returns
    -------
    list[str]
        Deduplicated list of values, preserving insertion order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for key in keys:
        val = doc.get(key)
        if val is None:
            continue
        items = val if isinstance(val, list) else [val]
        for item in items:
            cleaned = _clean_text(item)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
    return result


def _parse_date(date_str: str | None) -> str:
    """Parse a ReliefWeb date string to ``YYYY-MM-DD`` format.

    ReliefWeb uses formats like ``"20 Aug 2019"`` or ``"11 May 2025"``.

    Parameters
    ----------
    date_str : str | None
        Raw date string from ReliefWeb.

    Returns
    -------
    str
        Date formatted as ``YYYY-MM-DD``, or empty string if parsing fails.
    """
    if not date_str or not date_str.strip():
        return ""
    try:
        dt = datetime.strptime(date_str.strip(), "%d %b %Y")
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def _get_language_code(name: str) -> str:
    """Map a language name to its ISO 639 code.

    Parameters
    ----------
    name : str
        Language name (e.g., ``"English"``).

    Returns
    -------
    str
        ISO 639 code (e.g., ``"en"``), or empty string if unknown.
    """
    if not name:
        return ""
    return LANGUAGE_CODES.get(name.strip().lower(), "")


def _get_country_code(name: str) -> str:
    """Map a country name to its ISO3 code.

    Uses the ``countrycode`` library with a fallback for special cases.

    Parameters
    ----------
    name : str
        Country name (e.g., ``"Syrian Arab Republic"``).

    Returns
    -------
    str
        ISO3 code (e.g., ``"SYR"``), or empty string if resolution fails.
    """
    if not name or not name.strip():
        return ""
    name = name.strip()
    if name in COUNTRY_SPECIAL_CASES:
        return COUNTRY_SPECIAL_CASES[name]
    try:
        result = countrycode(
            [name], origin="country.name.en.regex", destination="iso3c"
        )
        if result and result[0]:
            return result[0]
    except Exception:
        pass
    return ""


def _derive_idno(filepath: Path) -> str:
    """Derive a document identifier from the metadata filename.

    Parameters
    ----------
    filepath : Path
        Path to the metadata JSON file.

    Returns
    -------
    str
        Filename stem with ``_metadata`` suffix removed if present.
    """
    stem = filepath.stem
    if stem.endswith("_metadata"):
        stem = stem[: -len("_metadata")]
    return stem


# ------------------------------------------------------------------------------
# Schema transformation
# ------------------------------------------------------------------------------


def unhcr_to_schema(
    doc: dict[str, Any],
    *,
    idno: str,
) -> dict[str, Any]:
    """Convert a UNHCR/ReliefWeb metadata document to the Document Metadata Schema.

    Parameters
    ----------
    doc : dict[str, Any]
        Raw UNHCR metadata dict loaded from a scraped JSON file.
    idno : str
        Document identifier (typically derived from the filename).

    Returns
    -------
    dict[str, Any]
        Document Metadata Schema compliant dict with ``metadata_information``,
        ``document_description``, ``provenance``, ``tags``, and ``additional``.
    """
    title = _clean_text(doc.get("title", ""))
    now = datetime.now()
    prod_date = now.strftime("%Y-%m-%d")

    # --- Merge singular/plural key variants ---
    sources = _merge_list_keys(doc, "Source", "Sources")
    themes = _merge_list_keys(doc, "Theme", "Themes")
    languages = _merge_list_keys(doc, "Language", "Languages")
    primary_countries = _merge_list_keys(doc, "Primary country")
    other_countries = _merge_list_keys(doc, "Other country", "Other countries")
    disasters = _merge_list_keys(doc, "Disaster", "Disasters")
    disaster_types = _merge_list_keys(doc, "Disaster type", "Disaster types")

    # --- Authors (from sources) ---
    list_authors: list[dict[str, str]] = [
        {
            "first_name": "",
            "initial": "",
            "last_name": "",
            "affiliation": "",
            "full_name": src,
        }
        for src in sources
    ]

    # --- Dates ---
    date_published = _parse_date(doc.get("Originally published"))
    date_available = _parse_date(doc.get("Posted"))

    # --- Countries ---
    all_countries = primary_countries + [
        c for c in other_countries if c not in primary_countries
    ]
    list_countries: list[dict[str, str]] = [
        {"name": name, "code": _get_country_code(name)} for name in all_countries
    ]

    # --- Languages ---
    list_languages: list[dict[str, str]] = [
        {"name": lang, "code": _get_language_code(lang)} for lang in languages
    ]
    if not list_languages:
        list_languages = [{"name": "English", "code": "en"}]

    # --- Themes ---
    list_themes: list[dict[str, str]] = [
        {
            "id": "",
            "name": theme,
            "parent_id": "",
            "vocabulary": "UNHCR - theme",
            "uri": "",
        }
        for theme in themes
    ]

    # --- URL and relations ---
    report_url = _clean_text(doc.get("report_url", ""))
    pdf_url = _clean_text(doc.get("pdf_url", ""))

    relations: list[dict[str, str]] = []
    if pdf_url:
        relations.append({"name": pdf_url, "type": "hasFormat"})

    # --- Title statement ---
    title_statement: dict[str, str] = {
        "idno": idno,
        "title": title,
        "sub_title": "",
        "alternate_title": "",
        "translated_title": "",
    }

    # --- Document description ---
    doc_type = _clean_text(doc.get("Format", ""))

    document_description: dict[str, Any] = {
        "title_statement": title_statement,
        "authors": list_authors,
        "date_created": "",
        "date_available": date_available,
        "date_modified": "",
        "date_published": date_published,
        "identifiers": [],
        "type": doc_type,
        "abstract": "",
        "ref_country": list_countries,
        "geographic_units": [],
        "spatial_coverage": "",
        "languages": list_languages,
        "volume": "",
        "number": "",
        "series": "",
        "publisher_address": "",
        "organization": "",
        "url": report_url,
        "keywords": [],
        "themes": list_themes,
        "topics": [],
        "relations": relations,
        "security_classification": "",
        "access_restrictions": "",
        "edition": "",
        "contacts": [],
        "usage_terms": "",
        "notes": [],
    }

    # --- Metadata information ---
    metadata_information: dict[str, Any] = {
        "title": title,
        "idno": idno,
        "producers": [
            {
                "name": "UNHCR",
                "abbr": "UNHCR",
                "affiliation": "UNHCR",
                "role": "Source",
            }
        ],
        "production_date": prod_date,
        "version": f"Converted from UNHCR/ReliefWeb metadata on {prod_date}",
    }

    # --- Provenance ---
    provenance: list[dict[str, Any]] = [
        {
            "origin_description": {
                "harvest_date": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "altered": False,
                "base_url": RELIEFWEB_BASE_URL,
                "identifier": report_url or idno,
                "date_stamp": date_published or date_available,
                "metadata_namespace": "UNHCR_RELIEFWEB",
            }
        }
    ]

    # --- Additional (UNHCR-specific) ---
    # Always emit ALL keys with defaults to ensure a uniform Arrow struct
    # schema across files (required by HuggingFace dataset streaming).
    additional: dict[str, Any] = {
        "additional.unhcr_report_url": report_url,
        "additional.unhcr_title": title,
        "additional.unhcr_format": doc_type,
        "additional.unhcr_sources": sources,
        "additional.unhcr_posted": _clean_text(doc.get("Posted", "")),
        "additional.unhcr_originally_published": _clean_text(
            doc.get("Originally published", "")
        ),
        "additional.unhcr_origin": _clean_text(doc.get("Origin", "")),
        "additional.unhcr_primary_country": primary_countries,
        "additional.unhcr_other_countries": other_countries,
        "additional.unhcr_themes": themes,
        "additional.unhcr_languages": languages,
        "additional.unhcr_disasters": disasters,
        "additional.unhcr_disaster_types": disaster_types,
        "additional.unhcr_pdf_url": pdf_url,
    }

    # --- Build output ---
    result: dict[str, Any] = {
        "type": "document",
        "metadata_information": metadata_information,
        "document_description": document_description,
        "provenance": provenance,
        "tags": [],
        "additional": additional,
        "schematype": "document",
    }

    return result


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def main(input_dir: str, output_dir: str) -> None:
    """Convert all UNHCR metadata JSON files in a directory.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing UNHCR metadata ``*.json`` files.
    output_dir : str
        Path to the output directory for converted schema files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        print(f"No *.json files found in {input_path}")
        return

    print(f"Found {len(json_files)} JSON files in {input_path}")

    success = 0
    errors = 0
    for filepath in tqdm(json_files):
        try:
            with open(filepath, encoding="utf-8") as f:
                doc = json.load(f)

            idno = _derive_idno(filepath)
            schema_doc = unhcr_to_schema(doc, idno=idno)

            out_file = output_path / filepath.name
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(schema_doc, f, ensure_ascii=False, indent=2)

            success += 1
        except Exception as e:
            print(f"ERROR processing {filepath.name}: {e}")
            errors += 1

    print(f"\nDone: {success} converted, {errors} errors")
    print(
        "\nREMINDER: Run enforce_metadata_schema with ALL subsets "
        "(--input_dir for each) to unify schemas before uploading to "
        "HuggingFace."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert UNHCR/ReliefWeb scraped metadata to Document Metadata Schema."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing UNHCR metadata *.json files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for converted schema files",
    )
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)
