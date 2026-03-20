"""
wds_to_schema.py
================
Convert World Bank Search API (WDS) v3 document metadata to the Document Metadata Schema
schema defined in the Metadata Standards for Improved Data Discoverability
(Chapter 5 Documents).

Implements the mapping documented in docs/WDS_TO_SCHEMA_MAPPING.md.

Source-tracking: When multiple WDS fields are combined (themes, topics, keywords),
the vocabulary field encodes the source as "World Bank - {wds_field}" for traceability.

Usage:
  # Transform a single document (from API response)
  from wds_to_schema import wds_to_schema
  doc = response["documents"]["D12345"]
  document_metadata_schema = wds_to_schema(doc)

  # Transform from JSON file
  python wds_to_schema.py --input=doc.json --output=document_12345.json

  # Transform from API URL (by guid)
  python wds_to_schema.py --guid=969971600710472848 --output_dir=metadata
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from countrycode import countrycode
except ImportError:
    countrycode = None  # type: ignore

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

WB_WDS_API_V3 = "https://search.worldbank.org/api/v3/wds"

# ISO2 "1W" = World
COUNTRYCODE_1W = "WLD"

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
}

REGION_SPECIAL_CASES: dict[str, str] = {
    "World": "WLD",
    "Africa": "AFR",
    "East Asia and Pacific": "EAP",
    "Europe and Central Asia": "ECA",
    "Latin America and Caribbean": "LAC",
    "Middle East and North Africa": "MNA",
    "South Asia": "SAR",
    "North America": "NAR",
}

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _clean_text(text: str | None) -> str:
    """Normalize whitespace: collapse newlines and multiple spaces, strip."""
    if text is None or text == "":
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _format_date(value: str | None) -> str:
    """Format date string to YYYY-MM-DD. Returns empty string if invalid."""
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def _get_country_code(name: str) -> str:
    """Map country/region name to ISO3 code."""
    if not name or not name.strip():
        return ""
    name = name.strip()
    if name in REGION_SPECIAL_CASES:
        return REGION_SPECIAL_CASES[name]
    if countrycode is None:
        return ""
    try:
        result = countrycode(
            [name], origin="country.name.en.regex", destination="iso3c"
        )
        if result and result[0]:
            return result[0]
    except Exception:
        pass
    return ""


def _iso2_to_iso3(code: str) -> str:
    """Convert ISO2 country code to ISO3."""
    if not code or code == "1W":
        return COUNTRYCODE_1W
    if countrycode is None:
        return code
    try:
        result = countrycode([code], origin="iso2c", destination="iso3c")
        if result and result[0]:
            return result[0]
    except Exception:
        pass
    return code


def _get_language_code(name: str) -> str:
    """Map language name to ISO 639-2 code."""
    if not name:
        return ""
    key = _clean_text(name).lower()
    return LANGUAGE_CODES.get(key, "")


def _extract_abstract(doc: dict[str, Any]) -> str:
    """Extract abstract from abstracts.cdata! or abstracts.#text."""
    abstracts = doc.get("abstracts") or {}
    if isinstance(abstracts, dict):
        return _clean_text(abstracts.get("cdata!", abstracts.get("#text", "")))
    if isinstance(abstracts, str):
        return _clean_text(abstracts)
    return ""


def _extract_from_indexed_object(obj: Any, key: str) -> list[str]:
    """Extract values from WDS indexed object like {"0":{"keywd":"x"},"1":{"keywd":"y"}}."""
    if not isinstance(obj, dict):
        return []
    result: list[str] = []
    for v in obj.values():
        if isinstance(v, dict) and key in v:
            val = _clean_text(v.get(key, ""))
            if val:
                result.append(val)
        elif isinstance(v, str) and v.strip():
            result.append(v.strip())
    return result


def _extract_alt_titles(alt_title_raw: Any) -> str:
    """Extract alternate titles from alt_title object or list."""
    if alt_title_raw is None:
        return ""
    titles: list[str] = []
    if isinstance(alt_title_raw, dict):
        for v in alt_title_raw.values():
            if isinstance(v, dict) and "alt_title" in v:
                t = _clean_text(v.get("alt_title", ""))
                if t:
                    titles.append(t)
            elif isinstance(v, str) and v.strip():
                titles.append(v.strip())
    elif isinstance(alt_title_raw, list):
        for a in alt_title_raw:
            if isinstance(a, dict) and "alt_title" in a:
                t = _clean_text(a.get("alt_title", ""))
                if t:
                    titles.append(t)
            elif isinstance(a, str) and a.strip():
                titles.append(a.strip())
    elif isinstance(alt_title_raw, str) and alt_title_raw.strip():
        titles.append(alt_title_raw.strip())
    return " / ".join(titles)


# ------------------------------------------------------------------------------
# Schema transformation
# ------------------------------------------------------------------------------


def wds_to_schema(
    doc: dict[str, Any],
    *,
    harvest_date: str | None = None,
    altered: bool = False,
    deduplicate_combined_fields: bool = True,
) -> dict[str, Any]:
    """
    Convert a WDS API v3 document to the Document Metadata Schema (Chapter 5).

    Args:
        doc: Raw document dict from WDS API (documents.<key>).
        harvest_date: ISO 8601 harvest timestamp. Default: current time.
        altered: Whether metadata was enriched/altered after harvest.
        deduplicate_combined_fields: If True (default), skip duplicate values when
            combining themes, topics, keywords from multiple WDS fields. If False,
            preserve all values, allowing the same term to appear multiple times
            with different vocabulary (source) tags for full provenance.

    Returns:
        Document Metadata Schema compliant document dict with metadata_information,
        document_description, provenance, and optional tags/additional.
    """
    idno = str(doc.get("id", ""))
    guid = str(doc.get("guid", ""))
    title = _clean_text(doc.get("display_title") or "")

    # Harvest metadata
    now = datetime.now()
    harvest = harvest_date or now.strftime("%Y-%m-%dT%H:%M:%SZ")
    prod_date = _format_date(doc.get("lupdate")) or now.strftime("%Y-%m-%d")

    # --- Identifiers ---
    list_ids: list[dict[str, str]] = [
        {"type": "World Bank D&R", "identifier": idno},
    ]
    if guid:
        list_ids.append({"type": "World Bank GUID", "identifier": guid})
    chronical = str(doc.get("chronical_docm_id") or "")
    if chronical:
        list_ids.append({"type": "World Bank Chronical ID", "identifier": chronical})

    entity_id = None
    if isinstance(doc.get("entityids"), dict):
        entity_id = doc["entityids"].get("entityid")
    elif isinstance(doc.get("entityids"), str):
        entity_id = doc.get("entityids")
    if entity_id:
        list_ids.append({"type": "World Bank Entity ID", "identifier": str(entity_id)})

    project_id = doc.get("projectid")
    if project_id:
        list_ids.append(
            {"type": "World Bank Project ID", "identifier": str(project_id)}
        )

    for id_type, id_key in [("DOI", "dois"), ("ISBN", "isbn"), ("ISSN", "issn")]:
        val = doc.get(id_key)
        if val:
            list_ids.append({"type": id_type, "identifier": str(val)})

    # --- Title statement ---
    docna = _extract_from_indexed_object(doc.get("docna") or {}, "docna")
    docna_str = docna[0] if docna else ""
    alt_title = _extract_alt_titles(doc.get("alt_title"))
    if not alt_title and docna_str and docna_str != title:
        alt_title = docna_str
    repnme = ""
    if isinstance(doc.get("repnme"), dict):
        repnme = _clean_text(doc["repnme"].get("repnme", ""))
    elif isinstance(doc.get("repnme"), str):
        repnme = _clean_text(doc.get("repnme", ""))
    # Use alternate only when different from title
    final_alt = alt_title or repnme
    if final_alt and _clean_text(final_alt) == title:
        final_alt = ""

    title_statement: dict[str, str] = {
        "idno": idno,
        "title": title,
        "sub_title": "",
        "alternate_title": final_alt,
        "translated_title": "",
    }

    # --- Authors ---
    list_authors: list[dict[str, Any]] = []
    for v in (doc.get("authors") or {}).values():
        if not isinstance(v, dict):
            continue
        author_val = v.get("author")
        if author_val is None:
            continue
        full_name = (
            author_val[0] if isinstance(author_val, (list, tuple)) else str(author_val)
        )
        if _clean_text(full_name):
            list_authors.append(
                {
                    "first_name": "",
                    "initial": "",
                    "last_name": "",
                    "affiliation": "",
                    "full_name": _clean_text(full_name),
                }
            )

    # --- Dates ---
    date_modified = _format_date(doc.get("last_modified_date") or doc.get("datestored"))
    date_published = _format_date(doc.get("docdt"))  # fallback for date_published

    # --- Countries ---
    list_countries: list[dict[str, str]] = []
    count_str = doc.get("count")
    countrycode_val = doc.get("countrycode", "")
    if count_str:
        names = [c.strip() for c in str(count_str).split(",") if c.strip()]
        if len(names) == 1 and countrycode_val:
            code = (
                _iso2_to_iso3(countrycode_val)
                if countrycode_val != "1W"
                else COUNTRYCODE_1W
            )
            list_countries.append({"name": names[0], "code": code})
        else:
            for name in names:
                code = _get_country_code(name)
                list_countries.append({"name": name, "code": code or ""})
    elif countrycode_val:
        if countrycode_val == "1W":
            list_countries.append({"name": "World", "code": COUNTRYCODE_1W})
        else:
            code = _iso2_to_iso3(countrycode_val)
            list_countries.append({"name": countrycode_val, "code": code})

    # --- Spatial coverage ---
    admreg = _clean_text(doc.get("admreg", ""))
    geo_regions = _extract_from_indexed_object(
        doc.get("geo_regions") or {}, "geo_region"
    )
    spatial_parts = [admreg] if admreg else []
    spatial_parts.extend(geo_regions)
    spatial_coverage = " ; ".join(spatial_parts) if spatial_parts else ""

    # --- Geographic units (from geo_regions) ---
    geographic_units: list[dict[str, str]] = []
    for gr in geo_regions:
        if gr and gr not in (admreg,):
            geographic_units.append({"name": gr, "code": "", "type": "region"})

    # --- Themes ---
    # Source-tracking: vocabulary includes WDS field name for traceability
    list_themes: list[dict[str, str]] = []
    for theme_str in (doc.get("theme") or "").split(","):
        thm = _clean_text(theme_str)
        if thm:
            list_themes.append(
                {
                    "id": "",
                    "name": thm,
                    "parent_id": "",
                    "vocabulary": "World Bank - theme",
                    "uri": "",
                }
            )
    majtheme = _clean_text(doc.get("majtheme", ""))
    if majtheme and (
        not deduplicate_combined_fields
        or not any(t["name"] == majtheme for t in list_themes)
    ):
        list_themes.append(
            {
                "id": "",
                "name": majtheme,
                "parent_id": "",
                "vocabulary": "World Bank - majtheme",
                "uri": "",
            }
        )
    sectr_vals = _extract_from_indexed_object(doc.get("sectr") or {}, "sector")
    for s in sectr_vals:
        if s and (
            not deduplicate_combined_fields
            or not any(t["name"] == s for t in list_themes)
        ):
            list_themes.append(
                {
                    "id": "",
                    "name": s,
                    "parent_id": "",
                    "vocabulary": "World Bank - sector",
                    "uri": "",
                }
            )

    # --- Topics ---
    # Source-tracking: vocabulary includes WDS field name (topicv3_name, ent_topic, subtopic, teratopic)
    list_topics: list[dict[str, str]] = []
    for topic_str in (doc.get("topicv3_name") or "").split(","):
        t = _clean_text(topic_str)
        if t:
            list_topics.append(
                {
                    "id": "",
                    "name": t,
                    "parent_id": "",
                    "vocabulary": "World Bank - topicv3_name",
                    "uri": "",
                }
            )
    for topic_str in (doc.get("ent_topic") or "").split(","):
        t = _clean_text(topic_str)
        if t and (
            not deduplicate_combined_fields
            or not any(p["name"] == t for p in list_topics)
        ):
            list_topics.append(
                {
                    "id": "",
                    "name": t,
                    "parent_id": "",
                    "vocabulary": "World Bank - ent_topic",
                    "uri": "",
                }
            )
    for topic_str in (doc.get("subtopic") or "").split(","):
        t = _clean_text(topic_str)
        if t and (
            not deduplicate_combined_fields
            or not any(p["name"] == t for p in list_topics)
        ):
            list_topics.append(
                {
                    "id": "",
                    "name": t,
                    "parent_id": "",
                    "vocabulary": "World Bank - subtopic",
                    "uri": "",
                }
            )
    for topic_str in (doc.get("teratopic") or "").split(","):
        t = _clean_text(topic_str)
        if t and (
            not deduplicate_combined_fields
            or not any(p["name"] == t for p in list_topics)
        ):
            list_topics.append(
                {
                    "id": "",
                    "name": t,
                    "parent_id": "",
                    "vocabulary": "World Bank - teratopic",
                    "uri": "",
                }
            )

    # --- Keywords ---
    # Source-tracking: vocabulary includes WDS field name (keywd, subsc)
    list_keywords: list[dict[str, str]] = []
    for kw in _extract_from_indexed_object(doc.get("keywd") or {}, "keywd"):
        if kw:
            list_keywords.append(
                {"name": kw, "vocabulary": "World Bank - keywd", "uri": ""}
            )
    for subsc in (doc.get("subsc") or "").split(","):
        s = _clean_text(subsc)
        if s and (
            not deduplicate_combined_fields
            or not any(k["name"] == s for k in list_keywords)
        ):
            list_keywords.append(
                {"name": s, "vocabulary": "World Bank - subsc", "uri": ""}
            )

    # --- Languages ---
    list_languages: list[dict[str, str]] = []
    seen_lang: set[str] = set()
    for lang_src in [doc.get("lang"), doc.get("available_in")]:
        if not lang_src:
            continue
        for lan in str(lang_src).split(","):
            lan = _clean_text(lan)
            if lan and lan.lower() not in seen_lang:
                seen_lang.add(lan.lower())
                code = _get_language_code(lan)
                list_languages.append({"name": lan, "code": code})
    full_avail = doc.get("fullavailablein")
    if isinstance(full_avail, list):
        for lan in full_avail:
            if isinstance(lan, str):
                lan = _clean_text(lan)
                if lan and lan.lower() not in seen_lang:
                    seen_lang.add(lan.lower())
                    list_languages.append(
                        {"name": lan, "code": _get_language_code(lan)}
                    )
    if not list_languages:
        list_languages = [{"name": "English", "code": "en"}]

    # --- URL ---
    url = _clean_text(doc.get("pdfurl") or doc.get("url") or "")

    # --- Relations (alternate formats) ---
    relations: list[dict[str, str]] = []
    for rel_url, rel_type in [
        (doc.get("txturl"), "hasFormat"),
        (doc.get("wrdurl"), "hasFormat"),
    ]:
        if _clean_text(rel_url):
            relations.append({"name": rel_url, "type": rel_type})

    # --- Project relation ---
    projn = _clean_text(doc.get("projn", ""))
    if projn:
        relations.append({"name": projn, "type": "isPartOf"})

    # --- Tags ---
    tags: list[dict[str, str]] = []
    for tag_val, tag_group in [
        (doc.get("majdocty"), "major_document_type"),
        (doc.get("docty"), "document_type"),
        (doc.get("prdln"), "product_line"),
        (doc.get("project_status"), "project_status"),
    ]:
        if _clean_text(tag_val):
            tags.append({"tag": _clean_text(tag_val), "tag_group": tag_group})
    for s in sectr_vals:
        if s:
            tags.append({"tag": s, "tag_group": "sector"})

    # --- Additional (WB-specific) ---
    # Preserve all WDS fields not mapped to schema elements (per schema 5.2.7).
    additional: dict[str, Any] = {}
    for wds_key, add_key in [
        ("action", "wds_action"),
        ("projectid_sort", "wds_projectid_sort"),
        ("docm_id", "wds_docm_id"),
        ("ml_repnme", "wds_ml_repnme"),
        ("origu", "wds_origin_unit"),
        ("owner", "wds_owner"),
        ("projectid", "project_id"),
        ("projn", "project_name"),
        ("lndinstr_exact", "wds_lndinstr_exact"),
        ("prdln_exact", "wds_prdln_exact"),
    ]:
        val = doc.get(wds_key)
        if val is not None and val != "":
            additional[f"additional.{add_key}"] = val
    if doc.get("entityids"):
        additional["additional.wds_entityids"] = doc["entityids"]
    if doc.get("historic_topic"):
        additional["additional.wds_historic_topic"] = doc["historic_topic"]
    if doc.get("totvolnb"):
        additional["additional.wds_totvolnb"] = doc["totvolnb"]
    # Preserve raw countrycode for traceability (e.g. "1W" -> WLD in ref_country)
    if doc.get("countrycode"):
        additional["additional.wds_countrycode"] = doc["countrycode"]
    # Catch-all: preserve any remaining WDS fields not yet mapped
    _ADDITIONAL_KEYS = {
        "action",
        "projectid_sort",
        "docm_id",
        "ml_repnme",
        "origu",
        "owner",
        "projectid",
        "projn",
        "lndinstr_exact",
        "prdln_exact",
        "entityids",
        "historic_topic",
        "totvolnb",
    }
    _MAPPED_KEYS = {
        "id",
        "guid",
        "display_title",
        "docna",
        "repnme",
        "alt_title",
        "docty",
        "majdocty",
        "majtheme",
        "authors",
        "docdt",
        "disclosure_date",
        "datestored",
        "last_modified_date",
        "lupdate",
        "count",
        "countrycode",
        "admreg",
        "geo_regions",
        "theme",
        "topicv3_name",
        "ent_topic",
        "subtopic",
        "teratopic",
        "subsc",
        "sectr",
        "keywd",
        "lang",
        "available_in",
        "fullavailablein",
        "seccl",
        "disclstat",
        "versiontyp",
        "pdfurl",
        "url",
        "txturl",
        "wrdurl",
        "projn",
        "projectid",
        "project_status",
        "prdln",
        "lndinstr",
        "chronical_docm_id",
        "abstracts",
        "volnb",
        "repnb",
        "colti",
        "placeprod",
        "dois",
        "isbn",
        "issn",
    }
    for wds_key, val in doc.items():
        if wds_key in _MAPPED_KEYS or wds_key in _ADDITIONAL_KEYS:
            continue
        if val is None or val == "" or val == [] or val == {}:
            continue
        safe_key = "additional.wds_" + wds_key.replace(".", "_")
        if safe_key not in additional:
            additional[safe_key] = val

    # --- Document description ---
    document_description: dict[str, Any] = {
        "title_statement": title_statement,
        "authors": list_authors,
        "date_created": _format_date(doc.get("docdt")),
        "date_available": _format_date(doc.get("disclosure_date")),
        "date_modified": date_modified,
        "date_published": date_published,
        "identifiers": list_ids,
        "type": _clean_text(doc.get("docty", "")),
        "abstract": _extract_abstract(doc),
        "ref_country": list_countries,
        "geographic_units": geographic_units,
        "spatial_coverage": spatial_coverage,
        "languages": list_languages,
        "volume": str(doc.get("volnb") or ""),
        "number": str(doc.get("repnb") or ""),
        "series": _clean_text(doc.get("colti", "")),
        "publisher_address": _clean_text(doc.get("placeprod", "")),
        "organization": _clean_text(doc.get("prdln", "")),
        "url": url,
        "keywords": list_keywords,
        "themes": list_themes,
        "topics": list_topics,
        "relations": relations,
        "security_classification": _clean_text(doc.get("seccl", "")),
        "access_restrictions": _clean_text(doc.get("disclstat", "")),
        "edition": _clean_text(doc.get("versiontyp", "")),
        "contacts": [
            {
                "name": "Documents and Reports Help Desk",
                "affiliation": "World Bank",
                "email": "documents@worldbank.org",
                "uri": "https://documents.worldbank.org/en/publication/documents-reports/faqs",
            }
        ],
        "usage_terms": "See World Bank Access to Information (https://www.worldbank.org/en/access-to-information)",
    }

    # Add notes for project_status, lndinstr when present
    notes: list[dict[str, str]] = []
    if doc.get("project_status"):
        notes.append({"note": f"Project status: {doc.get('project_status')}"})
    lndinstr = doc.get("lndinstr")
    if lndinstr:
        instr_str = ""
        if isinstance(lndinstr, dict):
            instr_str = " ; ".join(
                v.get("lndinstr", "") for v in lndinstr.values() if isinstance(v, dict)
            )
        elif isinstance(lndinstr, str):
            instr_str = lndinstr
        if instr_str:
            notes.append({"note": f"Lending instrument: {instr_str}"})
    if notes:
        document_description["notes"] = notes

    # --- Metadata information ---
    metadata_information: dict[str, Any] = {
        "title": title,
        "idno": idno,
        "producers": [
            {
                "name": "World Bank",
                "abbr": "WB",
                "affiliation": "World Bank",
                "role": "Source",
            }
        ],
        "production_date": prod_date,
        "version": f"Harvested from WDS API v3 on {prod_date}",
    }

    # --- Provenance ---
    provenance: list[dict[str, Any]] = [
        {
            "origin_description": {
                "harvest_date": harvest,
                "altered": altered,
                "base_url": WB_WDS_API_V3,
                "identifier": guid or idno,
                "date_stamp": doc.get("lupdate", harvest),
                "metadata_namespace": "WB_WDS",
            }
        }
    ]

    # --- Build output ---
    result: dict[str, Any] = {
        "type": "document",
        "metadata_information": metadata_information,
        "document_description": document_description,
        "provenance": provenance,
        "schematype": "document",
    }
    if tags:
        result["tags"] = tags
    if additional:
        result["additional"] = additional

    return result


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def main(
    input: str | None = None,
    output: str | None = None,
    output_dir: str | None = None,
    guid: str | None = None,
    deduplicate: bool = True,
) -> None:
    """
    Transform WDS document(s) to Document Metadata Schema.

    Either:
      - --input=doc.json --output=out.json
      - --guid=969971600710472848 --output_dir=metadata (fetches from API)
    """
    import requests

    if guid and output_dir:
        url = f"{WB_WDS_API_V3}?format=json&fl=*&guid={guid}&apilang=en"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("documents", {})
        if not docs:
            print(f"No document found for guid={guid}")
            return
        doc = next(v for k, v in docs.items() if isinstance(v, dict) and "id" in v)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        idno = str(doc.get("id", guid))
        output_file = out_path / f"document_{idno}.json"
        document_metadata_schema = wds_to_schema(
            doc, deduplicate_combined_fields=deduplicate
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(document_metadata_schema, f, ensure_ascii=False, indent=2)
        print(f"Saved {output_file}")

    elif input and output:
        with open(input, encoding="utf-8") as f:
            data = json.load(f)
        # Handle both raw doc and API response wrapper
        if "documents" in data:
            docs = data["documents"]
            doc = next(v for k, v in docs.items() if isinstance(v, dict) and "id" in v)
        else:
            doc = data
        document_metadata_schema = wds_to_schema(
            doc, deduplicate_combined_fields=deduplicate
        )
        with open(output, "w", encoding="utf-8") as f:
            json.dump(document_metadata_schema, f, ensure_ascii=False, indent=2)
        print(f"Saved {output}")

    else:
        print(
            "Usage: --input=doc.json --output=out.json  OR  --guid=... --output_dir=..."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert WDS API v3 document to Document Metadata Schema."
    )
    parser.add_argument("--input", help="Input JSON file (raw doc or API response)")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--guid", help="Fetch document by guid from WDS API v3")
    parser.add_argument(
        "--output_dir", default="metadata", help="Output directory when using --guid"
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Preserve duplicate values in themes/topics/keywords (full provenance)",
    )
    args = parser.parse_args()
    main(
        input=args.input,
        output=args.output,
        output_dir=args.output_dir,
        guid=args.guid,
        deduplicate=not args.no_deduplicate,
    )
