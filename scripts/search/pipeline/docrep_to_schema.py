"""
docrep_to_schema.py
=================
Fetch metadata from the World Bank Documents and Reports API and convert it
to the NADA document schema. Saves each document as metadata/document_<idno>.json.

Ported from DOCREP_to_NADA_ME.R (Author: Olivier Dupriez, World Bank).

Usage:
  python docrep_to_schema.py --output_dir=metadata --docty_key=620265 --rows=100
  python docrep_to_schema.py --output_dir=metadata --max_docs=10  # Preview with 10 docs
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from countrycode import countrycode

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

WB_SEARCH_API = "https://search.worldbank.org/api/v2/wds"

# World Bank administrative region codes (from DOCREP_to_NADA_ME.R)
AFR_CODES = frozenset(
    {
        "AFR",
        "AGO",
        "ETH",
        "NER",
        "BEN",
        "GAB",
        "NGA",
        "BWA",
        "GMB",
        "RWA",
        "BFA",
        "GHA",
        "STP",
        "BDI",
        "GIN",
        "SEN",
        "CPV",
        "GBI",
        "SYC",
        "CMR",
        "KEN",
        "SLE",
        "CAF",
        "LSO",
        "SOM",
        "TCD",
        "LBR",
        "ZAF",
        "COM",
        "MDG",
        "SSD",
        "COD",
        "COG",
        "MWI",
        "SDN",
        "TZA",
        "CIV",
        "MRT",
        "TGO",
        "EQG",
        "MUS",
        "UGA",
        "ERI",
        "MOZ",
        "ZMB",
        "SWZ",
        "NAM",
        "ZWE",
    }
)
EAP_CODES = frozenset(
    {
        "EAP",
        "ASM",
        "KOR",
        "PHL",
        "AUS",
        "LAO",
        "WSM",
        "BRN",
        "MAC",
        "CHN",
        "SGP",
        "KHM",
        "MYS",
        "SLB",
        "MHL",
        "TWN",
        "FJI",
        "FSM",
        "THA",
        "PYF",
        "MNG",
        "TLS",
        "GUM",
        "MMR",
        "PNG",
        "HKG",
        "NRU",
        "TON",
        "IDN",
        "NCL",
        "TUV",
        "JPN",
        "NZL",
        "VTU",
        "KIR",
        "MNP",
        "VNM",
        "PKR",
        "PLW",
    }
)
ECA_CODES = frozenset(
    {
        "ECA",
        "ALB",
        "AND",
        "ARM",
        "AUT",
        "AZE",
        "BLR",
        "BEL",
        "BIH",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FRO",
        "FIN",
        "FRA",
        "GEO",
        "DEU",
        "GIB",
        "GRC",
        "GRL",
        "HUN",
        "ISL",
        "IRL",
        "IMN",
        "ITA",
        "KAZ",
        "XKX",
        "KGZ",
        "LVA",
        "LIE",
        "LTU",
        "LUX",
        "MDA",
        "MCO",
        "MNE",
        "NLD",
        "MKD",
        "NOR",
        "POL",
        "PRT",
        "ROU",
        "RUS",
        "SMR",
        "SRB",
        "SVK",
        "SVN",
        "ESP",
        "SWE",
        "CHE",
        "TJK",
        "TUR",
        "TKM",
        "UKR",
        "GBR",
        "UZB",
    }
)
LAC_CODES = frozenset(
    {
        "LAC",
        "ATG",
        "ARG",
        "ABW",
        "BHS",
        "BRB",
        "BLZ",
        "BOL",
        "BRA",
        "VGB",
        "CYM",
        "CHL",
        "COL",
        "CRI",
        "CUB",
        "CUW",
        "DMA",
        "DOM",
        "ECU",
        "SLV",
        "GRD",
        "GTM",
        "GUY",
        "HTI",
        "HND",
        "JAM",
        "MEX",
        "NIC",
        "PAN",
        "PRY",
        "PER",
        "PRI",
        "KNA",
        "LCA",
        "MAF",
        "VCT",
        "SUR",
        "TTO",
        "TCA",
        "URY",
        "VEN",
        "VIR",
    }
)
MNA_CODES = frozenset(
    {
        "MNA",
        "DZA",
        "BHR",
        "DJI",
        "EGY",
        "IRN",
        "IRQ",
        "ISR",
        "JOR",
        "KWT",
        "LBN",
        "LBY",
        "MLT",
        "MAR",
        "OMN",
        "QAT",
        "SAU",
        "SYR",
        "TUN",
        "ARE",
        "PSE",
        "YEM",
    }
)
NAR_CODES = frozenset({"NAR", "BMU", "CAN", "USA"})
SAR_CODES = frozenset({"SAR", "AFG", "BGD", "BTN", "IND", "MDV", "NPL", "PAK", "LKA"})

PRWP_DISCLAIMER = (
    "The Policy Research Working Paper Series disseminates the findings of work in "
    "progress to encourage the exchange of ideas about development issues. An objective "
    "of the series is to get the findings out quickly, even if the presentations are "
    "less than fully polished. The papers carry the names of the authors and should be "
    "cited accordingly. The findings, interpretations, and conclusions expressed in "
    "this paper are entirely those of the authors. They do not necessarily represent "
    "the views of the International Bank for Reconstruction and Development/World Bank "
    "and its affiliated organizations, or those of the Executive Directors of the World "
    "Bank or the governments they represent."
)

LANGUAGE_CODES = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "russian": "ru",
}

REGION_SPECIAL_CASES = {
    "World": "WLD",
    "Africa": "AFR",
    "East Asia and Pacific": "EAP",
    "Europe and Central Asia": "ECA",
    "Latin America and Caribbean": "LAC",
    "Middle East and North Africa": "MNA",
    "South Asia": "SAR",
}

REGION_NAMES = {
    "AFR": "Africa",
    "EAP": "East Asia and Pacific",
    "ECA": "Europe and Central Asia",
    "LAC": "Latin America and Caribbean",
    "MNA": "Middle East and North Africa",
    "NAR": "North America",
    "SAR": "South Asia",
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _clean_text(text: str | None) -> str:
    """Normalize whitespace: collapse newlines and multiple spaces, strip."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _format_date(value: str | None) -> str:
    """Format date string to YYYY/MM/DD. Returns empty string if invalid."""
    if not value:
        return ""
    try:
        # Handle ISO format like "2025-03-10T00:00:00Z"
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y/%m/%d")
    except (ValueError, TypeError):
        return ""


def _get_country_code(name: str) -> str:
    """Map country/region name to ISO3 code with WB region fallbacks."""
    if not name or not name.strip():
        return "-"
    name = name.strip()
    if name in REGION_SPECIAL_CASES:
        return REGION_SPECIAL_CASES[name]
    try:
        result = countrycode(
            [name], origin="country.name.en.regex", destination="iso3c"
        )
        if result and result[0]:
            return result[0]
    except Exception:
        pass
    return "-"


# ------------------------------------------------------------------------------
# API
# ------------------------------------------------------------------------------


def fetch_page(
    offset: int,
    rows: int = 100,
    docty_key: str | None = None,
    doctype: str | None = None,
    strdate: str | None = None,
    enddate: str | None = None,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> dict[str, Any]:
    """Fetch a single page from the World Bank Documents and Reports API."""
    params: dict[str, Any] = {
        "format": "json",
        "rows": rows,
        "os": offset,
        "srt": "docdt",
        "order": "desc",
    }
    if docty_key:
        params["docty_key"] = docty_key
    elif doctype:
        params["docty_exact"] = doctype
        params["lang_exact"] = "English"
    if strdate:
        params["strdate"] = strdate
    if enddate:
        params["enddate"] = enddate

    for attempt in range(retries):
        try:
            resp = requests.get(WB_SEARCH_API, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt + 1}/{retries} after error: {e}")
            time.sleep(retry_delay)
    raise RuntimeError("Unreachable")


def iter_documents(
    docty_key: str | None = None,
    doctype: str | None = None,
    strdate: str | None = None,
    enddate: str | None = None,
    rows: int = 100,
    max_docs: int = 0,
    start_offset: int = 0,
):
    """Yield raw document dicts from the API, paginating as needed."""
    offset = start_offset
    total: int | None = None

    while True:
        if max_docs > 0 and offset >= max_docs:
            break
        batch_rows = min(rows, max_docs - offset) if max_docs > 0 else rows

        data = fetch_page(
            offset=offset,
            rows=batch_rows,
            docty_key=docty_key,
            doctype=doctype,
            strdate=strdate,
            enddate=enddate,
        )

        if total is None:
            total = int(data.get("total", 0))
            if total:
                print(f"  Total available: {total} documents")
                if max_docs > 0:
                    print(f"  Fetching up to: {max_docs}")

        docs = data.get("documents", {})
        if not docs:
            break

        count = 0
        for key, val in docs.items():
            if key == "facets" or not isinstance(val, dict):
                continue
            if "id" not in val:
                continue
            count += 1
            yield val

        offset += count
        total_str = f"/{total}" if total else ""
        print(f"  Fetched {offset}{total_str} documents...", end="\r")

        if count < batch_rows:
            break

    print()


# ------------------------------------------------------------------------------
# Schema transformation
# ------------------------------------------------------------------------------


def transform_to_schema(
    doc: dict[str, Any], docty_key: str | None = None
) -> dict[str, Any]:
    """Convert a raw API document to the NADA document schema."""
    idno = str(doc.get("id", ""))

    # Title
    title = _clean_text(doc.get("display_title") or "")

    # Alternate title
    alt_titles: list[str] = []
    alt_title_raw = doc.get("alt_title")
    if alt_title_raw is not None:
        if isinstance(alt_title_raw, list):
            for a in alt_title_raw:
                if isinstance(a, dict) and "alt_title" in a:
                    alt_titles.append(_clean_text(a.get("alt_title", "")))
                else:
                    alt_titles.append(_clean_text(str(a)))
        elif isinstance(alt_title_raw, str):
            alt_titles.append(_clean_text(alt_title_raw))
    alt_title = " / ".join(t for t in alt_titles if t).strip()
    if alt_title.startswith(" / "):
        alt_title = alt_title[3:]

    # Identifiers
    list_ids: list[dict[str, str]] = [
        {"type": "World Bank D&R", "identifier": str(doc.get("id", ""))},
        {"type": "World Bank ID", "identifier": str(doc.get("docm_id") or "")},
        {
            "type": "World Bank Chronical ID",
            "identifier": str(doc.get("chronical_docm_id") or ""),
        },
        {"type": "DOI", "identifier": str(doc.get("dois") or "")},
    ]
    if doc.get("isbn"):
        list_ids.append({"type": "ISBN", "identifier": str(doc["isbn"])})
    if doc.get("issn"):
        list_ids.append({"type": "ISSN", "identifier": str(doc["issn"])})

    # Authors
    list_authors: list[dict[str, str]] = []
    authors_raw = doc.get("authors", {})
    if isinstance(authors_raw, dict):
        for v in authors_raw.values():
            if not isinstance(v, dict):
                continue
            author_val = v.get("author")
            if author_val is None:
                continue
            full_name = (
                author_val[0]
                if isinstance(author_val, (list, tuple))
                else str(author_val)
            )
            if full_name:
                list_authors.append(
                    {
                        "first_name": "",
                        "initial": "",
                        "last_name": "",
                        "affiliation": "",
                        "full_name": full_name,
                    }
                )

    # Themes
    list_themes: list[dict[str, str]] = []
    theme_str = doc.get("theme")
    if theme_str:
        for t in re.split(r",", str(theme_str)):
            thm = _clean_text(t).lower()
            if thm:
                list_themes.append(
                    {
                        "id": "",
                        "name": thm,
                        "parent_id": "",
                        "vocabulary": "World Bank",
                        "uri": "",
                    }
                )

    # Keywords
    list_keywords: list[dict[str, str]] = []
    keywd_raw = doc.get("keywd")
    if keywd_raw:
        keywd_str = ""
        if isinstance(keywd_raw, list) and keywd_raw:
            first = keywd_raw[0]
            if isinstance(first, dict) and "keywd" in first:
                keywd_str = str(first.get("keywd", ""))
            elif isinstance(first, str):
                keywd_str = first
        elif isinstance(keywd_raw, str):
            keywd_str = keywd_raw

        if keywd_str:
            sep = ";" if ";" in keywd_str else ","
            for kw in keywd_str.split(sep):
                kwd = _clean_text(kw).lower().replace("\n", "")
                if kwd:
                    list_keywords.append(
                        {"id": "", "name": kwd, "vocabulary": "World Bank"}
                    )

    if docty_key == "563787":  # WDR
        has_wdr = any(k.get("name") == "WDR" for k in list_keywords)
        if not has_wdr:
            list_keywords.append({"id": "", "name": "WDR", "vocabulary": "World Bank"})

    # Abstract
    abstracts = doc.get("abstracts") or {}
    abstract = _clean_text(abstracts.get("cdata!", abstracts.get("#text", "")))

    # Languages
    list_languages: list[dict[str, str]] = []
    available_in = doc.get("available_in")
    if available_in:
        for lan in re.split(r",", str(available_in)):
            lan = _clean_text(lan).lower()
            if lan:
                code = LANGUAGE_CODES.get(lan, "")
                list_languages.append({"name": lan, "code": code})
    if not list_languages:
        list_languages = [{"name": "english", "code": "en"}]

    # Countries
    list_countries: list[dict[str, str]] = []
    count_str = doc.get("count")
    if count_str:
        count_str = str(count_str).replace(", ", " - ")
        for cou in re.split(r",", count_str):
            cou = _clean_text(cou)
            if not cou:
                continue
            ccode = _get_country_code(cou)
            list_countries.append({"name": cou, "code": ccode})
        for c in list_countries:
            c["name"] = c["name"].replace(" - ", ", ")

    # Geographic coverage
    geo_coverage = ""
    admreg = doc.get("admreg")
    if admreg:
        regs = list(dict.fromkeys(re.split(r",", str(admreg))))  # unique order
        regs = [r.strip() for r in regs if r.strip()]
        geo_coverage = " ; ".join(regs)

    if not geo_coverage and list_countries:
        regions_found: set[str] = set()
        for c in list_countries:
            code = c.get("code", "")
            if code in AFR_CODES:
                regions_found.add("AFR")
            elif code in EAP_CODES:
                regions_found.add("EAP")
            elif code in ECA_CODES:
                regions_found.add("ECA")
            elif code in LAC_CODES:
                regions_found.add("LAC")
            elif code in MNA_CODES:
                regions_found.add("MNA")
            elif code in NAR_CODES:
                regions_found.add("NAR")
            elif code in SAR_CODES:
                regions_found.add("SAR")
        parts = [REGION_NAMES[r] for r in sorted(regions_found)]
        geo_coverage = " ; ".join(parts)

    # Disclaimer
    disclaimer = PRWP_DISCLAIMER if docty_key == "620265" else ""

    # Metadata version
    today = datetime.now().strftime("%Y-%m-%d")
    metadata_version = f"Python script, from D&R API, generated on {today}"

    # Build NADA document
    this_doc: dict[str, Any] = {
        "type": "document",
        "metadata_information": {
            "title": title,
            "idno": idno,
            "producers": [
                {
                    "name": "Library",
                    "abbr": "ITSKI",
                    "affiliation": "World Bank",
                    "role": "Curation",
                }
            ],
            "production_date": datetime.now().strftime("%Y/%m/%d"),
            "version": metadata_version,
        },
        "document_description": {
            "title_statement": {
                "idno": idno,
                "title": title,
                "sub_title": "",
                "alternate_title": alt_title,
                "translated_title": "",
            },
            "authors": list_authors,
            "date_created": _format_date(doc.get("docdt")),
            "date_available": _format_date(doc.get("disclosure_date")),
            "date_published": _format_date(doc.get("publishtoextweb_dt")),
            "identifiers": list_ids,
            "type": str(doc.get("docty", "")),
            "abstract": abstract,
            "ref_country": list_countries,
            "spatial_coverage": geo_coverage,
            "languages": list_languages,
            "volume": str(doc.get("volnb") or ""),
            "number": str(doc.get("repnb") or ""),
            "url": str(doc.get("pdfurl") or doc.get("url") or ""),
            "contacts": [
                {
                    "name": "Documents and Reports Help Desk",
                    "affiliation": "World Bank",
                    "email": "documents@worldbank.org",
                    "uri": "https://documents.worldbank.org/en/publication/documents-reports/faqs",
                }
            ],
            "usage_terms": "See World Bank Access to Information (https://www.worldbank.org/en/access-to-information)",
            "disclaimer": disclaimer,
            "security_classification": str(doc.get("seccl") or ""),
            "keywords": list_keywords,
            "themes": list_themes,
            "reproducibility": {
                "statement": "",
                "links": [{"uri": "", "description": ""}],
            },
            "pricing": "Free",
        },
        "provenance": [
            {
                "origin_description": {
                    "harvest_date": datetime.now().strftime("%Y-%m-%d"),
                    "altered": False,
                    "base_url": "https://documents.worldbank.org/en/publication/documents-reports/api",
                    "identifier": "",
                    "date_stamp": datetime.now().isoformat(),
                    "metadata_namespace": "WB_DOCREP",
                }
            }
        ],
        "schematype": "document",
    }

    return this_doc


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main(
    output_dir: str = "metadata",
    docty_key: str | None = None,
    strdate: str | None = None,
    enddate: str | None = None,
    rows: int = 100,
    max_docs: int = 0,
    offset: int = 0,
) -> None:
    """Fetch documents from the D&R API, transform to NADA schema, and save as JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    docty_key_str = docty_key or ""
    if docty_key_str:
        print(
            f"Fetching documents (docty_key={docty_key_str}) from World Bank D&R API..."
        )
    else:
        print("Fetching documents from World Bank D&R API...")

    saved = 0
    for i, doc in enumerate(
        iter_documents(
            docty_key=docty_key_str or None,
            strdate=strdate or None,
            enddate=enddate or None,
            rows=rows,
            max_docs=max_docs if max_docs > 0 else 0,
            start_offset=offset,
        )
    ):
        idno = str(doc.get("id", ""))
        if not idno:
            continue
        try:
            nada_doc = transform_to_schema(doc, docty_key=docty_key_str or None)
            file_path = out_path / f"document_{idno}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(nada_doc, f, ensure_ascii=False, indent=2)
            saved += 1
            print(f"  Saved document_{idno}.json ({saved} total)")
        except Exception as e:
            print(f"  Error processing document {idno}: {e}")

    print(f"\nDone. Saved {saved} documents to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch metadata from WB Documents and Reports API and save as NADA JSON."
    )
    parser.add_argument(
        "--output_dir",
        default="metadata",
        help="Output directory for document_<idno>.json files (default: metadata)",
    )
    parser.add_argument(
        "--docty_key",
        default=None,
        help="Document type key: 563787=WDR, 620265=PRWP (optional)",
    )
    parser.add_argument(
        "--strdate", default=None, help="Start date YYYY-MM-DD (optional)"
    )
    parser.add_argument(
        "--enddate", default=None, help="End date YYYY-MM-DD (optional)"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Batch size per API request (default: 100)",
    )
    parser.add_argument(
        "--max_docs", type=int, default=0, help="Max documents to fetch (0=all)"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Starting offset for pagination"
    )
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        docty_key=args.docty_key,
        strdate=args.strdate,
        enddate=args.enddate,
        rows=args.rows,
        max_docs=args.max_docs,
        offset=args.offset,
    )
