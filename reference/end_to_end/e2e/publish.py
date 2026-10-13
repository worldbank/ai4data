"""Step 3: publish the records that passed validation.

The default target is a catalog directory: one JSON record per product in
the World Bank schema, one CSV of values per series with SDMX column names,
one schema.org Dataset description per product, and an index. That is the
form in which a catalog serves software and AI systems: stable addresses,
machine-readable records, and a description search engines read.

With a catalog URL and an API key, the same records are sent to a NADA
instance through its catalog administration API, which takes the record
in the same schema. The request is built and, in dry-run mode, written to
the report without being sent, so a reader sees what the call is.
"""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .records import Record, first, record_id

ORGANIZATION = "National Statistics Office (example)"
BASE_URL = "https://stats.example"


@dataclass
class Published:
    catalog_dir: Path
    records: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    nada: list[dict[str, Any]] = field(default_factory=list)


def _schema_org(kind: str, record: Record, values_file: str | None) -> dict[str, Any]:
    if kind == "indicator":
        sd = record["series_description"]
        idno = sd["idno"]
        periods = sd.get("time_periods") or [{}]
        out = {
            "@context": "https://schema.org",
            "@type": "Dataset",
            "identifier": idno,
            "name": sd["name"],
            "description": sd.get("definition_long", ""),
            "url": f"{BASE_URL}/series/{idno}",
            "creator": {"@type": "Organization", "name": ORGANIZATION},
            "dateModified": sd.get("date_last_update"),
            "temporalCoverage": f"{periods[0].get('start', '')}/{periods[0].get('end', '')}",
            "variableMeasured": {
                "@type": "PropertyValue",
                "name": sd["name"],
                "unitText": sd.get("measurement_unit"),
            },
            "keywords": [k["name"] for k in sd.get("keywords", [])],
        }
        if sd.get("license"):
            out["license"] = sd["license"][0].get("uri")
        if values_file:
            out["distribution"] = [
                {
                    "@type": "DataDownload",
                    "encodingFormat": "text/csv",
                    "contentUrl": f"{BASE_URL}/series/{idno}.csv",
                }
            ]
        return out
    ts = record["study_desc"]["title_statement"]
    info = record["study_desc"].get("study_info", {})
    coll = (info.get("coll_dates") or [{}])[0]
    out = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "identifier": ts["idno"],
        "name": ts["title"],
        "description": info.get("abstract", ""),
        "url": first(
            record, "study_desc.data_access.dataset_availability.access_place_url"
        )
        or f"{BASE_URL}/microdata/{ts['idno']}",
        "creator": {
            "@type": "Organization",
            "name": (record["study_desc"].get("authoring_entity") or [{}])[0].get(
                "name", ORGANIZATION
            ),
        },
        "temporalCoverage": f"{coll.get('start', '')}/{coll.get('end', '')}",
        "keywords": [k["keyword"] for k in info.get("keywords", [])],
    }
    doi = next(
        (i["identifier"] for i in ts.get("identifiers", []) if i.get("type") == "doi"),
        None,
    )
    if doi:
        out["sameAs"] = f"https://doi.org/{doi}"
    return out


def publish_catalog(
    passed: list[tuple[str, Record]], values: list[dict[str, str]], out_dir: Path
) -> Published:
    catalog = out_dir / "catalog"
    for sub in ("records", "values", "jsonld"):
        (catalog / sub).mkdir(parents=True, exist_ok=True)
    pub = Published(catalog_dir=catalog)
    index = []
    for kind, record in passed:
        idno = record_id(kind, record)
        rec_path = catalog / "records" / f"{idno}.json"
        rec_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        pub.files.append(str(rec_path.relative_to(out_dir)))
        values_file = None
        if kind == "indicator":
            rows = [v for v in values if v["SERIES"] == idno]
            if rows:
                values_file = catalog / "values" / f"{idno}.csv"
                with values_file.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                pub.files.append(str(values_file.relative_to(out_dir)))
        jsonld = catalog / "jsonld" / f"{idno}.jsonld"
        jsonld.write_text(
            json.dumps(
                _schema_org(kind, record, values_file and values_file.name),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        pub.files.append(str(jsonld.relative_to(out_dir)))
        name = (
            first(record, "series_description.name")
            if kind == "indicator"
            else first(record, "study_desc.title_statement.title")
        )
        index.append(
            {
                "idno": idno,
                "type": kind,
                "name": name,
                "record": f"records/{idno}.json",
                "values": f"values/{idno}.csv" if values_file else None,
            }
        )
        pub.records.append(idno)
    (catalog / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pub.files.append(str((catalog / "index.json").relative_to(out_dir)))
    return pub


def nada_requests(
    passed: list[tuple[str, Record]], base_url: str
) -> list[dict[str, Any]]:
    """The catalog administration API calls that create each record in NADA.

    A series record goes to POST /datasets/create/timeseries and a survey
    record to POST /datasets/create/survey/{idno}; the body is the record in
    the same schema, with the publication flags NADA reads at the top level.
    """
    base = base_url.rstrip("/")
    out = []
    for kind, record in passed:
        idno = record_id(kind, record)
        if kind == "indicator":
            url = f"{base}/index.php/api/datasets/create/timeseries"
        else:
            url = f"{base}/index.php/api/datasets/create/survey/{idno}"
        body = {
            "idno": idno,
            "repositoryid": "central",
            "access_policy": "open",
            "published": 1,
            "overwrite": "yes",
        }
        body.update(record)
        out.append({"idno": idno, "method": "POST", "url": url, "body": body})
    return out


def publish_to_nada(
    passed: list[tuple[str, Record]], base_url: str, api_key: str | None, dry_run: bool
) -> list[dict[str, Any]]:
    results = []
    for req in nada_requests(passed, base_url):
        entry = {"idno": req["idno"], "url": req["url"]}
        if dry_run or not api_key:
            entry["status"] = "dry-run" if dry_run else "skipped: no API key"
            results.append(entry)
            continue
        data = json.dumps(req["body"]).encode("utf-8")
        request = urllib.request.Request(
            req["url"],
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "X-API-KEY": api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                entry["status"] = f"sent: HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            entry["status"] = f"failed: HTTP {exc.code}"
        except urllib.error.URLError as exc:
            entry["status"] = f"failed: {exc.reason}"
        results.append(entry)
    return results
