"""
WDS Schema Mapper Mini App
==========================
Flask server that fetches WDS document metadata, transforms it via wds_to_schema,
and serves a single-page UI to display original and mapped schema side-by-side.

Run: uv run python scripts/metadata/wds_mapper_app.py
Open: http://localhost:5050
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure scripts/metadata is on path when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from flask import Flask, jsonify, request, send_from_directory

# Import from same package
from wds_to_schema import wds_to_schema
from wds_schema_validator import validate_wds_to_schema

WB_WDS_API = "https://search.worldbank.org/api/v3/wds"
GUID_PATTERN = re.compile(r"guid=(\d+)")

app = Flask(__name__, static_folder=Path(__file__).parent)


# CORS for local dev
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def extract_guid(value: str) -> str | None:
    """Extract guid from full WDS URL or return bare guid if numeric."""
    value = (value or "").strip()
    if not value:
        return None
    match = GUID_PATTERN.search(value)
    if match:
        return match.group(1)
    if value.isdigit():
        return value
    return None


def fetch_wds_document(guid: str) -> dict:
    """Fetch document from WDS API by guid."""
    url = f"{WB_WDS_API}?format=json&fl=*&guid={guid}&apilang=en"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    docs = data.get("documents", {})
    if not docs:
        raise ValueError(f"No document found for guid={guid}")
    doc = next(
        (v for k, v in docs.items() if isinstance(v, dict) and "id" in v),
        None,
    )
    if not doc:
        raise ValueError(f"No valid document in response for guid={guid}")
    return doc


@app.route("/")
def index():
    """Serve the mapper HTML page."""
    return send_from_directory(Path(__file__).parent, "wds_mapper.html")


@app.route("/api/map")
def api_map():
    """
    GET /api/map?url=... or ?guid=...
    Returns { original, mapped } for the WDS document.
    """
    url_param = request.args.get("url", "").strip()
    guid_param = request.args.get("guid", "").strip()
    guid = extract_guid(url_param) or (guid_param if guid_param.isdigit() else None)
    if not guid:
        return jsonify(
            {"error": "Provide url or guid (e.g. ?url=... or ?guid=969971600710472848)"}
        ), 400
    try:
        doc = fetch_wds_document(guid)
        deduplicate = request.args.get("deduplicate", "1").lower() not in ("0", "false", "no")
        mapped = wds_to_schema(doc, deduplicate_combined_fields=deduplicate)
        payload: dict = {"original": doc, "mapped": mapped}
        if request.args.get("validate"):
            validation = validate_wds_to_schema(doc, mapped)
            payload["validation"] = validation
        return jsonify(payload)
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to fetch from WDS API: {e}"}), 502
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


def main():
    app.run(host="0.0.0.0", port=5051, debug=False)


if __name__ == "__main__":
    main()
