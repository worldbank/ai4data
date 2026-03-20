"""
WDS Schema Mapper Mini App
==========================
FastAPI server that fetches WDS document metadata, transforms it via wds_to_schema,
and serves a single-page UI to display original and mapped schema side-by-side.

Run: uv run python scripts/metadata/wds_mapper_app.py
     or: uv run uvicorn scripts.metadata.wds_mapper_app:app --reload --port 5051
Open: http://localhost:5051
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure scripts/metadata is on path when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Import from same package
from wds_to_schema import wds_to_schema
from wds_schema_validator import validate_wds_to_schema

WB_WDS_API = "https://search.worldbank.org/api/v3/wds"
GUID_PATTERN = re.compile(r"guid=(\d+)")

app = FastAPI(
    title="WDS Schema Mapper", description="Map WDS metadata to NADA document schema"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)


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


@app.get("/")
def index():
    """Serve the mapper HTML page."""
    return FileResponse(Path(__file__).parent / "wds_mapper.html")


@app.get("/api/map")
def api_map(
    url: str | None = Query(None, description="Full WDS URL or guid"),
    guid: str | None = Query(None, description="Document guid"),
    validate: str | None = Query(None, description="Run validation (any value)"),
    deduplicate: str = Query(
        "1", description="Deduplicate themes/topics/keywords (1/0)"
    ),
):
    """
    Map WDS document to NADA schema.
    Provide either url or guid.
    """
    resolved_guid = extract_guid(url or "") or (
        guid if guid and guid.isdigit() else None
    )
    if not resolved_guid:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Provide url or guid (e.g. ?url=... or ?guid=969971600710472848)"
            },
        )
    try:
        doc = fetch_wds_document(resolved_guid)
        do_deduplicate = deduplicate.lower() not in ("0", "false", "no")
        mapped = wds_to_schema(doc, deduplicate_combined_fields=do_deduplicate)
        payload: dict = {"original": doc, "mapped": mapped}
        if validate:
            payload["validation"] = validate_wds_to_schema(doc, mapped)
        return payload
    except requests.RequestException as e:
        return JSONResponse(
            status_code=502, content={"error": f"Failed to fetch from WDS API: {e}"}
        )
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5051)


if __name__ == "__main__":
    main()
