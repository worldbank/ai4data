"""Anomaly Review App - FastAPI backend for reviewing anomaly explanations with timeseries."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from apps.anomaly_review.feedback import (
    FEEDBACK_SCHEMA,
    export_feedback_csv,
    get_feedback,
    get_feedback_for_item,
    init_feedback_store,
    submit_feedback,
)

app = FastAPI(
    title="Anomaly Explanation Reviewer",
    description="Review LLM-generated anomaly explanations alongside timeseries data",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory store for loaded data (set by load endpoint or CLI)
_review_data: dict | None = None


def set_review_data(data: dict) -> None:
    """Set the review data (called when loading a JSON file)."""
    global _review_data
    _review_data = data


@app.get("/")
async def root():
    """Serve the main review UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Static files not found. Run from app directory."}


@app.get("/api/review")
async def get_review():
    """Get the full review payload (anomalies + timeseries)."""
    if _review_data is None:
        return {"items": [], "count": 0, "message": "No data loaded"}
    return _review_data


@app.get("/api/items")
async def get_items():
    """Get list of anomaly items for navigation."""
    if _review_data is None:
        return []
    items = _review_data.get("items", [])
    return [
        {
            "id": i,
            "indicator_code": it.get("indicator_code", ""),
            "indicator_name": it.get("indicator_name", ""),
            "geography_code": it.get("geography_code", ""),
            "geography_name": it.get("geography_name", ""),
            "window_str": it.get("window_str", ""),
            "classification": it.get("explanation", {}).get("classification", ""),
        }
        for i, it in enumerate(items)
    ]


@app.get("/api/items/{item_id:int}")
async def get_item(item_id: int):
    """Get a single anomaly item with full details."""
    if _review_data is None:
        return None
    items = _review_data.get("items", [])
    if 0 <= item_id < len(items):
        return items[item_id]
    return None


class FeedbackIn(BaseModel):
    item_id: int
    indicator_code: str
    geography_code: str
    window_str: str
    verdict: str  # approved | rejected | needs_review
    comment: str | None = None
    suggested_classification: str | None = None
    facets: dict[str, dict[str, str]] | None = None
    reference_explainer: str | None = None
    best_explainer: str | None = None
    overall_basis: str | None = None

    @field_validator("facets", mode="before")
    @classmethod
    def empty_facets_none(cls, v):
        if v == {} or v is None:
            return None
        return v


@app.post("/api/feedback")
async def post_feedback(fb: FeedbackIn):
    """Submit reviewer feedback for an anomaly item."""
    if fb.verdict not in ("approved", "rejected", "needs_review"):
        raise HTTPException(400, "verdict must be approved, rejected, or needs_review")
    try:
        return submit_feedback(
            item_id=fb.item_id,
            indicator_code=fb.indicator_code,
            geography_code=fb.geography_code,
            window_str=fb.window_str,
            verdict=fb.verdict,
            comment=fb.comment,
            suggested_classification=fb.suggested_classification,
            facets=fb.facets,
            reference_explainer=fb.reference_explainer,
            best_explainer=fb.best_explainer,
            overall_basis=fb.overall_basis,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/feedback")
async def list_feedback(
    item_id: int | None = None,
    indicator_code: str | None = None,
    geography_code: str | None = None,
    window_str: str | None = None,
):
    """List feedback entries, optionally filtered."""
    return get_feedback(
        item_id=item_id,
        indicator_code=indicator_code,
        geography_code=geography_code,
        window_str=window_str,
    )


@app.get("/api/feedback/item")
async def get_item_feedback(
    indicator_code: str,
    geography_code: str,
    window_str: str,
):
    """Get feedback for a specific anomaly item by stable key."""
    fb = get_feedback_for_item(indicator_code, geography_code, window_str)
    return fb if fb is not None else {}


@app.get("/api/feedback/schema")
async def feedback_schema():
    """Get the feedback schema for integration."""
    return FEEDBACK_SCHEMA


@app.get("/api/feedback/export")
async def export_feedback():
    """Export feedback to CSV (returns path)."""
    out = Path("feedback_export.csv")
    export_feedback_csv(out)
    return {"path": str(out), "message": "Exported to feedback_export.csv"}


def run_app(
    data_path: str | Path | None = None,
    feedback_path: str | Path | None = None,
):
    """Run the app, optionally loading review data and feedback store."""
    import uvicorn

    if data_path:
        path = Path(data_path)
        if path.exists():
            import json

            set_review_data(json.loads(path.read_text()))
    elif _review_data is None:
        # Load sample data by default when no path provided
        sample_path = STATIC_DIR / "sample_review.json"
        if sample_path.exists():
            import json

            set_review_data(json.loads(sample_path.read_text()))
    init_feedback_store(feedback_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    feedback_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_app(data_path, feedback_path)
