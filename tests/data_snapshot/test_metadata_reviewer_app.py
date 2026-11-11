"""Exercise human review, nested edits, validation, and navigation in Streamlit."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


@pytest.fixture
def reviewer_app(tmp_path: Path) -> AppTest:
    """Open two production-style image and metadata pairs in the reviewer."""
    snapshots = tmp_path / "snapshots"
    generated = tmp_path / "metadata"
    snapshots.mkdir()
    generated.mkdir()
    for snapshot_id in ("alpha", "beta"):
        (snapshots / f"{snapshot_id}.png").write_bytes(_PNG)
        (generated / f"{snapshot_id}.json").write_text(
            json.dumps({"title": f"Generated {snapshot_id}"}), encoding="utf-8"
        )
    args = ["--snapshots-root", str(snapshots)]
    app = AppTest.from_string(
        "from ai4data.data_snapshot.metadata_reviewer.app import main\n"
        f"main({args!r})\n",
        default_timeout=30,
    ).run()
    assert not app.exception
    return app


def test_edit_reference_and_guarded_navigation(
    reviewer_app: AppTest, tmp_path: Path
) -> None:
    """Retain drafts across views, guard navigation, and save separate copies."""
    app = reviewer_app
    original_generated = (tmp_path / "metadata" / "alpha.json").read_bytes()
    assert app.session_state["metadata_reviewer_item_id"] == "alpha"
    assert app.session_state["metadata_reviewer_working"] == {
        "title": "Generated alpha"
    }
    app.text_input(key="metadata_reviewer_widget:alpha:title").set_value(
        "Reviewed alpha"
    ).run()
    assert any(item.value == ":orange[**Title**]" for item in app.markdown)

    for section in ("Temporal", "Geographic", "Context", "Structure", "Overview"):
        app.button_group(key="metadata_reviewer_widget:alpha:section").set_value(
            section
        ).run()
        assert not app.exception
    assert app.session_state["metadata_reviewer_working"] == {"title": "Reviewed alpha"}

    app.button_group(key="metadata_reviewer_widget:alpha:view").set_value(
        "Generated reference"
    ).run()
    assert any(
        json.loads(item.value) == {"title": "Generated alpha"} for item in app.json
    )
    app.button_group(key="metadata_reviewer_widget:alpha:view").set_value(
        "Edit metadata"
    ).run()
    assert app.text_input(key="metadata_reviewer_widget:alpha:title").value == (
        "Reviewed alpha"
    )

    app.button(key="review_next").click().run()
    assert app.session_state["metadata_reviewer_item_id"] == "alpha"
    assert any("Unsaved edits for alpha" in warning.value for warning in app.warning)
    app.button(key="review_guard_cancel").click().run()
    assert app.text_input(key="metadata_reviewer_widget:alpha:title").value == (
        "Reviewed alpha"
    )
    app.button(key="review_next").click().run()
    app.button(key="review_guard_save").click().run()
    assert not app.exception
    assert app.session_state["metadata_reviewer_item_id"] == "beta"
    assert json.loads((tmp_path / "reviewed" / "alpha.json").read_text()) == {
        "title": "Reviewed alpha"
    }
    assert (tmp_path / "metadata" / "alpha.json").read_bytes() == original_generated

    app.text_input(key="metadata_reviewer_widget:beta:title").set_value("Discard").run()
    app.button(key="review_previous").click().run()
    app.button(key="review_guard_discard").click().run()
    assert app.text_input(key="metadata_reviewer_widget:alpha:title").value == (
        "Reviewed alpha"
    )
    assert not (tmp_path / "reviewed" / "beta.json").exists()
    app.button(key="review_next").click().run()
    assert app.text_input(key="metadata_reviewer_widget:beta:title").value == (
        "Generated beta"
    )
    app.button(key="review_save").click().run()
    assert not app.exception
    assert json.loads((tmp_path / "reviewed" / "beta.json").read_text()) == {
        "title": "Generated beta"
    }


def test_nested_edits_and_validation(reviewer_app: AppTest, tmp_path: Path) -> None:
    """Add, reorder, and remove nested values without saving invalid drafts."""
    app = reviewer_app
    app.button_group(key="metadata_reviewer_widget:alpha:section").set_value(
        "Structure"
    ).run()
    app.text_input(key="metadata_reviewer_widget:alpha:variables[0].name").set_value(
        "Population"
    ).run()
    app.button(key="metadata_reviewer_widget:alpha:variables:add").click().run()
    app.button(key="review_save").click().run()
    assert not app.exception
    assert any("not saved" in error.value for error in app.error)
    assert not (tmp_path / "reviewed").exists()
    assert app.session_state["metadata_reviewer_item_id"] == "alpha"
    app.text_input(key="metadata_reviewer_widget:alpha:variables[1].name").set_value(
        "Income"
    ).run()
    app.button(key="metadata_reviewer_widget:alpha:variables[1]:up").click().run()
    assert app.session_state["metadata_reviewer_working"]["variables"] == [
        {"name": "Income"},
        {"name": "Population"},
    ]
    app.button(key="metadata_reviewer_widget:alpha:variables[1]:remove").click().run()
    app.text_input(
        key="metadata_reviewer_widget:alpha:variables[0].unit.source_text"
    ).set_value("Thousands of dollars").run()
    app.text_input(
        key="metadata_reviewer_widget:alpha:variables[0].unit.multiplier_exponent"
    ).set_value("3").run()
    app.button(key="review_save").click().run()
    assert not app.exception
    assert json.loads((tmp_path / "reviewed" / "alpha.json").read_text()) == {
        "title": "Generated alpha",
        "variables": [
            {
                "name": "Income",
                "unit": {
                    "source_text": "Thousands of dollars",
                    "multiplier_exponent": 3,
                },
            }
        ],
    }


def test_invalid_input_displays_error(reviewer_app: AppTest, tmp_path: Path) -> None:
    """Report malformed input without a widget crash or a reviewed output."""
    (tmp_path / "metadata" / "beta.json").write_text('{"variables": "invalid"}')
    app = reviewer_app.button(key="review_next").click().run()
    assert not app.exception
    assert any("input metadata is invalid" in error.value for error in app.error)
    assert not (tmp_path / "reviewed").exists()
