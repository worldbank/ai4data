"""Check the read-only metadata example and recorded cookbook artifacts."""

import importlib.util
import json
from pathlib import Path

import pytest

from ai4data.data_snapshot import DataSnapshotMetadata

ASSETS = (
    Path(__file__).resolve().parents[1]
    / "website/static/cookbook-files/data-snapshot-end-to-end"
)
SPEC = importlib.util.spec_from_file_location(
    "snapshot_inspection", ASSETS / "inspect_metadata.py"
)
assert SPEC is not None and SPEC.loader is not None
INSPECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTION)


def test_inspection_preserves_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validate a record without modifying its source.

    Parameters
    ----------
    tmp_path : Path
        Temporary test directory.
    capsys : pytest.CaptureFixture[str]
        Captured command output.
    """
    path = tmp_path / "record.json"
    original = '{"title": "Example", "document_label": null}\n'
    path.write_text(original, encoding="utf-8")
    assert INSPECTION.main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"title": "Example"}
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("content", ['{"unknown_field": 1}', '{"title":'])
def test_inspection_rejects_invalid_input(tmp_path: Path, content: str) -> None:
    """Reject malformed JSON and records outside the canonical contract.

    Parameters
    ----------
    tmp_path : Path
        Temporary test directory.
    content : str
        Invalid input text.
    """
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        INSPECTION.main([str(path)])
    assert error.value.code == 1
    assert path.read_text(encoding="utf-8") == content


def test_recorded_metadata_validates() -> None:
    """Validate both raw responses and confirm their companion crops exist."""
    records = sorted(ASSETS.glob("*.json"))
    assert len(records) == 2
    for path in records:
        assert path.with_suffix(".png").is_file()
        DataSnapshotMetadata.model_validate_json(path.read_text())
