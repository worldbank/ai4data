"""Check Data Snapshot cookbook utilities and recorded artifacts."""

import importlib.util
import json
import shutil
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
COUNCIL = ASSETS.parent / "data-snapshot-metadata-council"
REGISTER_SPEC = importlib.util.spec_from_file_location(
    "snapshot_registration", COUNCIL / "register_samples.py"
)
assert REGISTER_SPEC is not None and REGISTER_SPEC.loader is not None
REGISTRATION = importlib.util.module_from_spec(REGISTER_SPEC)
REGISTER_SPEC.loader.exec_module(REGISTRATION)


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


@pytest.fixture
def council_batch(tmp_path: Path) -> Path:
    """Copy the four real inputs into the documented registration layout.

    Parameters
    ----------
    tmp_path : Path
        Temporary experiment directory.

    Returns
    -------
    Path
        Experiment directory containing the batch and source CSV.
    """
    register = json.loads((COUNCIL / "samples.json").read_text())
    static_root = COUNCIL.parents[1]
    for sample in register["samples"]:
        folder = tmp_path / "batch" / sample["split"] / sample["sample_id"]
        folder.mkdir(parents=True)
        shutil.copyfile(static_root / sample["image"], folder / "snapshot.png")
        if sample["gold_reference"]:
            shutil.copyfile(
                static_root / sample["gold_reference"], folder / "metadata.json"
            )
    shutil.copyfile(COUNCIL / "batch_sources.csv", tmp_path / "batch_sources.csv")
    return tmp_path


def test_council_registration_is_repeatable(council_batch: Path) -> None:
    """Preserve the real sample associations and reuse an identical register.

    Parameters
    ----------
    council_batch : Path
        Prepared experiment directory.
    """
    output = council_batch / "samples.json"
    arguments = [
        str(council_batch / "batch"),
        "--sources",
        str(council_batch / "batch_sources.csv"),
        "--output",
        str(output),
    ]
    assert REGISTRATION.main(arguments) == 0
    original = output.read_bytes()
    assert REGISTRATION.main(arguments) == 0
    assert output.read_bytes() == original
    register = json.loads(original)
    expected = json.loads((COUNCIL / "samples.json").read_text())
    for sample, reference in zip(register["samples"], expected["samples"], strict=True):
        for key in ("sample_id", "split", "image_sha256", "gold_sha256", "pdf_page"):
            assert sample[key] == reference[key]
        assert (output.parent / sample["image"]).is_file()
    (council_batch / "batch/discovery/D1/snapshot.png").write_bytes(b"changed")
    assert REGISTRATION.main(arguments) == 1
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "problem",
    [
        "missing_reference",
        "invalid_json",
        "challenge_gold",
        "missing_source",
        "multiple_images",
        "duplicate_id",
    ],
)
def test_council_registration_rejects_invalid_batch(
    council_batch: Path, problem: str
) -> None:
    """Reject broken image/reference associations before creating output.

    Parameters
    ----------
    council_batch : Path
        Prepared experiment directory.
    problem : str
        Invalid batch condition to introduce.
    """
    reference = council_batch / "batch/discovery/D1/metadata.json"
    if problem == "missing_reference":
        reference.unlink()
    elif problem == "invalid_json":
        reference.write_text("[]")
    elif problem == "challenge_gold":
        (council_batch / "batch/challenge/C1/metadata.json").write_text("{}")
    elif problem == "missing_source":
        sources = council_batch / "batch_sources.csv"
        sources.write_text(sources.read_text().replace("D1,", "MISSING,", 1))
    elif problem == "multiple_images":
        shutil.copyfile(
            council_batch / "batch/discovery/D1/snapshot.png",
            council_batch / "batch/discovery/D1/another.png",
        )
    else:
        shutil.copytree(
            council_batch / "batch/challenge/C1", council_batch / "batch/challenge/D1"
        )
    output = council_batch / "samples.json"
    assert (
        REGISTRATION.main(
            [
                str(council_batch / "batch"),
                "--sources",
                str(council_batch / "batch_sources.csv"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()
