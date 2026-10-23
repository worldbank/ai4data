"""Test the canonical Data Snapshot Metadata Schema v1.4 port."""

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

import ai4data.data_snapshot.metadata_schema as metadata_models
from ai4data.data_snapshot import DataSnapshotMetadata


def test_minimal_record_is_empty_and_forbids_unknown_fields() -> None:
    """Accept an empty record while rejecting fields outside Schema v1.4."""
    assert DataSnapshotMetadata().model_dump(exclude_none=True) == {}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DataSnapshotMetadata.model_validate({"source_document_title": "Removed"})


def test_every_public_field_example_validates() -> None:
    """Validate every published example against its canonical field type."""
    for name in metadata_models.__all__:
        model = getattr(metadata_models, name)
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            continue
        for field_name, field in model.model_fields.items():
            assert field.examples, f"{name}.{field_name} has no examples"
            adapter = TypeAdapter(field.rebuild_annotation())
            for example in field.examples:
                adapter.validate_python(example)
                if model is DataSnapshotMetadata:
                    model.model_validate({field_name: example})


def test_export_identifies_schema_v1_4() -> None:
    """Preserve the canonical schema title, dialect, and version marker."""
    schema = DataSnapshotMetadata.model_json_schema()

    assert schema["title"] == "Data Snapshot Metadata Schema v1.4"
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["x-schema-version"] == "1.4"
