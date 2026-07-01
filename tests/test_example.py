import unittest
import unittest.mock
from unittest.mock import MagicMock


def test_example_extraction_import(monkeypatch):
    """Test that example_extraction module can be imported (smoke test)."""

    # Mock gliner2 module
    mock_gliner_class = MagicMock()
    mock_model = MagicMock()

    mock_schema = MagicMock()
    mock_schema.structure.return_value = mock_schema
    mock_schema.field.return_value = mock_schema
    mock_model.create_schema.return_value = mock_schema
    mock_model.load_adapter = MagicMock()

    mock_gliner_class.from_pretrained.return_value = mock_model

    mock_gliner_module = MagicMock()
    mock_gliner_module.GLiNER2 = mock_gliner_class

    # Mock huggingface_hub.snapshot_download so no network calls are made
    mock_hf_module = MagicMock()
    mock_hf_module.snapshot_download.return_value = "/tmp/fake_adapter"

    with unittest.mock.patch.dict(
        "sys.modules",
        {"gliner2": mock_gliner_module, "huggingface_hub": mock_hf_module},
    ):
        import importlib

        import ai4data.data_use.example_extraction as example_extraction

        importlib.reload(example_extraction)

        # Verify the base model was loaded and the adapter was applied
        assert example_extraction.model is mock_model
        mock_model.load_adapter.assert_called()
