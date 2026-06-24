"""Tests for utility functions."""

import os

import pytest

from ai4data.data_use.utils.document_parser import DocumentParser


class TestDocumentParser:
    """Test suite for DocumentParser class."""

    def test_load_pdf_chunks_file_not_found(self):
        """Test that non-existent file raises appropriate error."""
        # fitz.open raises FileNotFound/RuntimeError for bad paths
        # We'll just verify it propagates the error from fitz
        with pytest.raises(Exception):
            DocumentParser.load_pdf_chunks("nonexistent_file.pdf")

    def test_load_pdf_chunks_basic(self, monkeypatch):
        """Test loading PDF chunks from a file path."""
        mock_doc = MockPDFDocument(["Page 1 text", "Page 2 text"])

        def mock_open(path):
            return mock_doc

        import fitz

        monkeypatch.setattr(fitz, "open", mock_open)

        # We must provide a dummy path
        chunks = DocumentParser.load_pdf_chunks("dummy.pdf", n_pages=1, as_markdown=False)

        assert len(chunks) == 2
        assert chunks[0]["text"] == "Page 1 text"
        assert chunks[0]["pages"] == [0]
        assert chunks[1]["text"] == "Page 2 text"
        assert chunks[1]["pages"] == [1]

    def test_load_pdf_chunks_pagination(self, monkeypatch):
        """Test chunking with n_pages > 1."""
        # 10 pages total
        mock_doc = MockPDFDocument([f"Page {i}" for i in range(10)])

        def mock_open(path):
            return mock_doc

        import fitz

        monkeypatch.setattr(fitz, "open", mock_open)

        # distinct chunks of 3 pages
        chunks = DocumentParser.load_pdf_chunks("dummy.pdf", n_pages=3, as_markdown=False)

        # Should be 4 chunks: [0,1,2], [3,4,5], [6,7,8], [9]
        assert len(chunks) == 4
        assert chunks[0]["pages"] == [0, 1, 2]
        assert "Page 0" in chunks[0]["text"]
        assert "Page 2" in chunks[0]["text"]

        assert chunks[3]["pages"] == [9]
        assert "Page 9" in chunks[3]["text"]

    def test_load_pdf_chunks_page_filtering(self, monkeypatch):
        """Test that chunks can be filtered using the pages parameter."""
        # 10 pages total
        mock_doc = MockPDFDocument([f"Page {i}" for i in range(10)])

        def mock_open(path):
            return mock_doc

        import fitz

        monkeypatch.setattr(fitz, "open", mock_open)

        # Process only pages 2, 4, and 5
        chunks = DocumentParser.load_pdf_chunks(
            "dummy.pdf", n_pages=1, as_markdown=False, pages=[2, 4, 5]
        )

        assert len(chunks) == 3
        assert chunks[0]["pages"] == [2]
        assert chunks[0]["text"] == "Page 2"
        assert chunks[1]["pages"] == [4]
        assert chunks[1]["text"] == "Page 4"
        assert chunks[2]["pages"] == [5]
        assert chunks[2]["text"] == "Page 5"

    def test_load_pdf_chunks_url(self, monkeypatch):
        """Test loading PDF chunks from a URL."""
        mock_doc = MockPDFDocument(["URL content"])

        def mock_open(path):
            return mock_doc

        # Mock requests responses
        class MockResponse:
            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                yield b"fake pdf content"

        class MockHeadResponse:
            headers = {"Content-Length": "1024"}  # 1KB

        def mock_get(url, **kwargs):
            return MockResponse()

        def mock_head(url, **kwargs):
            return MockHeadResponse()

        import fitz
        import requests

        # Mock os.unlink to avoid error when deleting non-existent temp file
        monkeypatch.setattr(os, "unlink", lambda x: None)
        monkeypatch.setattr(fitz, "open", mock_open)
        monkeypatch.setattr(requests, "get", mock_get)
        monkeypatch.setattr(requests, "head", mock_head)

        chunks = DocumentParser.load_pdf_chunks("https://example.com/test.pdf", as_markdown=False)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "URL content"
        assert chunks[0]["pages"] == [0]

    def test_load_json_data_json(self, tmp_path):
        """Test loading standard JSON file."""
        import json

        data = [{"text": "Sample text", "id": 1}, {"text": "Another text", "id": 2}]

        json_file = tmp_path / "test.json"
        with open(json_file, "w") as f:
            json.dump(data, f)

        loaded = DocumentParser.load_json_data(str(json_file))
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert loaded[0]["id"] == 1
        assert loaded[1]["text"] == "Another text"

    def test_load_json_data_jsonl(self, tmp_path):
        """Test loading JSONL (newline-delimited) file."""
        import json

        data = [{"text": "Line 1", "id": 1}, {"text": "Line 2", "id": 2}]

        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        loaded = DocumentParser.load_json_data(str(jsonl_file))
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert loaded[0]["id"] == 1
        assert loaded[1]["text"] == "Line 2"


class MockPDFDocument:
    """Mock PDF document for testing."""

    def __init__(self, pages_text):
        self.pages_text = pages_text
        self.closed = False

    def __len__(self):
        return len(self.pages_text)

    def __getitem__(self, index):
        return MockPage(self.pages_text[index])

    def close(self):
        self.closed = True


class MockPage:
    """Mock PDF page for testing."""

    def __init__(self, text):
        self.text = text

    def get_text(self):
        return self.text
