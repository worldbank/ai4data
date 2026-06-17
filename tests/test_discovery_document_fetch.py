"""Tests for :mod:`ai4data.discovery.metadata.document_fetch`.

Covers the IHSN / NADA failure mode where the catalog returns ``200 OK`` with a 0-byte body
and a ``Refresh: 0;url=.../auth/login/...`` header for documents that require authentication.
Without validation, ``cache_download_pdf`` would persist that empty body and short-circuit on
every subsequent call (``fpath.exists()`` is True), so ``load_pdf`` always failed with
"Cannot open empty file" and the document was silently skipped.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai4data.discovery import paths as discovery_paths
from ai4data.discovery.metadata import document_fetch


class _FakeResponse:
    """Minimal stand-in for :class:`httpx.Response`."""

    def __init__(self, content: bytes, status_code: int = 200, headers: dict | None = None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None


_MIN_PDF = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


class _DocumentFetchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_root = Path(self._tmpdir.name)
        discovery_paths.init_discovery_paths(self.tmp_root)
        self.addCleanup(discovery_paths.init_discovery_paths, None)


class TestDownloadPdf(_DocumentFetchTestCase):
    def test_valid_pdf_returns_bytes(self) -> None:
        with mock.patch.object(
            document_fetch.httpx, "get", return_value=_FakeResponse(_MIN_PDF)
        ):
            content = document_fetch.download_pdf("https://example.test/doc.pdf")
        self.assertEqual(content, _MIN_PDF)

    def test_empty_body_raises_empty_download_error(self) -> None:
        with mock.patch.object(
            document_fetch.httpx, "get", return_value=_FakeResponse(b"")
        ):
            with self.assertRaises(document_fetch.EmptyDownloadError):
                document_fetch.download_pdf("https://example.test/doc.pdf")

    def test_empty_body_with_login_refresh_header_surfaces_hint(self) -> None:
        # Reproduces the IHSN / NADA login-wall response observed in production:
        # 200 OK, content-length 0, Refresh: 0;url=.../auth/login/...
        refresh = (
            "0;url=https://training.ihsn.org/index.php/auth/login/?destination=..."
        )
        with mock.patch.object(
            document_fetch.httpx,
            "get",
            return_value=_FakeResponse(b"", headers={"refresh": refresh}),
        ):
            with self.assertRaises(document_fetch.EmptyDownloadError) as ctx:
                document_fetch.download_pdf("https://example.test/doc.pdf")
        self.assertIn("login", str(ctx.exception).lower())

    def test_non_pdf_body_raises_empty_download_error(self) -> None:
        with mock.patch.object(
            document_fetch.httpx,
            "get",
            return_value=_FakeResponse(b"<html>Login required</html>"),
        ):
            with self.assertRaises(document_fetch.EmptyDownloadError):
                document_fetch.download_pdf("https://example.test/doc.pdf")


class TestCacheDownloadPdf(_DocumentFetchTestCase):
    def test_writes_pdf_and_returns_path(self) -> None:
        with mock.patch.object(
            document_fetch.httpx, "get", return_value=_FakeResponse(_MIN_PDF)
        ) as mocked_get:
            fpath = document_fetch.cache_download_pdf(
                "https://example.test/doc.pdf", "IDN-OK", "document"
            )
        self.assertTrue(fpath.exists())
        self.assertGreater(fpath.stat().st_size, 0)
        self.assertEqual(fpath.read_bytes(), _MIN_PDF)
        mocked_get.assert_called_once()

    def test_existing_zero_byte_cache_is_discarded_and_redownloaded(self) -> None:
        # Simulate the legacy bug: an earlier run cached an empty body. The next call should
        # NOT short-circuit on `fpath.exists()`; it should drop the stale file and re-fetch.
        fpath = discovery_paths.get_document_cache_path("IDN-EMPTY", "document")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(b"")
        self.assertEqual(fpath.stat().st_size, 0)

        with mock.patch.object(
            document_fetch.httpx, "get", return_value=_FakeResponse(_MIN_PDF)
        ) as mocked_get:
            result = document_fetch.cache_download_pdf(
                "https://example.test/doc.pdf", "IDN-EMPTY", "document"
            )

        self.assertEqual(result, fpath)
        self.assertEqual(fpath.read_bytes(), _MIN_PDF)
        mocked_get.assert_called_once()

    def test_existing_non_empty_cache_is_reused(self) -> None:
        fpath = discovery_paths.get_document_cache_path("IDN-CACHED", "document")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(_MIN_PDF)

        with mock.patch.object(document_fetch.httpx, "get") as mocked_get:
            result = document_fetch.cache_download_pdf(
                "https://example.test/doc.pdf", "IDN-CACHED", "document"
            )

        self.assertEqual(result, fpath)
        mocked_get.assert_not_called()

    def test_empty_response_does_not_persist_zero_byte_file(self) -> None:
        # The previous implementation happily wrote `b""` to disk on a 200/empty response,
        # poisoning the cache. The hardened version must raise before touching the file.
        with mock.patch.object(
            document_fetch.httpx, "get", return_value=_FakeResponse(b"")
        ):
            with self.assertRaises(ValueError):
                document_fetch.cache_download_pdf(
                    "https://example.test/doc.pdf", "IDN-AUTH", "document"
                )
        fpath = discovery_paths.get_document_cache_path("IDN-AUTH", "document")
        self.assertFalse(fpath.exists())

    def test_resource_id_uses_double_hyphen_filename(self) -> None:
        idno = "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1"
        resource_id = "772"
        expected = discovery_paths.get_document_cache_path(
            idno, "document", resource_id=resource_id
        )
        self.assertEqual(
            expected.name,
            f"document_{idno}--{resource_id}.pdf",
        )

        with mock.patch.object(
            document_fetch.httpx, "get", return_value=_FakeResponse(_MIN_PDF)
        ):
            fpath = document_fetch.cache_download_pdf(
                "https://example.test/doc.pdf",
                idno,
                "document",
                resource_id=resource_id,
            )

        self.assertEqual(fpath, expected)
        self.assertTrue(fpath.exists())


if __name__ == "__main__":
    unittest.main()
