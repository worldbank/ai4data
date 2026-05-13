"""Tests for :mod:`ai4data.discovery.auth` and the auth-header wiring across catalog/HTTP
and document download paths.

Covers ``AI4DATA_METADATA_CATALOG_X_API_KEY`` (and host allow-list) being attached to:
- :func:`ai4data.discovery.catalog.http.get_metadata_json`
- :func:`ai4data.discovery.catalog.http.search_metadata`
- :func:`ai4data.discovery.metadata.document_fetch.download_pdf`

The header is host-scoped: arbitrary URLs only receive the key when the host matches the
configured catalog host (or is allow-listed via ``x_api_key_hosts``).
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai4data.discovery import auth as discovery_auth
from ai4data.discovery import paths as discovery_paths
from ai4data.discovery.catalog import http as catalog_http
from ai4data.discovery.config import metadata_catalog
from ai4data.discovery.metadata import document_fetch


@contextlib.contextmanager
def _override_catalog(*, url: str | None = None, x_api_key: str | None = None,
                     x_api_key_hosts: str | None = None):
    """Temporarily mutate ``metadata_catalog`` for a test, restoring previous values."""
    saved = (metadata_catalog.url, metadata_catalog.x_api_key, metadata_catalog.x_api_key_hosts)
    if url is not None:
        metadata_catalog.url = url
    metadata_catalog.x_api_key = x_api_key
    metadata_catalog.x_api_key_hosts = x_api_key_hosts
    try:
        yield
    finally:
        metadata_catalog.url, metadata_catalog.x_api_key, metadata_catalog.x_api_key_hosts = saved


class _FakeResponse:
    def __init__(self, content: bytes = b"", payload: dict | None = None,
                 status_code: int = 200, headers: dict | None = None):
        self.content = content
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload or {}


_MIN_PDF = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


class TestGetCatalogAuthHeaders(unittest.TestCase):
    def test_no_key_returns_empty(self) -> None:
        with _override_catalog(x_api_key=None):
            self.assertEqual(discovery_auth.get_catalog_auth_headers(), {})
            self.assertEqual(
                discovery_auth.get_catalog_auth_headers("https://anywhere.example/x.pdf"), {}
            )

    def test_key_without_url_is_sent_unconditionally(self) -> None:
        # For catalog-only endpoints (search, JSON metadata) we always send the header.
        with _override_catalog(x_api_key="secret"):
            self.assertEqual(
                discovery_auth.get_catalog_auth_headers(), {"x-api-key": "secret"}
            )

    def test_key_sent_for_matching_catalog_host(self) -> None:
        with _override_catalog(
            url="https://catalog.example/index.php", x_api_key="secret"
        ):
            self.assertEqual(
                discovery_auth.get_catalog_auth_headers(
                    "https://catalog.example/index.php/catalog/1/download/2/file.pdf"
                ),
                {"x-api-key": "secret"},
            )

    def test_key_withheld_from_other_hosts(self) -> None:
        with _override_catalog(
            url="https://catalog.example/index.php", x_api_key="secret"
        ):
            self.assertEqual(
                discovery_auth.get_catalog_auth_headers(
                    "https://elsewhere.example/file.pdf"
                ),
                {},
            )

    def test_key_sent_for_allow_listed_host(self) -> None:
        with _override_catalog(
            url="https://catalog.example/index.php",
            x_api_key="secret",
            x_api_key_hosts="training.ihsn.org, files.ihsn.org",
        ):
            self.assertEqual(
                discovery_auth.get_catalog_auth_headers(
                    "https://training.ihsn.org/index.php/catalog/1/download/2/file.pdf"
                ),
                {"x-api-key": "secret"},
            )
            self.assertEqual(
                discovery_auth.get_catalog_auth_headers(
                    "https://files.ihsn.org/x.pdf"
                ),
                {"x-api-key": "secret"},
            )

    def test_relative_url_treated_as_catalog_local(self) -> None:
        with _override_catalog(x_api_key="secret"):
            self.assertEqual(
                discovery_auth.get_catalog_auth_headers("/api/catalog/json/IDN-X"),
                {"x-api-key": "secret"},
            )


class _DiscoveryPathsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        discovery_paths.init_discovery_paths(Path(self._tmpdir.name))
        self.addCleanup(discovery_paths.init_discovery_paths, None)


class TestHttpAttachesAuthHeader(_DiscoveryPathsTestCase):
    def test_get_metadata_json_attaches_x_api_key_when_configured(self) -> None:
        fake = _FakeResponse(payload={"type": "indicator"})
        with _override_catalog(x_api_key="secret"):
            with mock.patch.object(
                catalog_http.httpx, "get", return_value=fake
            ) as mocked_get:
                catalog_http.get_metadata_json("IDN-AUTH", "indicator")
        _, kwargs = mocked_get.call_args
        self.assertEqual(kwargs.get("headers"), {"x-api-key": "secret"})

    def test_get_metadata_json_omits_header_when_no_key(self) -> None:
        fake = _FakeResponse(payload={"type": "indicator"})
        with _override_catalog(x_api_key=None):
            with mock.patch.object(
                catalog_http.httpx, "get", return_value=fake
            ) as mocked_get:
                catalog_http.get_metadata_json("IDN-NOKEY", "indicator")
        _, kwargs = mocked_get.call_args
        self.assertEqual(kwargs.get("headers"), {})

    def test_search_metadata_attaches_x_api_key_when_configured(self) -> None:
        fake = _FakeResponse(payload={"result": {"rows": [], "found": 0}})
        with _override_catalog(x_api_key="secret"):
            with mock.patch.object(
                catalog_http.httpx, "get", return_value=fake
            ) as mocked_get:
                catalog_http.search_metadata({"sk": "test"})
        _, kwargs = mocked_get.call_args
        self.assertEqual(kwargs.get("headers"), {"x-api-key": "secret"})


class TestDownloadPdfAttachesAuthHeader(_DiscoveryPathsTestCase):
    def test_download_pdf_attaches_x_api_key_to_catalog_host(self) -> None:
        with _override_catalog(
            url="https://catalog.example/index.php", x_api_key="secret"
        ):
            with mock.patch.object(
                document_fetch.httpx, "get", return_value=_FakeResponse(content=_MIN_PDF)
            ) as mocked_get:
                document_fetch.download_pdf(
                    "https://catalog.example/index.php/catalog/1/download/2/file.pdf"
                )
        _, kwargs = mocked_get.call_args
        self.assertEqual(kwargs.get("headers"), {"x-api-key": "secret"})

    def test_download_pdf_withholds_x_api_key_from_third_party_host(self) -> None:
        with _override_catalog(
            url="https://catalog.example/index.php", x_api_key="secret"
        ):
            with mock.patch.object(
                document_fetch.httpx, "get", return_value=_FakeResponse(content=_MIN_PDF)
            ) as mocked_get:
                document_fetch.download_pdf(
                    "https://third-party.example/file.pdf"
                )
        _, kwargs = mocked_get.call_args
        self.assertEqual(kwargs.get("headers"), {})

    def test_download_pdf_uses_allow_listed_host(self) -> None:
        with _override_catalog(
            url="https://catalog.example/index.php",
            x_api_key="secret",
            x_api_key_hosts="training.ihsn.org",
        ):
            with mock.patch.object(
                document_fetch.httpx, "get", return_value=_FakeResponse(content=_MIN_PDF)
            ) as mocked_get:
                document_fetch.download_pdf(
                    "https://training.ihsn.org/index.php/catalog/529/download/1293/file.pdf"
                )
        _, kwargs = mocked_get.call_args
        self.assertEqual(kwargs.get("headers"), {"x-api-key": "secret"})


if __name__ == "__main__":
    unittest.main()
