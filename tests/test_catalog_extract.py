"""Tests for search-metadata-extract catalog backend."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai4data.discovery import paths as discovery_paths
from ai4data.discovery.catalog import extract as catalog_extract
from ai4data.discovery.catalog import http as catalog_http
from ai4data.discovery.config import metadata_catalog

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EXTRACT_LIST = json.loads((FIXTURES / "extract_study_with_metadata.json").read_text())
SAMPLE_STUDY = EXTRACT_LIST["studies"][0]


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class ExtractModeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        discovery_paths.init_discovery_paths(Path(self._tmpdir.name))

        self._extract_patch = mock.patch.object(
            metadata_catalog,
            "extract_path",
            "api/admin/search-metadata-extract",
        )
        self._extract_patch.start()
        self.addCleanup(self._extract_patch.stop)

        self.assertTrue(catalog_extract.is_extract_mode())


class TestStudyNormalization(unittest.TestCase):
    def test_study_to_catalog_metadata_document(self):
        metadata = catalog_extract.study_to_catalog_metadata(SAMPLE_STUDY)
        self.assertEqual(metadata["type"], "document")
        self.assertEqual(metadata["idno"], "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1")
        self.assertIn("_extract_filters", metadata)
        self.assertEqual(metadata["_extract_filters"]["dataset_type"], "document")

    def test_study_to_search_row(self):
        row = catalog_extract.study_to_search_row(SAMPLE_STUDY)
        self.assertEqual(row["idno"], "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1")
        self.assertEqual(row["type"], "document")
        self.assertEqual(row["id"], 42)

    def test_map_catalog_params_to_extract(self):
        mapped = catalog_extract.map_catalog_params_to_extract(
            {"ps": 50, "page": 3, "type": "document", "source": "nada"}
        )
        self.assertEqual(mapped["limit"], 50)
        self.assertEqual(mapped["offset"], 100)
        self.assertEqual(mapped["type"], "document")
        self.assertEqual(mapped["source"], "nada")

    def test_type_alias_timeseries(self):
        study = {
            "core_fields": {"idno": "IND-1"},
            "metadata": {"type": "timeseries", "idno": "IND-1"},
        }
        metadata = catalog_extract.study_to_catalog_metadata(study)
        self.assertEqual(metadata["type"], "indicator")

    def test_study_download_resources_filters_link_type(self):
        metadata = catalog_extract.study_to_catalog_metadata(SAMPLE_STUDY)
        resources = metadata.get("external_resources", [])
        self.assertEqual(len(resources), 1)
        self.assertEqual(resources[0]["resource_id"], "772")
        self.assertEqual(
            resources[0]["url"],
            "https://training.ihsn.org/index.php/api/admin/resources/"
            "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1/resources/download/772",
        )
        self.assertEqual(resources[0]["is_url"], "0")
        self.assertEqual(resources[0]["dcformat"], "application/pdf")

    def test_study_download_resources_empty_when_no_download_type(self):
        study = dict(SAMPLE_STUDY)
        study["resources"] = [
            {
                "resource_id": "1",
                "_links": {"download": "http://example.test/doc.pdf", "type": "link"},
            }
        ]
        metadata = catalog_extract.study_to_catalog_metadata(study)
        self.assertNotIn("external_resources", metadata)


class TestExtractHttp(ExtractModeTestCase):
    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_search_metadata_extract_shape(self, mock_get):
        mock_get.return_value = _FakeResponse(EXTRACT_LIST)

        data = catalog_http.search_metadata({"ps": 1, "page": 1, "type": "document"})

        self.assertEqual(len(data["rows"]), 1)
        self.assertEqual(data["found"], 901)
        self.assertEqual(data["rows"][0]["idno"], "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1")

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_get_metadata_json_extract_writes_cache(self, mock_get):
        single = {"status": "success", "study": SAMPLE_STUDY}
        mock_get.return_value = _FakeResponse(single)

        metadata = catalog_http.get_metadata_json(
            "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1",
            "document",
            force=True,
        )

        self.assertEqual(metadata["type"], "document")
        cache_path = discovery_paths.get_metadata_cache_path(
            "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1",
            "document",
        )
        self.assertTrue(cache_path.exists())

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_get_metadata_ids_cache_metadata(self, mock_get):
        mock_get.return_value = _FakeResponse(EXTRACT_LIST)

        rows = catalog_http.get_metadata_ids(
            {"ps": 100, "type": "document"},
            cache_metadata=True,
        )

        self.assertEqual(len(rows), 1)
        cache_path = discovery_paths.get_metadata_cache_path(
            "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1",
            "document",
        )
        self.assertTrue(cache_path.exists())

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_access_denied_raises(self, mock_get):
        mock_get.return_value = _FakeResponse({"status": "ACCESS-DENIED"})

        with self.assertRaises(catalog_extract.CatalogExtractError):
            catalog_extract.fetch_extract_page({"offset": 0, "limit": 1})


class TestClassicCatalogRegression(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        discovery_paths.init_discovery_paths(Path(self._tmpdir.name))

    def test_extract_mode_disabled_by_default(self):
        with mock.patch.object(metadata_catalog, "extract_path", None):
            self.assertFalse(catalog_extract.is_extract_mode())

    @mock.patch("ai4data.discovery.catalog.http.httpx.get")
    def test_search_metadata_uses_catalog_search(self, mock_get):
        with mock.patch.object(metadata_catalog, "extract_path", None):
            mock_get.return_value = _FakeResponse(
                {"result": {"rows": [{"idno": "X", "type": "document"}], "found": 1}}
            )

            data = catalog_http.search_metadata({"ps": 10, "page": 1})

            self.assertEqual(len(data["rows"]), 1)
            called_url = mock_get.call_args.args[0]
            self.assertIn("/api/catalog/search", called_url)

    @mock.patch("ai4data.discovery.catalog.http.httpx.get")
    def test_get_metadata_json_uses_catalog_json(self, mock_get):
        with mock.patch.object(metadata_catalog, "extract_path", None):
            mock_get.return_value = _FakeResponse({"type": "document", "idno": "DOC-1"})

            metadata = catalog_http.get_metadata_json("DOC-1", "document", force=True)

            self.assertEqual(metadata["idno"], "DOC-1")
            called_url = mock_get.call_args.args[0]
            self.assertIn("/api/catalog/json/DOC-1", called_url)


class TestBatchExtractSinglePass(ExtractModeTestCase):
    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_scrape_all_metadata_single_http_sequence(self, mock_get):
        from ai4data.discovery.catalog import batch as catalog_batch

        page_one = dict(EXTRACT_LIST)
        page_one["has_more"] = True
        page_one["total"] = 2
        page_two = {
            "status": "success",
            "offset": 1,
            "limit": 1,
            "total": 2,
            "has_more": False,
            "studies": [
                {
                    "core_fields": {"idno": "DOC-SECOND"},
                    "filters": {"dataset_type": "document"},
                    "metadata": {
                        "type": "document",
                        "idno": "DOC-SECOND",
                        "metadata_information": {"title": "Second"},
                    },
                }
            ],
        }
        mock_get.side_effect = [
            _FakeResponse(page_one),
            _FakeResponse(page_two),
        ]

        catalog_batch.scrape_all_metadata(type="document", ps=1, force=True)

        self.assertEqual(mock_get.call_count, 2)
        ids_path = discovery_paths.get_metadata_ids_path("document")
        saved = json.loads(ids_path.read_text())
        self.assertEqual(len(saved), 2)

        for idno in ("RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1", "DOC-SECOND"):
            cache_path = discovery_paths.get_metadata_cache_path(idno, "document")
            self.assertTrue(cache_path.exists())
