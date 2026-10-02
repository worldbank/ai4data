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
        self.assertEqual(metadata["_extract_core_fields"]["catalog_id"], 42)
        self.assertEqual(metadata["_extract_core_fields"]["idno"], "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1")

    def test_study_to_catalog_metadata_without_core_fields(self):
        study = {key: value for key, value in SAMPLE_STUDY.items() if key != "core_fields"}
        study["idno"] = "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1"
        self.assertNotIn("_extract_core_fields", catalog_extract.study_to_catalog_metadata(study))

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
    def test_iter_extract_studies_with_a_non_positive_max_items_yields_nothing_and_sends_no_request(self, mock_get):
        mock_get.return_value = _FakeResponse(EXTRACT_LIST)

        for max_items in (0, -1):
            self.assertEqual(list(catalog_extract.iter_extract_studies(max_items=max_items)), [])
        mock_get.assert_not_called()

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


class TestVariablesExtract(ExtractModeTestCase):
    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_fetch_extract_variables_page_hits_the_variables_route(self, mock_get):
        mock_get.return_value = _FakeResponse(
            {"status": "success", "total": 5, "has_more": True, "next_after_uid": 7, "variables": [{"a": 1}]}
        )

        data = catalog_extract.fetch_extract_variables_page({"limit": 2})

        self.assertEqual(data["total"], 5)
        self.assertEqual(data["variables"], [{"a": 1}])
        self.assertTrue(mock_get.call_args.args[0].endswith("/variables"))

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_fetch_extract_survey_variables_is_one_page_of_the_study_route(self, mock_get):
        mock_get.return_value = _FakeResponse({"status": "success", "found": 1, "variables": [{"a": 1}]})

        data = catalog_extract.fetch_extract_survey_variables("PSE-PCBS-AGC-2010-V1.0", params={"limit": 50})

        self.assertEqual(data["found"], 1)
        self.assertTrue(mock_get.call_args.args[0].endswith("/variables/PSE-PCBS-AGC-2010-V1.0"))
        self.assertEqual(mock_get.call_args.kwargs["params"], {"limit": 50})

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_iter_extract_variables_follows_the_keyset_cursor(self, mock_get):
        mock_get.side_effect = [
            _FakeResponse({"status": "success", "has_more": True, "next_after_uid": 2, "variables": [{"a": 1}, {"a": 2}]}),
            _FakeResponse({"status": "success", "has_more": False, "next_after_uid": None, "variables": [{"a": 3}]}),
        ]

        rows = list(catalog_extract.iter_extract_variables(page_size=2))

        self.assertEqual([r["a"] for r in rows], [1, 2, 3])
        first, second = (c.kwargs["params"] for c in mock_get.call_args_list)
        self.assertEqual(first, {"limit": 2})  # no cursor on the first page
        self.assertEqual(second, {"limit": 2, "after_uid": 2})

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_iter_extract_survey_variables_walks_every_page_of_one_study(self, mock_get):
        mock_get.side_effect = [
            _FakeResponse({"status": "success", "has_more": True, "next_after_uid": 9, "variables": [{"a": 1}]}),
            _FakeResponse({"status": "success", "has_more": False, "next_after_uid": None, "variables": [{"a": 2}]}),
        ]

        rows = list(catalog_extract.iter_extract_survey_variables("S-1", page_size=1))

        self.assertEqual([r["a"] for r in rows], [1, 2])
        self.assertTrue(all(c.args[0].endswith("/variables/S-1") for c in mock_get.call_args_list))
        self.assertEqual(mock_get.call_args_list[1].kwargs["params"], {"limit": 1, "after_uid": 9})

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_on_page_sees_every_raw_response_including_the_first_pages_total(self, mock_get):
        mock_get.side_effect = [
            _FakeResponse({"status": "success", "total": 3, "has_more": True, "next_after_uid": 2, "variables": [{"a": 1}, {"a": 2}]}),
            _FakeResponse({"status": "success", "total": None, "has_more": False, "variables": [{"a": 3}]}),
        ]
        pages = []

        rows = list(catalog_extract.iter_extract_variables(page_size=2, on_page=pages.append))

        self.assertEqual(len(rows), 3)
        self.assertEqual([p["total"] for p in pages], [3, None])

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_a_page_that_says_has_more_without_a_cursor_stops_instead_of_looping(self, mock_get):
        mock_get.return_value = _FakeResponse({"status": "success", "has_more": True, "variables": [{"a": 1}]})

        rows = list(catalog_extract.iter_extract_variables())

        self.assertEqual(len(rows), 1)
        self.assertEqual(mock_get.call_count, 1)

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_iter_extract_variables_stops_at_an_empty_page(self, mock_get):
        mock_get.return_value = _FakeResponse({"status": "success", "has_more": True, "variables": []})

        self.assertEqual(list(catalog_extract.iter_extract_variables()), [])

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_iter_extract_variables_respects_max_items(self, mock_get):
        mock_get.return_value = _FakeResponse(
            {"status": "success", "has_more": True, "next_after_uid": 3, "variables": [{"a": 1}, {"a": 2}, {"a": 3}]}
        )

        self.assertEqual(len(list(catalog_extract.iter_extract_variables(max_items=2))), 2)

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_a_non_positive_max_items_yields_nothing_and_sends_no_request(self, mock_get):
        mock_get.return_value = _FakeResponse(
            {"status": "success", "has_more": False, "variables": [{"a": 1}], "citations": [{"id": 1}]}
        )

        for max_items in (0, -1):
            self.assertEqual(list(catalog_extract.iter_extract_variables(max_items=max_items)), [])
            self.assertEqual(list(catalog_extract.iter_extract_survey_variables("S1", max_items=max_items)), [])
            self.assertEqual(list(catalog_extract.iter_extract_citations(max_items=max_items)), [])
        mock_get.assert_not_called()

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_access_denied_raises_for_variables_too(self, mock_get):
        mock_get.return_value = _FakeResponse({"status": "ACCESS-DENIED"})

        with self.assertRaises(catalog_extract.CatalogExtractError):
            catalog_extract.fetch_extract_variables_page({"limit": 1})


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


class TestCitationsExtract(ExtractModeTestCase):
    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_fetch_extract_citation_returns_the_citation_document(self, mock_get):
        mock_get.return_value = _FakeResponse({"status": "success", "citation": {"core_fields": {"citation_id": 7}}})

        citation = catalog_extract.fetch_extract_citation(7)

        self.assertEqual(citation, {"core_fields": {"citation_id": 7}})
        self.assertTrue(mock_get.call_args.args[0].endswith("/citations/7"))

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_fetch_extract_citation_without_a_citation_raises(self, mock_get):
        mock_get.return_value = _FakeResponse({"status": "success"})

        with self.assertRaises(catalog_extract.CatalogExtractError):
            catalog_extract.fetch_extract_citation(7)

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_iter_extract_citations_follows_the_keyset_cursor(self, mock_get):
        mock_get.side_effect = [
            _FakeResponse({"status": "success", "has_more": True, "next_after_id": 2, "citations": [{"a": 1}, {"a": 2}]}),
            _FakeResponse({"status": "success", "has_more": False, "next_after_id": None, "citations": [{"a": 3}]}),
        ]

        rows = list(catalog_extract.iter_extract_citations(page_size=2))

        self.assertEqual([r["a"] for r in rows], [1, 2, 3])
        first, second = (c.kwargs["params"] for c in mock_get.call_args_list)
        self.assertEqual(first, {"limit": 2})
        self.assertEqual(second, {"limit": 2, "after_id": 2})
        self.assertTrue(mock_get.call_args.args[0].endswith("/citations"))

    @mock.patch("ai4data.discovery.catalog.extract.httpx.get")
    def test_iter_extract_citations_reports_each_page_and_stops_at_max_items(self, mock_get):
        mock_get.return_value = _FakeResponse(
            {"status": "success", "total": 9, "has_more": True, "next_after_id": 2, "citations": [{"a": 1}, {"a": 2}]}
        )
        pages: list[dict] = []

        rows = list(catalog_extract.iter_extract_citations(page_size=2, max_items=3, on_page=pages.append))

        self.assertEqual(len(rows), 3)
        self.assertEqual([p["total"] for p in pages], [9, 9])
