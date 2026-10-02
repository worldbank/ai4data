"""Tests for two robustness fixes found by ingesting a real NADA catalog.

* A study's dataset type comes from NADA (``filters`` / ``core_fields``), not from the record's own
  ``metadata.type`` — ISO geospatial records carry ``type: "dataset"``, which used to be taken as
  the dataset type and rejected as unsupported.
* An unparsable ``date_created`` / ``date_published`` (free-text placeholders such as ``"string"``)
  yields no date instead of making the whole document fail to index.
"""

from __future__ import annotations

import unittest
from unittest import mock

from ai4data.discovery.catalog import extract as catalog_extract
from ai4data.discovery.metadata.filters import DocumentFilterFacets
from ai4data.discovery.metadata.handler import MetadataLoader
from ai4data.discovery.metadata.parsers import DocumentParser


def _study(*, metadata_type=None, filters_type=None, core_type=None, idno="X1") -> dict:
    metadata = {"description": {"idno": idno}}
    if metadata_type is not None:
        metadata["type"] = metadata_type
    study = {"core_fields": {"idno": idno, "id": 1}, "metadata": metadata}
    if filters_type is not None:
        study["filters"] = {"dataset_type": filters_type}
    if core_type is not None:
        study["core_fields"]["dataset_type"] = core_type
    return study


class StudyTypeResolutionTests(unittest.TestCase):
    def test_nada_dataset_type_wins_over_the_records_own_type(self) -> None:
        study = _study(metadata_type="dataset", filters_type="geospatial")

        self.assertEqual(catalog_extract.study_metadata_type(study), "geospatial")
        self.assertEqual(catalog_extract.study_to_search_row(study)["type"], "geospatial")
        self.assertEqual(catalog_extract.study_to_catalog_metadata(study)["type"], "geospatial")

    def test_core_fields_dataset_type_is_used_when_filters_are_absent(self) -> None:
        study = _study(metadata_type="dataset", core_type="geospatial")

        self.assertEqual(catalog_extract.study_to_catalog_metadata(study)["type"], "geospatial")

    def test_dataset_type_is_normalized(self) -> None:
        cases = {"survey": "microdata", "timeseries": "indicator", "timeseriesdb": "indicator-db", "table": "table"}
        for dataset_type, expected in cases.items():
            study = _study(filters_type=dataset_type)
            self.assertEqual(catalog_extract.study_metadata_type(study), expected, dataset_type)
            self.assertEqual(catalog_extract.study_to_catalog_metadata(study)["type"], expected, dataset_type)

    def test_records_own_type_is_only_a_fallback(self) -> None:
        study = _study(metadata_type="timeseries")

        self.assertEqual(catalog_extract.study_metadata_type(study), "indicator")
        self.assertEqual(catalog_extract.study_to_catalog_metadata(study)["type"], "indicator")

    def test_no_type_anywhere(self) -> None:
        self.assertIsNone(catalog_extract.study_metadata_type(_study()))
        self.assertIsNone(catalog_extract.study_to_search_row(_study())["type"])


def _document(**description) -> dict:
    return {
        "type": "document",
        "document_description": {"title_statement": {"idno": "D1", "title": "A title"}, **description},
    }


class DocumentDateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DocumentParser()

    def test_valid_dates_are_formatted(self) -> None:
        record = _document(date_created="2019-05-06", date_published="2020-01-02T10:00:00")

        self.assertEqual(self.parser.parse_date_created(record), "2019-05-06")
        self.assertEqual(self.parser.parse_date_published(record), "2020-01-02")

    def test_unparsable_or_missing_dates_yield_none(self) -> None:
        for bad in ("string", "n.d.", "", "   ", None, [], {}, "NaT", True):
            record = _document(date_created=bad, date_published=bad)

            self.assertIsNone(self.parser.parse_date_created(record), repr(bad))
            self.assertIsNone(self.parser.parse_date_published(record), repr(bad))
            self.assertIsNone(self.parser.parse_periods(record))

    def test_year_only_int_is_read_as_a_year_not_epoch_nanoseconds(self) -> None:
        self.assertTrue(self.parser.parse_date_created(_document(date_created=2019)).startswith("2019"))

    def test_one_bad_date_does_not_lose_the_other(self) -> None:
        record = _document(date_created="2018-03-04", date_published="string")

        self.assertEqual(self.parser.parse_date_created(record), "2018-03-04")
        self.assertIsNone(self.parser.parse_date_published(record))
        self.assertEqual(self.parser.parse_periods(record), "2018")

    def test_facets_build_with_junk_dates(self) -> None:
        facets = DocumentFilterFacets.from_metadata(_document(date_created="string", date_published="string"))

        self.assertEqual(facets.idno, "D1")
        self.assertIsNone(facets.date_created)
        self.assertIsNone(facets.date_published)
        self.assertIsNone(facets.year_start)

    def test_document_with_junk_dates_still_indexes_its_title(self) -> None:
        record = _document(date_created="string", date_published="string", abstract="An abstract.")
        with mock.patch("ai4data.discovery.metadata.handler.get_metadata_json", return_value=dict(record)):
            handler = MetadataLoader(idno="D1", metadata_type="document").get_metadata_handler()

        langdocs = handler.get_langdocs()

        self.assertEqual({d.metadata["qfield"] for d in langdocs}, {"title", "abstract"})


if __name__ == "__main__":
    unittest.main()
