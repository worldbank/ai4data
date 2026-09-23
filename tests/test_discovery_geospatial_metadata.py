"""Tests for geospatial metadata parsing and filter facets on real-world record shapes.

Covers:

* ``description.identificationInfo`` stored as a one-item list (``gmd:identificationInfo``
  is a repeatable ISO 19139 element, so a properly imported record legitimately has
  this shape) in :class:`GeospatialParser` and the geospatial title/abstract templates.
* Malformed or missing ``identificationInfo`` degrading to empty results, not raising.
* Filter-facet models accepting ``None`` for fields whose parsers return ``None``.
"""

from __future__ import annotations

import unittest

from ai4data.discovery.metadata.filters import (
    DocumentFilterFacets,
    GeospatialFilterFacets,
    IndicatorFilterFacets,
)
from ai4data.discovery.metadata.parsers import GeospatialParser
from ai4data.discovery.metadata.templates.render import render_embedding_content

_IDENTIFICATION_INFO = {
    "citation": {
        "title": "Poverty Map",
        "date": [
            {"date": "2019", "type": "temporal coverage"},
            {"date": "2020", "type": "temporal coverage"},
        ],
        "citedResponsibleParty": [
            {"organisationName": "World Bank"},
            {"organisationName": "NSO"},
        ],
        "identifier": {"code": "10.1/abc", "authority": "DOI"},
    },
    "abstract": "An abstract.",
    "extent": {"geographicElement": [{"geographicDescription": "Senegal"}]},
}

_PARSE_METHODS = ("parse_source", "parse_geographies", "parse_periods", "parse_doi", "parse_abstract")


def _record(identification_info) -> dict:
    return {"description": {"idno": "G1", "identificationInfo": identification_info}}


class GeospatialIdentificationInfoShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = GeospatialParser()

    def test_dict_form_is_parsed(self) -> None:
        record = _record(_IDENTIFICATION_INFO)

        self.assertEqual(self.parser.parse_source(record), ["World Bank", "NSO"])
        self.assertEqual(self.parser.parse_geographies(record), ["Senegal"])
        self.assertEqual(self.parser.parse_periods(record), "2019 - 2020")
        self.assertEqual(self.parser.parse_doi(record), "10.1/abc")
        self.assertEqual(self.parser.parse_abstract(record), "An abstract.")

    def test_one_item_list_form_parses_same_as_dict_form(self) -> None:
        as_dict = _record(_IDENTIFICATION_INFO)
        as_list = _record([_IDENTIFICATION_INFO])

        for method in _PARSE_METHODS:
            with self.subTest(method=method):
                self.assertEqual(getattr(self.parser, method)(as_list), getattr(self.parser, method)(as_dict))

    def test_malformed_identification_info_yields_empty_results(self) -> None:
        empty = _record({})
        for label, bad in {
            "empty list": [],
            "string": "oops",
            "list of non-dicts": ["oops"],
            "none": None,
        }.items():
            for method in _PARSE_METHODS:
                with self.subTest(shape=label, method=method):
                    self.assertEqual(
                        getattr(self.parser, method)(_record(bad)),
                        getattr(self.parser, method)(empty),
                    )

    def test_missing_or_non_dict_description_yields_empty_results(self) -> None:
        for label, record in {
            "empty description": {"description": {}},
            "no description": {},
            "non-dict description": {"description": "oops"},
        }.items():
            for method in _PARSE_METHODS:
                with self.subTest(shape=label, method=method):
                    self.assertIsNone(getattr(self.parser, method)(record))


class GeospatialTemplateShapeTests(unittest.TestCase):
    def render(self, record: dict, field: str) -> str:
        return render_embedding_content(record, "geospatial", field)

    def test_dict_and_list_forms_render_title_and_abstract(self) -> None:
        for label, record in {
            "dict": _record(_IDENTIFICATION_INFO),
            "list": _record([_IDENTIFICATION_INFO]),
        }.items():
            with self.subTest(shape=label):
                self.assertEqual(self.render(record, "title"), "Poverty Map")
                self.assertEqual(self.render(record, "abstract"), "An abstract.")

    def test_missing_or_malformed_identification_info_renders_empty(self) -> None:
        for label, record in {
            "empty list": _record([]),
            "string": _record("oops"),
            "list of non-dicts": _record(["oops"]),
            "missing key": {"description": {"idno": "G1"}},
            "no description": {"idno": "G1"},
            "no citation": _record({"abstract": "only an abstract"}),
        }.items():
            with self.subTest(shape=label):
                self.assertEqual(self.render(record, "title"), "")

    def test_abstract_without_citation_still_renders(self) -> None:
        record = _record([{"abstract": "only an abstract"}])
        self.assertEqual(self.render(record, "abstract"), "only an abstract")
        self.assertEqual(self.render(record, "title"), "")


class GeospatialFilterFacetsTests(unittest.TestCase):
    def test_list_form_record_builds_facets(self) -> None:
        facets = GeospatialFilterFacets.from_metadata(_record([_IDENTIFICATION_INFO]))

        self.assertEqual(facets.idno, "G1")
        self.assertEqual(facets.source, ["World Bank", "NSO"])
        self.assertEqual(facets.geographies, ["Senegal"])
        self.assertEqual((facets.year_start, facets.year_end), (2019, 2020))

    def test_record_without_authors_builds_facets_with_no_source(self) -> None:
        facets = GeospatialFilterFacets.from_metadata(_record({"citation": {"title": "T"}}))

        self.assertIsNone(facets.source)


class FilterFacetsNoneDefaultTests(unittest.TestCase):
    """Facet fields default to ``None`` and must accept an explicit ``None``.

    Before the ``| None`` annotations, building a facet from a record whose parser
    returned ``None`` for one of these fields raised a pydantic ``ValidationError``.
    """

    def test_sparse_document_builds_facets(self) -> None:
        facets = DocumentFilterFacets.from_metadata(
            {"idno": "D1", "document_description": {"title_statement": {"idno": "D1", "title": "T"}}}
        )

        self.assertEqual(facets.idno, "D1")
        self.assertIsNone(facets.document_type)
        self.assertIsNone(facets.date_published)

    def test_sparse_indicator_builds_facets(self) -> None:
        facets = IndicatorFilterFacets.from_metadata(
            {"idno": "I1", "series_description": {"idno": "I1", "name": "N"}}
        )

        self.assertEqual(facets.idno, "I1")
        self.assertIsNone(facets.periodicity)

    def test_none_is_accepted_for_every_optional_facet_field(self) -> None:
        DocumentFilterFacets(idno="D", document_type=None, date_published=None, authors=None)
        IndicatorFilterFacets(idno="I", periodicity=None, source=None)
        GeospatialFilterFacets(idno="G", source=None)


if __name__ == "__main__":
    unittest.main()
