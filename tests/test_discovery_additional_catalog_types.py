"""Tests for the table, script, image, video and indicator-db (NADA ``timeseriesdb``) metadata types.

Covers, per type:

* the embedding templates (title / sub_title / abstract) on schema-shaped records and on
  missing or malformed shapes, which must render empty rather than raise;
* the parsers behind the year / geography / source filter facets;
* ``get_idno`` and the ``MetadataLoader`` handler dispatch end to end.

Plus the NADA-API <-> normalized type-name mapping (``timeseriesdb`` <-> ``indicator-db``).
"""

from __future__ import annotations

import unittest
from unittest import mock

from ai4data.discovery.catalog.batch import _normalize_scrape_params
from ai4data.discovery.catalog.http import get_ids_type
from ai4data.discovery.metadata.filters import (
    FilterFacets,
    ImageFilterFacets,
    IndicatorDbFilterFacets,
    ScriptFilterFacets,
    TableFilterFacets,
    VideoFilterFacets,
    get_filter_facets,
)
from ai4data.discovery.metadata.handler import MetadataLoader
from ai4data.discovery.metadata.parsers import (
    ImageParser,
    IndicatorDbParser,
    ScriptParser,
    TableParser,
    VideoParser,
)
from ai4data.discovery.metadata.templates.render import render_embedding_content
from ai4data.discovery.metadata.utils import get_idno
from ai4data.discovery.type_normalization import normalize_catalog_metadata_type, to_catalog_api_type

_INDICATOR_DB = {
    "type": "indicator-db",
    "database_description": {
        "title_statement": {"idno": "DB1", "title": "World Development DB", "sub_title": "Annual series"},
        "abstract": "A database of development indicators.",
        "authoring_entity": [{"name": "World Bank"}, {"name": "IMF"}, {"name": "World Bank"}],
        "ref_country": [{"name": "Kenya", "code": "KEN"}, {"name": "Chad", "code": "TCD"}],
        "time_coverage": [{"start": "1990", "end": 2020}],
    },
}

_TABLE = {
    "type": "table",
    "table_description": {
        "title_statement": {"idno": "T1", "title": "Labour force by region", "sub_title": "2015 census"},
        "description": "Employment counts.",
        "notes": [{"note": "Note one."}, {"note": "Note two."}],
        "authoring_entity": [{"name": "NSO"}],
        "publisher": [{"name": "Ignored Publisher"}],
        "ref_country": [{"name": "Ghana"}],
        "time_periods": [{"from": "2010", "to": "2015"}],
        "date_published": "2018-06-01",
    },
}

_SCRIPT = {
    "type": "script",
    "project_desc": {
        "title_statement": {"idno": "S1", "title": "Poverty replication", "sub_title": "Stata do-files"},
        "abstract": "Reproduces the poverty estimates.",
        "authoring_entity": [{"name": "Analyst A"}, {"name": "Analyst B"}],
        "geographic_units": [{"name": "Peru"}],
        "production_date": "2019-05",
    },
}

_IMAGE_DCMI = {
    "type": "image",
    "image_description": {
        "idno": "I1",
        "dcmi": {
            "title": "Market day",
            "description": "People at a market.",
            "caption": "Ignored caption",
            "creator": "A. Photographer",
            "country": [{"name": "Mali"}],
            "date": "2012-03-04",
        },
    },
}

_IMAGE_IPTC = {
    "type": "image",
    "image_description": {
        "idno": "I2",
        "iptc": {
            "photoVideoMetadataIPTC": {
                "headline": "Flood waters",
                "description": "A flooded street.",
                "creatorNames": ["B. Photographer", "B. Photographer"],
                "locationsShown": [{"name": "Main Street", "countryName": "Bangladesh"}],
                "dateCreated": "2020-07-01T10:00:00Z",
            }
        },
    },
}

_VIDEO = {
    "type": "video",
    "video_description": {
        "idno": "V1",
        "title": "Survey fieldwork",
        "description": "Interviewers at work.",
        "creator": "Field Team",
        "country": [{"name": "Nepal"}],
        "date_published": "2021-02-03",
        "date_created": "2020-01-01",
    },
}


def _render(record: dict, metadata_type: str, field: str) -> str:
    return render_embedding_content(record, metadata_type, field).strip()


class TypeNameMappingTests(unittest.TestCase):
    def test_timeseriesdb_spellings_normalize_to_indicator_db(self) -> None:
        self.assertEqual(normalize_catalog_metadata_type("timeseriesdb"), "indicator-db")
        self.assertEqual(normalize_catalog_metadata_type("timeseries-db"), "indicator-db")

    def test_existing_aliases_are_unchanged(self) -> None:
        self.assertEqual(normalize_catalog_metadata_type("timeseries"), "indicator")
        self.assertEqual(normalize_catalog_metadata_type("survey"), "microdata")
        for same in ("document", "geospatial", "table", "script", "image", "video"):
            self.assertEqual(normalize_catalog_metadata_type(same), same)

    def test_api_type_round_trips(self) -> None:
        self.assertEqual(to_catalog_api_type("indicator-db"), "timeseriesdb")
        self.assertEqual(to_catalog_api_type("indicator"), "timeseries")
        self.assertEqual(to_catalog_api_type("microdata"), "survey")
        self.assertEqual(to_catalog_api_type("table"), "table")

    def test_catalog_row_type_is_normalized(self) -> None:
        row = get_ids_type({"id": 7, "idno": "DB1", "type": "timeseriesdb"})
        self.assertEqual(row, {"id": 7, "idno": "DB1", "type": "indicator-db"})

    def test_scrape_params_query_nada_with_its_own_type_name(self) -> None:
        params, dtype = _normalize_scrape_params({"ps": 10, "type": "indicator-db"})
        self.assertEqual((params["type"], dtype), ("timeseriesdb", "indicator-db"))

        params, dtype = _normalize_scrape_params({"ps": 10, "type": "indicator"})
        self.assertEqual((params["type"], dtype), ("timeseries", "indicator"))


class IdnoTests(unittest.TestCase):
    def test_idno_per_type(self) -> None:
        self.assertEqual(get_idno(_INDICATOR_DB, "indicator-db"), "DB1")
        self.assertEqual(get_idno(_TABLE, "table"), "T1")
        self.assertEqual(get_idno(_SCRIPT, "script"), "S1")
        self.assertEqual(get_idno(_IMAGE_DCMI, "image"), "I1")
        self.assertEqual(get_idno(_VIDEO, "video"), "V1")

    def test_unknown_type_still_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_idno({}, "citation")

    def test_falls_back_to_the_catalog_idno_when_the_schema_idno_is_absent(self) -> None:
        # image/video schemas make ``idno`` optional; real NADA image records often omit it.
        image = {"idno": "GEMS_1", "image_description": {"iptc": {}}}
        video = {"idno": "VID_1", "video_description": {"title": "No idno here"}}

        self.assertEqual(get_idno(image, "image"), "GEMS_1")
        self.assertEqual(get_idno(video, "video"), "VID_1")
        self.assertEqual(get_filter_facets({"type": "image", **image}).idno, "GEMS_1")

    def test_schema_idno_wins_over_the_top_level_one(self) -> None:
        self.assertEqual(get_idno({"idno": "TOP", **_IMAGE_DCMI}, "image"), "I1")

    def test_missing_everywhere_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_idno({"image_description": {}}, "image")


class TemplateTests(unittest.TestCase):
    def test_indicator_db_templates(self) -> None:
        self.assertEqual(_render(_INDICATOR_DB, "indicator-db", "title"), "World Development DB")
        self.assertEqual(_render(_INDICATOR_DB, "indicator-db", "sub_title"), "Annual series")
        self.assertEqual(_render(_INDICATOR_DB, "indicator-db", "abstract"), "A database of development indicators.")

    def test_table_abstract_prefers_description_then_joins_notes(self) -> None:
        self.assertEqual(_render(_TABLE, "table", "abstract"), "Employment counts.")

        no_description = {"table_description": {"notes": [{"note": "Note one."}, {"note": "Note two."}]}}
        self.assertEqual(_render(no_description, "table", "abstract"), "Note one. Note two.")

    def test_script_templates(self) -> None:
        self.assertEqual(_render(_SCRIPT, "script", "title"), "Poverty replication")
        self.assertEqual(_render(_SCRIPT, "script", "sub_title"), "Stata do-files")
        self.assertEqual(_render(_SCRIPT, "script", "abstract"), "Reproduces the poverty estimates.")

    def test_image_prefers_dcmi_over_iptc(self) -> None:
        both = {
            "image_description": {
                **_IMAGE_DCMI["image_description"],
                "iptc": _IMAGE_IPTC["image_description"]["iptc"],
            }
        }
        self.assertEqual(_render(both, "image", "title"), "Market day")
        self.assertEqual(_render(both, "image", "abstract"), "People at a market.")

    def test_image_falls_back_to_iptc_and_to_dcmi_caption(self) -> None:
        self.assertEqual(_render(_IMAGE_IPTC, "image", "title"), "Flood waters")
        self.assertEqual(_render(_IMAGE_IPTC, "image", "abstract"), "A flooded street.")

        caption_only = {"image_description": {"dcmi": {"caption": "Just a caption"}}}
        self.assertEqual(_render(caption_only, "image", "abstract"), "Just a caption")

    def test_video_templates(self) -> None:
        self.assertEqual(_render(_VIDEO, "video", "title"), "Survey fieldwork")
        self.assertEqual(_render(_VIDEO, "video", "abstract"), "Interviewers at work.")

    def test_missing_and_malformed_shapes_render_empty(self) -> None:
        cases = {
            "indicator-db": ("database_description", ("title", "sub_title", "abstract")),
            "table": ("table_description", ("title", "sub_title", "abstract")),
            "script": ("project_desc", ("title", "sub_title", "abstract")),
            "image": ("image_description", ("title", "abstract")),
            "video": ("video_description", ("title", "abstract")),
        }
        bad_values = (None, [], "text", 5, {}, {"title_statement": "oops", "dcmi": "oops", "iptc": []})
        for metadata_type, (section, fields) in cases.items():
            for field in fields:
                self.assertEqual(_render({}, metadata_type, field), "", (metadata_type, field, "missing"))
                for bad in bad_values:
                    self.assertEqual(
                        _render({section: bad}, metadata_type, field),
                        "",
                        (metadata_type, field, bad),
                    )


class ParserTests(unittest.TestCase):
    def test_indicator_db(self) -> None:
        parser = IndicatorDbParser()
        self.assertEqual(parser.parse_source(_INDICATOR_DB), ["IMF", "World Bank"])
        self.assertEqual(parser.parse_geographies(_INDICATOR_DB), ["Chad", "Kenya"])
        # ``end`` is an int in the record: it must be read as the year 2020, not epoch nanoseconds.
        self.assertEqual(parser.parse_periods(_INDICATOR_DB), "1990 - 2020")

    def test_table_uses_time_periods_then_date_published(self) -> None:
        parser = TableParser()
        self.assertEqual(parser.parse_source(_TABLE), ["NSO"])
        self.assertEqual(parser.parse_geographies(_TABLE), ["Ghana"])
        self.assertEqual(parser.parse_periods(_TABLE), "2010 - 2015")

        published_only = {"table_description": {"date_published": "2018-06-01", "publisher": [{"name": "Pub"}]}}
        self.assertEqual(parser.parse_periods(published_only), "2018")
        self.assertEqual(parser.parse_source(published_only), ["Pub"])

    def test_script_production_date_is_a_string(self) -> None:
        parser = ScriptParser()
        self.assertEqual(parser.parse_periods(_SCRIPT), "2019")
        self.assertEqual(parser.parse_periods(_SCRIPT, out_format="details")["years"], ["2019"])
        self.assertEqual(parser.parse_source(_SCRIPT), ["Analyst A", "Analyst B"])
        self.assertEqual(parser.parse_geographies(_SCRIPT), ["Peru"])

    def test_script_production_date_list_is_still_accepted(self) -> None:
        record = {"project_desc": {"production_date": ["2018-01", "2020-06"]}}
        self.assertEqual(ScriptParser().parse_periods(record), "2018 - 2020")

    def test_image_dcmi_and_iptc(self) -> None:
        parser = ImageParser()
        self.assertEqual(parser.parse_source(_IMAGE_DCMI), ["A. Photographer"])
        self.assertEqual(parser.parse_geographies(_IMAGE_DCMI), ["Mali"])
        self.assertEqual(parser.parse_periods(_IMAGE_DCMI), "2012")

        self.assertEqual(parser.parse_source(_IMAGE_IPTC), ["B. Photographer"])
        self.assertEqual(parser.parse_geographies(_IMAGE_IPTC), ["Bangladesh"])
        self.assertEqual(parser.parse_periods(_IMAGE_IPTC), "2020")

    def test_video_prefers_date_published(self) -> None:
        parser = VideoParser()
        self.assertEqual(parser.parse_source(_VIDEO), ["Field Team"])
        self.assertEqual(parser.parse_geographies(_VIDEO), ["Nepal"])
        self.assertEqual(parser.parse_periods(_VIDEO), "2021")

        created_only = {"video_description": {"date_created": "2020-01-01"}}
        self.assertEqual(parser.parse_periods(created_only), "2020")

    def test_malformed_shapes_give_empty_results(self) -> None:
        bad_records = (
            {},
            {
                "database_description": "x",
                "table_description": [],
                "project_desc": 5,
                "image_description": {"dcmi": "x", "iptc": {"photoVideoMetadataIPTC": []}},
                "video_description": {"country": "Kenya", "creator": ["", 1, None], "date_published": {}},
            },
        )
        parsers = (IndicatorDbParser(), TableParser(), ScriptParser(), ImageParser(), VideoParser())
        for parser in parsers:
            for record in bad_records:
                self.assertIsNone(parser.parse_source(record))
                self.assertIsNone(parser.parse_geographies(record))
                self.assertIsNone(parser.parse_periods(record))
                self.assertEqual(
                    parser.parse_periods(record, out_format="details"),
                    {"year_start": None, "year_end": None, "years": None},
                )


class FilterFacetTests(unittest.TestCase):
    def test_facets_per_type(self) -> None:
        cases = (
            (_INDICATOR_DB, IndicatorDbFilterFacets, "DB1", ["Chad", "Kenya"], ["IMF", "World Bank"], 1990, 2020),
            (_TABLE, TableFilterFacets, "T1", ["Ghana"], ["NSO"], 2010, 2015),
            (_SCRIPT, ScriptFilterFacets, "S1", ["Peru"], ["Analyst A", "Analyst B"], 2019, 2019),
            (_IMAGE_DCMI, ImageFilterFacets, "I1", ["Mali"], ["A. Photographer"], 2012, 2012),
            (_VIDEO, VideoFilterFacets, "V1", ["Nepal"], ["Field Team"], 2021, 2021),
        )
        for record, facets_class, idno, geographies, source, year_start, year_end in cases:
            facets = get_filter_facets(record)
            self.assertIsInstance(facets, facets_class)
            self.assertEqual(facets.type, record["type"])
            self.assertEqual(facets.idno, idno)
            self.assertEqual(facets.geographies, geographies)
            self.assertEqual(facets.source, source)
            self.assertEqual((facets.year_start, facets.year_end), (year_start, year_end))
            self.assertTrue(facets.idno_uuid)

    def test_record_with_no_facet_values_still_validates(self) -> None:
        facets = get_filter_facets({"type": "video", "video_description": {"idno": "V2"}})

        self.assertIsNone(facets.geographies)
        self.assertIsNone(facets.source)
        self.assertIsNone(facets.year_start)
        self.assertIsNone(facets.years)

    def test_unknown_type_still_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_filter_facets({"type": "citation"})

    def test_all_new_facets_are_filter_facets(self) -> None:
        for cls in (IndicatorDbFilterFacets, TableFilterFacets, ScriptFilterFacets, ImageFilterFacets, VideoFilterFacets):
            self.assertTrue(issubclass(cls, FilterFacets))


class HandlerTests(unittest.TestCase):
    def _handler(self, record: dict, idno: str):
        with mock.patch("ai4data.discovery.metadata.handler.get_metadata_json", return_value=dict(record)):
            return MetadataLoader(idno=idno, metadata_type=record["type"]).get_metadata_handler()

    def test_langdocs_and_payload_per_type(self) -> None:
        cases = (
            (_INDICATOR_DB, "DB1", {"title", "sub_title", "abstract"}),
            (_TABLE, "T1", {"title", "sub_title", "abstract"}),
            (_SCRIPT, "S1", {"title", "sub_title", "abstract"}),
            (_IMAGE_DCMI, "I1", {"title", "abstract"}),
            (_IMAGE_IPTC, "I2", {"title", "abstract"}),
            (_VIDEO, "V1", {"title", "abstract"}),
        )
        for record, idno, qfields in cases:
            handler = self._handler(record, idno)
            langdocs = handler.get_langdocs()

            self.assertEqual({d.metadata["qfield"] for d in langdocs}, qfields, record["type"])
            self.assertTrue(all(d.page_content for d in langdocs))
            for doc in langdocs:
                self.assertEqual(doc.metadata["type"], record["type"])
                self.assertEqual(doc.metadata["idno"], idno)

    def test_empty_fields_are_skipped(self) -> None:
        record = {"type": "script", "project_desc": {"title_statement": {"idno": "S2", "title": "Only a title"}}}
        langdocs = self._handler(record, "S2").get_langdocs()

        self.assertEqual([d.metadata["qfield"] for d in langdocs], ["title"])

    def test_unsupported_type_still_raises(self) -> None:
        with mock.patch(
            "ai4data.discovery.metadata.handler.get_metadata_json",
            return_value={"type": "citation", "idno": "C1"},
        ):
            loader = MetadataLoader(idno="C1", metadata_type="citation")
            with self.assertRaises(ValueError):
                loader.get_metadata_handler()


if __name__ == "__main__":
    unittest.main()
