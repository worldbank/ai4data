"""Tests for the ``include_resources`` plumbing across discovery paths, HTTP, and loader.

Covers the changes that thread an ``include_resources`` flag from
:class:`ai4data.discovery.metadata.handler.MetadataLoader` through
:func:`ai4data.discovery.catalog.http.get_metadata_json` into the cache filename
chosen by :func:`ai4data.discovery.paths.get_metadata_cache_path`.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai4data.discovery import paths as discovery_paths
from ai4data.discovery.catalog import http as catalog_http
from ai4data.discovery.metadata import handler as metadata_handler


class _FakeResponse:
    """Minimal stand-in for :class:`httpx.Response` used in cache-miss tests."""

    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _DiscoveryPathsTestCase(unittest.TestCase):
    """Base class that redirects discovery caches to a per-test temp dir."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_root = Path(self._tmpdir.name)
        discovery_paths.init_discovery_paths(self.tmp_root)
        self.addCleanup(discovery_paths.init_discovery_paths, None)


class TestGetMetadataCachePath(_DiscoveryPathsTestCase):
    def test_default_filename_omits_resources_suffix(self):
        path = discovery_paths.get_metadata_cache_path("IDN-001", "indicator")
        self.assertEqual(path.name, "indicator_IDN-001.json")
        self.assertEqual(path.parent, self.tmp_root / "metadata_cache" / "indicator")

    def test_include_resources_false_matches_default(self):
        default = discovery_paths.get_metadata_cache_path("IDN-001", "indicator")
        explicit_false = discovery_paths.get_metadata_cache_path(
            "IDN-001", "indicator", include_resources=False
        )
        self.assertEqual(default, explicit_false)

    def test_include_resources_true_uses_res_suffix(self):
        path = discovery_paths.get_metadata_cache_path(
            "IDN-001", "indicator", include_resources=True
        )
        self.assertEqual(path.name, "indicator_IDN-001.res.json")
        self.assertEqual(path.parent, self.tmp_root / "metadata_cache" / "indicator")

    def test_resource_and_default_paths_are_distinct(self):
        default = discovery_paths.get_metadata_cache_path("IDN-001", "indicator")
        with_res = discovery_paths.get_metadata_cache_path(
            "IDN-001", "indicator", include_resources=True
        )
        self.assertNotEqual(default, with_res)

    def test_parent_directory_is_created(self):
        path = discovery_paths.get_metadata_cache_path(
            "IDN-001", "document", include_resources=True
        )
        self.assertTrue(path.parent.exists())
        self.assertTrue(path.parent.is_dir())


class TestGetMetadataJson(_DiscoveryPathsTestCase):
    def test_cache_hit_short_circuits_http(self):
        idno = "IDN-CACHED"
        metadata_type = "indicator"
        cache_path = discovery_paths.get_metadata_cache_path(idno, metadata_type)
        cached_payload = {"idno": idno, "type": metadata_type, "from": "cache"}
        cache_path.write_text(json.dumps(cached_payload), encoding="utf-8")

        with mock.patch.object(catalog_http.httpx, "get") as mocked_get:
            result = catalog_http.get_metadata_json(idno, metadata_type)

        self.assertEqual(result, cached_payload)
        mocked_get.assert_not_called()

    def test_force_bypasses_cache_and_fetches(self):
        idno = "IDN-FORCE"
        metadata_type = "indicator"
        cache_path = discovery_paths.get_metadata_cache_path(idno, metadata_type)
        cache_path.write_text(json.dumps({"stale": True}), encoding="utf-8")

        fresh_payload = {"type": "indicator", "fresh": True}
        with mock.patch.object(
            catalog_http.httpx, "get", return_value=_FakeResponse(fresh_payload)
        ) as mocked_get:
            result = catalog_http.get_metadata_json(
                idno, metadata_type, force=True
            )

        self.assertEqual(result, fresh_payload)
        mocked_get.assert_called_once()
        on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, fresh_payload)

    def test_load_exists_false_returns_none_on_cache_hit(self):
        idno = "IDN-SKIPLOAD"
        metadata_type = "indicator"
        cache_path = discovery_paths.get_metadata_cache_path(idno, metadata_type)
        cache_path.write_text(json.dumps({"any": "value"}), encoding="utf-8")

        with mock.patch.object(catalog_http.httpx, "get") as mocked_get:
            result = catalog_http.get_metadata_json(
                idno, metadata_type, load_exists=False
            )

        self.assertIsNone(result)
        mocked_get.assert_not_called()

    def test_cache_miss_passes_include_resources_false_query_param(self):
        idno = "IDN-NEW"
        metadata_type = "indicator"
        fake_response = _FakeResponse({"type": "indicator", "from": "http"})

        with mock.patch.object(
            catalog_http.httpx, "get", return_value=fake_response
        ) as mocked_get:
            result = catalog_http.get_metadata_json(idno, metadata_type)

        self.assertEqual(result["from"], "http")
        mocked_get.assert_called_once()
        _, kwargs = mocked_get.call_args
        self.assertEqual(kwargs.get("params"), {"include_resources": False})

        cache_path = discovery_paths.get_metadata_cache_path(idno, metadata_type)
        self.assertTrue(cache_path.exists())
        self.assertEqual(
            json.loads(cache_path.read_text(encoding="utf-8")),
            {"type": "indicator", "from": "http"},
        )

    def test_cache_miss_passes_include_resources_true_query_param(self):
        idno = "IDN-RES"
        metadata_type = "indicator"
        fake_response = _FakeResponse(
            {"type": "indicator", "resources": [{"id": 1}]}
        )

        with mock.patch.object(
            catalog_http.httpx, "get", return_value=fake_response
        ) as mocked_get:
            catalog_http.get_metadata_json(
                idno, metadata_type, include_resources=True
            )

        _, kwargs = mocked_get.call_args
        self.assertEqual(kwargs.get("params"), {"include_resources": True})

        res_path = discovery_paths.get_metadata_cache_path(
            idno, metadata_type, include_resources=True
        )
        default_path = discovery_paths.get_metadata_cache_path(idno, metadata_type)
        self.assertTrue(res_path.exists())
        self.assertFalse(default_path.exists())

    def test_include_resources_cache_paths_are_independent(self):
        idno = "IDN-SPLIT"
        metadata_type = "indicator"

        default_payload = {"type": "indicator", "variant": "no-res"}
        res_payload = {"type": "indicator", "variant": "with-res"}

        with mock.patch.object(
            catalog_http.httpx,
            "get",
            side_effect=[_FakeResponse(default_payload), _FakeResponse(res_payload)],
        ) as mocked_get:
            first = catalog_http.get_metadata_json(idno, metadata_type)
            second = catalog_http.get_metadata_json(
                idno, metadata_type, include_resources=True
            )

        self.assertEqual(first["variant"], "no-res")
        self.assertEqual(second["variant"], "with-res")
        self.assertEqual(mocked_get.call_count, 2)

        with mock.patch.object(catalog_http.httpx, "get") as mocked_get_after:
            cached_default = catalog_http.get_metadata_json(idno, metadata_type)
            cached_res = catalog_http.get_metadata_json(
                idno, metadata_type, include_resources=True
            )

        mocked_get_after.assert_not_called()
        self.assertEqual(cached_default["variant"], "no-res")
        self.assertEqual(cached_res["variant"], "with-res")

    def test_timeseries_type_is_normalized_to_indicator(self):
        with mock.patch.object(
            catalog_http.httpx,
            "get",
            return_value=_FakeResponse({"type": "timeseries"}),
        ):
            result = catalog_http.get_metadata_json("IDN-TS", "indicator")
        self.assertEqual(result["type"], "indicator")

    def test_survey_type_is_normalized_to_microdata(self):
        with mock.patch.object(
            catalog_http.httpx,
            "get",
            return_value=_FakeResponse({"type": "survey"}),
        ):
            result = catalog_http.get_metadata_json("IDN-SV", "microdata")
        self.assertEqual(result["type"], "microdata")

    def test_type_mismatch_raises_assertion_error(self):
        with mock.patch.object(
            catalog_http.httpx,
            "get",
            return_value=_FakeResponse({"type": "document"}),
        ):
            with self.assertRaises(AssertionError):
                catalog_http.get_metadata_json("IDN-X", "indicator")

    def test_no_metadata_type_skips_cache_layer(self):
        fake_response = _FakeResponse({"type": "indicator"})
        with mock.patch.object(
            catalog_http.httpx, "get", return_value=fake_response
        ) as mocked_get:
            result = catalog_http.get_metadata_json("IDN-NOTYPE")

        self.assertEqual(result, {"type": "indicator"})
        _, kwargs = mocked_get.call_args
        self.assertEqual(kwargs.get("params"), {"include_resources": False})
        listed = list((self.tmp_root / "metadata_cache").rglob("*.json"))
        self.assertEqual(listed, [])


class TestMetadataLoaderIncludeResources(_DiscoveryPathsTestCase):
    def test_default_include_resources_is_false(self):
        with mock.patch.object(
            metadata_handler,
            "get_metadata_json",
            return_value={"type": "indicator"},
        ) as mocked:
            loader = metadata_handler.MetadataLoader(
                idno="IDN-LOAD", metadata_type="indicator"
            )

        self.assertFalse(loader.include_resources)
        mocked.assert_called_once_with(
            "IDN-LOAD",
            "indicator",
            force=False,
            include_resources=False,
        )
        self.assertEqual(loader.metadata["idno"], "IDN-LOAD")

    def test_include_resources_true_is_forwarded(self):
        with mock.patch.object(
            metadata_handler,
            "get_metadata_json",
            return_value={"type": "indicator", "resources": []},
        ) as mocked:
            loader = metadata_handler.MetadataLoader(
                idno="IDN-LOAD-RES",
                metadata_type="indicator",
                force=True,
                include_resources=True,
            )

        self.assertTrue(loader.include_resources)
        mocked.assert_called_once_with(
            "IDN-LOAD-RES",
            "indicator",
            force=True,
            include_resources=True,
        )
        self.assertEqual(loader.metadata["idno"], "IDN-LOAD-RES")


if __name__ == "__main__":
    unittest.main()
