"""Tests for anomaly explanation mapping and CSV loading helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from ai4data.anomaly.explanation.adapters import load_csv_filtered
from ai4data.anomaly.explanation.batch_builder import _compact_custom_id
from ai4data.anomaly.explanation.mapping_suggest import suggest_column_mapping
from ai4data.anomaly.explanation.legacy_custom_id import (
    new_compact_id_from_legacy_parts,
    parse_legacy_nosearch_custom_id,
    write_custom_id_map_from_legacy_batch_output,
)
from ai4data.anomaly.explanation.output_parser import parse_batch_output


class TestSuggestColumnMapping(unittest.TestCase):
    def test_wdi_style_headers(self):
        headers = [
            "Country.Code",
            "Country.Name",
            "Indicator.Code",
            "Year",
            "Value",
            "absZscore_zscore",
            "Zscore",
            "Indicator.Name",
            "Imputed",
            "outlier_indicator_total",
        ]
        m = suggest_column_mapping(headers)
        self.assertEqual(m.get("geography_id"), "Country.Code")
        self.assertEqual(m.get("indicator_id"), "Indicator.Code")
        self.assertEqual(m.get("period"), "Year")
        self.assertEqual(m.get("value"), "Value")
        self.assertEqual(m.get("is_imputed"), "Imputed")
        self.assertEqual(m.get("outlier_count"), "outlier_indicator_total")
        self.assertEqual(m.get("anomaly_score"), "absZscore_zscore")
        self.assertEqual(m.get("indicator_name"), "Indicator.Name")
        self.assertEqual(m.get("geography_name"), "Country.Name")


class TestLoadCsvFiltered(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "indicator_id": "ind",
            "indicator_name": "iname",
            "geography_id": "geo",
            "geography_name": "gname",
            "period": "yr",
            "value": "val",
            "is_imputed": "imp",
            "anomaly_score": "sc",
            "outlier_count": "oc",
        }

    def test_regex_filter(self):
        csv = (
            "ind,iname,geo,gname,yr,val,imp,sc,oc\n"
            "SI.POV.DUMMY,x,AA,aa,2020,1.0,False,1.0,3\n"
            "EG.ELC.ACCS,x,BB,bb,2020,2.0,False,1.0,3\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv)
            path = f.name
        try:
            df = load_csv_filtered(
                path,
                self.mapping,
                indicator_id_pattern=r"^SI\.",
            )
            self.assertEqual(len(df), 1)
            self.assertEqual(df["indicator_id"].iloc[0], "SI.POV.DUMMY")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_chunksize_concat(self):
        csv = (
            "ind,iname,geo,gname,yr,val,imp,sc,oc\n"
            "SI.A,x,AA,aa,2020,1.0,False,1.0,3\n"
            "SI.B,x,AA,aa,2021,1.0,False,1.0,3\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(csv)
            path = f.name
        try:
            df = load_csv_filtered(
                path,
                self.mapping,
                indicator_id_pattern=r"^SI\.",
                chunksize=1,
            )
            self.assertEqual(len(df), 2)
        finally:
            Path(path).unlink(missing_ok=True)


class TestLegacyNosearchCustomId(unittest.TestCase):
    def test_parse_roundtrip_shape(self):
        md5_32 = "a" * 32
        cid = f"nosearch-abcdef12-SI.POV.GAPS-CZE-{md5_32}"
        p = parse_legacy_nosearch_custom_id(cid)
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p["prompt_hash"], "abcdef12")
        self.assertEqual(p["indicator_id"], "SI.POV.GAPS")
        self.assertEqual(p["geography_id"], "CZE")
        self.assertEqual(p["context_md5_hex"], md5_32)

    def test_new_compact_matches_pipeline(self):
        ph = "abcd1234"
        ctx = '{"Indicator": "SI.X", "Country": "YY", "Series": []}'
        old = f"nosearch-{ph}-SI.X-YY-{'b' * 32}"
        parsed = parse_legacy_nosearch_custom_id(old)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        new_id = new_compact_id_from_legacy_parts(
            parsed["prompt_hash"],
            parsed["indicator_id"],
            parsed["geography_id"],
            ctx,
        )
        from ai4data.anomaly.explanation.batch_builder import compact_custom_id

        self.assertEqual(
            new_id,
            compact_custom_id(ph, "SI.X", "YY", ctx),
        )

    def test_write_map_from_legacy_output(self):
        md5_32 = "c" * 32
        cid = f"nosearch-phash01-SI.ZZ-USA-{md5_32}"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            outp = tmp / "job_out.jsonl"
            outp.write_text(
                json.dumps({"custom_id": cid, "response": {}}) + "\n",
                encoding="utf-8",
            )
            mpath = write_custom_id_map_from_legacy_batch_output(outp)
            self.assertTrue(mpath.exists())
            data = json.loads(mpath.read_text(encoding="utf-8"))
            self.assertEqual(
                data[cid],
                {"indicator_id": "SI.ZZ", "geography_id": "USA"},
            )


class TestCompactCustomId(unittest.TestCase):
    def test_length_within_anthropic_limit(self):
        cid = _compact_custom_id(
            "abcd1234",
            "SI.POV.DUMMY.LONG.INDICATOR.CODE",
            "CZE",
            '{"Indicator": "x"}',
        )
        self.assertLessEqual(len(cid), 64)
        self.assertTrue(cid.startswith("a1"))


class TestParseBatchOutputWithMap(unittest.TestCase):
    def test_resolves_indicator_from_sidecar_map(self):
        cid = _compact_custom_id("ph", "SI.TEST", "USA", "{}")
        content = json.dumps(
            {
                "anomalies": [
                    {
                        "window": [2000, 2001],
                        "is_anomaly": False,
                        "classification": "insufficient_data",
                        "confidence": 0.5,
                        "explanation": "n/a",
                        "evidence_strength": "no_evidence",
                        "evidence_source": [],
                        "source": "llm_inferred",
                    }
                ]
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            map_path = tmp / "batch_custom_id_map.json"
            out_path = tmp / "batch_out.jsonl"
            map_path.write_text(
                json.dumps(
                    {cid: {"indicator_id": "SI.TEST", "geography_id": "USA"}},
                ),
                encoding="utf-8",
            )
            line = {
                "custom_id": cid,
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": content}},
                        ]
                    }
                },
            }
            out_path.write_text(json.dumps(line) + "\n", encoding="utf-8")
            df = parse_batch_output(
                out_path,
                "openai",
                {"SI.TEST": "Test ind"},
                {"USA": "United States"},
                custom_id_map_path=map_path,
            )
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["indicator_code"], "SI.TEST")
            self.assertEqual(df.iloc[0]["country_code"], "USA")

    def test_auto_resolve_map_next_to_out_file(self):
        cid = _compact_custom_id("ph", "SI.X", "YY", "{}")
        content = json.dumps({"anomalies": []})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "job_custom_id_map.json").write_text(
                json.dumps(
                    {cid: {"indicator_id": "SI.X", "geography_id": "YY"}},
                ),
                encoding="utf-8",
            )
            out_path = tmp / "job_out.jsonl"
            line = {
                "custom_id": cid,
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": content}},
                        ]
                    }
                },
            }
            out_path.write_text(json.dumps(line) + "\n", encoding="utf-8")
            df = parse_batch_output(
                out_path,
                "openai",
                {"SI.X": "x"},
                {"YY": "y"},
            )
            self.assertEqual(len(df), 0)


class TestAnthropicCompatibleSchema(unittest.TestCase):
    def test_window_min_items_relaxed_for_anthropic(self):
        from ai4data.anomaly.explanation.batch_builder import (
            anthropic_compatible_anomaly_json_schema,
        )
        from ai4data.anomaly.explanation.schemas import AnomalyExplanation

        raw = AnomalyExplanation.model_json_schema()
        self.assertEqual(
            raw["$defs"]["Anomaly"]["properties"]["window"]["minItems"],
            2,
        )

        fixed = anthropic_compatible_anomaly_json_schema()
        win = fixed["$defs"]["Anomaly"]["properties"]["window"]
        self.assertEqual(win["minItems"], 1)
        self.assertNotIn("maxItems", win)

    def test_no_array_min_items_outside_zero_or_one(self):
        from ai4data.anomaly.explanation.batch_builder import (
            anthropic_compatible_anomaly_json_schema,
        )

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "array":
                    mi = node.get("minItems")
                    if mi is not None:
                        self.assertIn(mi, (0, 1), msg=f"bad minItems={mi!r}")
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for x in node:
                    walk(x)

        walk(anthropic_compatible_anomaly_json_schema())

    def test_no_array_has_max_items(self):
        from ai4data.anomaly.explanation.batch_builder import (
            anthropic_compatible_anomaly_json_schema,
        )

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "array":
                    self.assertNotIn(
                        "maxItems",
                        node,
                        msg="Anthropic rejects maxItems on arrays",
                    )
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for x in node:
                    walk(x)

        walk(anthropic_compatible_anomaly_json_schema())


class TestExplainersAnthropic(unittest.TestCase):
    def test_parse_anthropic_row(self):
        from ai4data.anomaly.explanation.explainers import _parse_anthropic_row

        row = {
            "custom_id": "x",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"anomalies": []}',
                        }
                    ]
                },
            },
        }
        out = _parse_anthropic_row(row)
        self.assertIsNotNone(out)
        self.assertEqual(out.get("anomalies"), [])


class TestGeminiBatchRowIncludesSchema(unittest.TestCase):
    def test_generation_config_has_structured_output(self):
        from ai4data.anomaly.explanation.batch_builder import _format_row_gemini
        from ai4data.anomaly.explanation.llm_client import build_payload
        from ai4data.anomaly.explanation.prompts import get_anomaly_response_format

        payload = build_payload(
            endpoint="responses",
            model_id="gemini-2.5-flash",
            system_prompt="sys",
            user_prompt="user",
            response_format=get_anomaly_response_format(),
            with_search=False,
        )
        row = _format_row_gemini(
            "kid",
            payload,
            "gemini",
            system_prompt="sys",
            user_prompt="user",
        )
        gc = row["request"]["generation_config"]
        self.assertEqual(gc.get("responseMimeType"), "application/json")
        self.assertIn("responseJsonSchema", gc)
        self.assertEqual(
            gc["responseJsonSchema"].get("title"),
            "AnomalyExplanation",
        )


class TestExplainersGemini(unittest.TestCase):
    def test_parse_gemini_row_markdown_fenced_and_wrapped_object(self):
        from ai4data.anomaly.explanation.explainers import _parse_gemini_row

        text = '```json\n{"anomalies": [{"window": [2000, 2001], "is_anomaly": true, "classification": "data_error", "confidence": 0.5, "explanation": "x", "evidence_strength": "no_evidence", "evidence_source": [], "source": "llm_inferred"}]}\n```'
        row = {
            "response": {
                "candidates": [
                    {"content": {"parts": [{"text": text}]}}
                ]
            }
        }
        out = _parse_gemini_row(row)
        self.assertIsNotNone(out)
        self.assertEqual(len(out["anomalies"]), 1)
        self.assertEqual(out["anomalies"][0]["window"], [2000, 2001])

    def test_parse_gemini_row_top_level_array(self):
        from ai4data.anomaly.explanation.explainers import _parse_gemini_row

        text = "```json\n[]\n```"
        row = {
            "response": {
                "candidates": [
                    {"content": {"parts": [{"text": text}]}}
                ]
            }
        }
        out = _parse_gemini_row(row)
        self.assertIsNotNone(out)
        self.assertEqual(out["anomalies"], [])


if __name__ == "__main__":
    unittest.main()
