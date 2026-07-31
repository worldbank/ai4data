"""End-to-end tests for the 3-model swarm pipeline with real model weights.

These tests load the actual GLiNER2 adapters from HuggingFace to verify the
full production pipeline: Call 1 (entity) → Call 1b (relation) → Call 2 (classification).

Marked ``slow`` — skipped by default. Run with:
    pytest tests/test_e2e_swarm_pipeline.py -v --run-slow
"""

import time

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("gliner2")


def _get_extractor():
    from ai4data.data_use.extractors.dataset_extractor import DatasetExtractor

    return DatasetExtractor()


@pytest.fixture(scope="session")
def extractor():
    return _get_extractor()


@pytest.fixture(scope="session")
def warm_models(extractor):
    _ = extractor.model
    _ = extractor.relation_model
    _ = extractor.classification_model
    return extractor


# ── Test data ─────────────────────────────────────────────────────────────────

TEXT_DHS = (
    "The World Bank published the 2021 DHS survey for Kenya, used by " "the Ministry of Health."
)

TEXT_MULTI = "WHO publishes GHO data and FAO maintains FAOSTAT. " "Both were used by UNDP."

TEXT_WDI = "The World Development Indicators (WDI) database is maintained by the World Bank."

TEXT_ALL_RELATIONS = (
    "The 2022 Demographic and Health Survey (DHS) was published by "
    "ICF International for Malawi and analyzed by the Ministry of Health."
)

TEXT_LONG = ". ".join(
    [
        "The World Bank published the 2021 DHS survey for Kenya.",
        "UNICEF conducts MICS surveys across multiple countries.",
        "WHO GHO data was used by the Ministry of Health.",
        "FAOSTAT is maintained by FAO with global coverage.",
        "The World Development Indicators (WDI) from the World Bank.",
        "ILO STAT provides labor market data used by researchers.",
        "The 2020 Ghana Population Census was conducted by GSS.",
        "EM-DAT disaster database is maintained by CRED.",
    ]
    * 5
)


def _field(ds, key):
    return ds.get(key, {}).get("text", "")


def _conf(ds, key):
    return ds.get(key, {}).get("confidence", 0.0)


class TestPipelineBasic:
    """Basic pipeline sanity checks with real model weights."""

    def test_basic_extraction(self, warm_models):
        result = warm_models.extract_from_text(TEXT_DHS, include_confidence=True)
        datasets = result.get("datasets", [])
        assert len(datasets) >= 1
        ds = datasets[0]
        assert "mention_name" in ds
        assert "producer" in ds
        assert "user" in ds
        assert "reference_year" in ds
        assert "geography" in ds
        assert "acronym" in ds
        assert "usage_context" in ds
        assert "typology_tag" in ds
        assert "purpose_action" in ds

    def test_output_has_input_text(self, warm_models):
        result = warm_models.extract_from_text(TEXT_DHS)
        assert result.get("input_text") == TEXT_DHS

    def test_empty_text(self, warm_models):
        result = warm_models.extract_from_text("")
        assert result.get("datasets") == []

    def test_no_dataset_text(self, warm_models):
        """Text with no dataset mentions should return empty."""
        result = warm_models.extract_from_text("The weather is nice today. Please pass the salt.")
        assert result.get("datasets") == []


class TestPipelineRelations:
    """All 5 relation types fire correctly."""

    def test_has_organization(self, warm_models):
        r = warm_models.extract_from_text(TEXT_WDI, include_confidence=True)
        ds = r["datasets"][0]
        assert _field(ds, "producer") in ("World Bank", "the World Bank")

    def test_used_by(self, warm_models):
        r = warm_models.extract_from_text(TEXT_DHS, include_confidence=True)
        ds = r["datasets"][0]
        assert _field(ds, "user") in ("Ministry of Health", "the Ministry of Health")

    def test_has_acronym(self, warm_models):
        r = warm_models.extract_from_text(TEXT_ALL_RELATIONS, include_confidence=True)
        ds = r["datasets"][0]
        assert _field(ds, "acronym").upper() in ("DHS",)
        assert _conf(ds, "acronym") >= 0.3

    def test_has_timeframe(self, warm_models):
        r = warm_models.extract_from_text(TEXT_DHS, include_confidence=True)
        ds = r["datasets"][0]
        yr = _field(ds, "reference_year")
        assert yr == "2021" or "2021" in yr
        assert _conf(ds, "reference_year") >= 0.3

    def test_has_geography(self, warm_models):
        r = warm_models.extract_from_text(TEXT_DHS, include_confidence=True)
        ds = r["datasets"][0]
        assert _field(ds, "geography").lower() == "kenya"
        assert _conf(ds, "geography") >= 0.3

    def test_multiple_datasets(self, warm_models):
        r = warm_models.extract_from_text(TEXT_MULTI, include_confidence=True)
        dsets = r.get("datasets", [])
        assert len(dsets) >= 2


class TestPipelineClassification:
    """Classification outputs are populated correctly."""

    def test_usage_context_present(self, warm_models):
        r = warm_models.extract_from_text(TEXT_DHS, include_confidence=True)
        ds = r["datasets"][0]
        usage = ds.get("usage_context", {}).get("text", "")
        assert usage in ("primary", "supporting", "background", "")

    def test_typology_present(self, warm_models):
        r = warm_models.extract_from_text(TEXT_DHS, include_confidence=True)
        ds = r["datasets"][0]
        typ = ds.get("typology_tag", {}).get("text", "")
        assert typ in (
            "survey",
            "census",
            "database",
            "administrative",
            "indicator",
            "geospatial",
            "microdata",
            "report",
            "estimates",
            "other",
        )

    def test_purpose_action_present(self, warm_models):
        r = warm_models.extract_from_text(TEXT_DHS, include_confidence=True)
        ds = r["datasets"][0]
        pa = ds.get("purpose_action", {}).get("text", "")
        assert pa in (
            "policy_informing",
            "contextual_reference",
            "programmatic_guidance",
            "operational_implementation",
            "capacity_building",
            "",
        )

    def test_is_used_derived_from_usage(self, warm_models):
        r = warm_models.extract_from_text(TEXT_DHS, include_confidence=True)
        ds = r["datasets"][0]
        is_used = ds.get("is_used", {}).get("text", "")
        assert is_used in ("True", "False")


class TestPipelineParallel:
    """Parallel vs sequential produce identical results."""

    def test_long_text_parallel_sequential_match(self, warm_models):
        r_seq = warm_models.extract_from_text(TEXT_LONG, include_confidence=True, parallel=False)
        r_par = warm_models.extract_from_text(TEXT_LONG, include_confidence=True, parallel=True)

        ds_seq = r_seq.get("datasets", [])
        ds_par = r_par.get("datasets", [])

        assert len(ds_seq) == len(
            ds_par
        ), f"Dataset count mismatch: seq={len(ds_seq)} par={len(ds_par)}"
        for i in range(len(ds_seq)):
            for key in (
                "mention_name",
                "producer",
                "user",
                "reference_year",
                "geography",
                "acronym",
            ):
                assert _field(ds_seq[i], key) == _field(ds_par[i], key), (
                    f"[{i}] {key}: seq={_field(ds_seq[i], key)!r} "
                    f"par={_field(ds_par[i], key)!r}"
                )
            for key in ("usage_context", "typology_tag", "purpose_action"):
                assert _field(ds_seq[i], key) == _field(ds_par[i], key), (
                    f"[{i}] {key}: seq={_field(ds_seq[i], key)!r} "
                    f"par={_field(ds_par[i], key)!r}"
                )

    def test_parallel_faster_than_sequential(self, warm_models):
        t0 = time.time()
        warm_models.extract_from_text(TEXT_LONG, include_confidence=True, parallel=False)
        t_seq = time.time() - t0

        t0 = time.time()
        warm_models.extract_from_text(TEXT_LONG, include_confidence=True, parallel=True)
        t_par = time.time() - t0

        assert t_par <= t_seq * 1.5, (
            f"Parallel ({t_par:.2f}s) should not be slower than " f"sequential ({t_seq:.2f}s)"
        )

    def test_batch_parallel_sequential_match(self, warm_models):
        texts = [TEXT_DHS, TEXT_WDI, TEXT_MULTI, TEXT_ALL_RELATIONS] * 2
        r_seq = warm_models.extract_batch(texts, include_confidence=True, parallel=False)
        r_par = warm_models.extract_batch(texts, include_confidence=True, parallel=True)
        assert len(r_seq) == len(r_par)
        for i in range(len(r_seq)):
            names_seq = [d["mention_name"]["text"] for d in r_seq[i].get("datasets", [])]
            names_par = [d["mention_name"]["text"] for d in r_par[i].get("datasets", [])]
            assert names_seq == names_par, f"[{i}] seq={names_seq} par={names_par}"


class TestPipelineEdgeCases:
    """Edge cases and robustness."""

    def test_single_word_input(self, warm_models):
        r = warm_models.extract_from_text("Hello")
        assert "datasets" in r

    def test_numeric_input(self, warm_models):
        r = warm_models.extract_from_text("12345 67890")
        assert "datasets" in r

    def test_very_long_single_chunk(self, warm_models):
        text = "Data from the DHS survey. " * 500
        r = warm_models.extract_from_text(text, include_confidence=True, enable_chunking=True)
        assert "datasets" in r
        assert isinstance(r["datasets"], list)

    def test_confidence_scores(self, warm_models):
        r = warm_models.extract_from_text(TEXT_DHS, include_confidence=True)
        ds = r["datasets"][0]
        assert "confidence" in ds["mention_name"]
        assert ds["mention_name"]["confidence"] > 0

    def test_acronym_validation_accepts_known_acronym(self, warm_models):
        from ai4data.data_use.schemas.dataset_schema import DatasetSchema

        assert DatasetSchema._is_valid_relation(
            "has_acronym", "Demographic and Health Survey", "DHS"
        )
        assert DatasetSchema._is_valid_relation(
            "has_acronym", "World Development Indicators", "WDI"
        )

    def test_acronym_validation_rejects_bad_year(self, warm_models):
        from ai4data.data_use.schemas.dataset_schema import DatasetSchema

        assert not DatasetSchema._is_valid_relation(
            "has_acronym", "Survey", "2020"
        ), "Year should not be accepted as acronym"

    def test_is_valid_org_rejects_generic_terms(self, warm_models):
        from ai4data.data_use.schemas.dataset_schema import DatasetSchema

        assert not DatasetSchema._is_valid_org("we")
        assert not DatasetSchema._is_valid_org("the authors")
        assert DatasetSchema._is_valid_org("World Bank")


class TestPipelineChunking:
    """Multi-chunk processing preserves results."""

    def test_chunked_contains_at_least_flat(self, warm_models):
        r_chunked = warm_models.extract_from_text(
            TEXT_LONG, include_confidence=True, enable_chunking=True
        )
        r_flat = warm_models.extract_from_text(
            TEXT_LONG, include_confidence=True, enable_chunking=False
        )
        names_chunked = set(d["mention_name"]["text"] for d in r_chunked.get("datasets", []))
        names_flat = set(d["mention_name"]["text"] for d in r_flat.get("datasets", []))
        missing = names_flat - names_chunked
        assert not missing, f"Chunked mode is missing datasets found by flat mode: {missing}"


class TestPipelineModelDefaults:
    """Verify default adapter IDs point to ai4data namespace."""

    def test_default_relation_adapter(self):
        ext = _get_extractor()
        assert ext.relation_adapter_id == "ai4data/datause-relation-v0"

    def test_default_classification_adapter(self):
        ext = _get_extractor()
        assert ext.classification_adapter_id == "ai4data/datause-impact-v0"
