"""Tests for DatasetExtractor."""

from ai4data.data_use.extractors.dataset_extractor import DatasetExtractor


class TestDatasetExtractor:
    """Test suite for DatasetExtractor class."""

    def test_initialization(self):
        """Test extractor initialization with default parameters."""
        extractor = DatasetExtractor()
        assert extractor.threshold == 0.3
        assert extractor.model_id is None
        assert extractor._model is None
        assert extractor._schema_core is None
        assert extractor._schema_provenance is None

    def test_initialization_with_custom_params(self):
        """Test extractor initialization with custom parameters."""
        extractor = DatasetExtractor(
            model_id="custom-model", threshold=0.9, cache_dir="./test_cache"
        )
        assert extractor.threshold == 0.9
        assert extractor.model_id == "custom-model"
        assert extractor.model_manager.cache_dir == "./test_cache"

    def test_lazy_model_loading(self, mock_model_manager, mock_gliner_model):
        """Test that model loads lazily on first access."""
        extractor = DatasetExtractor()
        assert extractor._model is None

        # Access model property triggers loading
        model = extractor.model
        assert model is not None

    def test_extract_from_text_basic(self, mock_model_manager, sample_text):
        """Test basic text extraction."""
        extractor = DatasetExtractor()
        results = extractor.extract_from_text(sample_text, include_confidence=False)

        assert isinstance(results, dict)
        assert "datasets" in results

    def test_extract_from_text_with_confidence(self, mock_model_manager, sample_text):
        """Test text extraction with confidence scores."""
        extractor = DatasetExtractor()
        results = extractor.extract_from_text(sample_text, include_confidence=True)

        assert isinstance(results, dict)
        # Confidence should be included in mock response
        if results.get("datasets"):
            first_mention = results["datasets"][0]
            if "dataset_name" in first_mention:
                assert "confidence" in first_mention["dataset_name"]

    def test_extract_batch(self, mock_model_manager):
        """Test batch extraction."""
        extractor = DatasetExtractor()
        texts = [
            "We used the DHS survey data.",
            "The census data from 2020 was analyzed.",
            "Administrative records were collected.",
        ]

        results = extractor.extract_batch(texts)

        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)

    def test_extract_from_text_empty_string(self, mock_model_manager):
        """Test extraction from empty string."""
        extractor = DatasetExtractor()
        results = extractor.extract_from_text("")

        assert isinstance(results, dict)

    def test_custom_schema(self, mock_model_manager, sample_text):
        """Test extraction with custom schema."""
        from ai4data.data_use.schemas.dataset_schema import DatasetSchema

        extractor = DatasetExtractor()
        schema_builder = DatasetSchema(threshold=0.95)
        custom_schema = schema_builder.build(extractor.model)

        results = extractor.extract_from_text(sample_text, custom_schema=custom_schema)

        assert isinstance(results, dict)


class TestClassifierPreFilter:
    """Tests for the use_classifier pre-filtering logic in extract_from_text."""

    def test_use_classifier_skips_non_english(self, mock_model_manager, mock_gliner_model):
        """When use_classifier=True and text is non-English, extraction is skipped entirely."""
        non_english_text = "Este es un análisis de los datos de pobreza en América Latina."

        extractor = DatasetExtractor()
        result = extractor.extract_from_text(non_english_text, use_classifier=True)
        assert result["datasets"] == []
        assert result["skip_reason"] == "non_english"
        mock_gliner_model.extract_json.assert_not_called()

    def test_use_classifier_passes_english(
        self, mock_model_manager, mock_gliner_model, mock_classifier_pipeline
    ):
        """When use_classifier=True and text is English, extraction proceeds."""
        english_text = "We used the Demographic and Health Survey (DHS) for this analysis."
        extractor = DatasetExtractor()
        result = extractor.extract_from_text(english_text, use_classifier=True)

        assert "skip_reason" not in result
        mock_gliner_model.extract.assert_called_once()

    def test_use_classifier_false_does_not_filter(
        self, mock_model_manager, mock_gliner_model, mock_classifier_pipeline
    ):
        """When use_classifier=False, classification is skipped and extraction proceeds."""
        non_english_text = "Este es un análisis de los datos de pobreza en América Latina."
        extractor = DatasetExtractor()
        extractor.extract_from_text(non_english_text, use_classifier=False)

        mock_classifier_pipeline.assert_not_called()
        mock_gliner_model.extract.assert_called_once()


class TestExtractFromDocumentForwarding:
    """Tests that extract_from_document correctly forwards use_classifier and skip_references."""

    def test_extract_from_document_forwards_use_classifier(self, monkeypatch, mock_model_manager):
        """extract_from_document passes use_classifier down to each extract_from_text call."""
        captured = {}

        def fake_extract_from_text(self, text, **kwargs):
            captured["use_classifier"] = kwargs.get("use_classifier")
            return {"input_text": text, "datasets": []}

        from ai4data.data_use.extractors import dataset_extractor as de_module

        monkeypatch.setattr(de_module.DatasetExtractor, "extract_from_text", fake_extract_from_text)

        # Fake a 1-page PDF via load_pdf_chunks
        monkeypatch.setattr(
            de_module.DocumentParser,
            "load_pdf_chunks",
            lambda *a, **kw: [{"text": "Some page text", "pages": [0]}],
        )

        extractor = de_module.DatasetExtractor()
        extractor.extract_from_document("dummy.pdf", use_classifier=True)

        assert captured.get("use_classifier") is True

    def test_extract_from_document_forwards_skip_references(self, monkeypatch, mock_model_manager):
        """extract_from_document passes skip_references (and verbose) to load_pdf_chunks."""
        captured = {}

        def fake_load_pdf_chunks(source, n_pages=1, skip_references=False, verbose=False, **kw):
            captured["skip_references"] = skip_references
            captured["verbose"] = verbose
            return [{"text": "Some page text", "pages": [0]}]

        from ai4data.data_use.extractors import dataset_extractor as de_module

        monkeypatch.setattr(de_module.DocumentParser, "load_pdf_chunks", fake_load_pdf_chunks)

        extractor = de_module.DatasetExtractor()
        extractor.extract_from_document("dummy.pdf", skip_references=True, verbose=True)

        assert captured.get("skip_references") is True
        assert captured.get("verbose") is True


class TestDatasetExtractorInit:
    """Tests for new slots added to DatasetExtractor.__init__."""

    def test_classifier_slot_initialised_to_none(self):
        """_classifier should start as None and be loaded lazily."""
        extractor = DatasetExtractor()
        assert extractor._classifier is None

    def test_classifier_property_calls_load_classifier(
        self, mock_model_manager, mock_classifier_pipeline
    ):
        """Accessing .classifier loads via ModelManager and caches the result."""
        extractor = DatasetExtractor()
        clf = extractor.classifier
        assert clf is mock_classifier_pipeline
        # Second access returns the same cached object
        clf2 = extractor.classifier
        assert clf2 is mock_classifier_pipeline
        assert extractor._classifier is mock_classifier_pipeline


class TestBertClassifierPreFilter:
    """Tests for the BERT classifier (stage 2) in extract_from_text."""

    def test_no_data_prediction_skips_extraction(
        self, mock_model_manager, mock_gliner_model, mock_classifier_pipeline
    ):
        """When BERT returns NO_DATA, extraction is skipped and skip_reason='no_data'."""
        mock_classifier_pipeline.return_value = [{"label": "NO_DATA", "score": 0.98}]

        english_text = "The regression coefficient on the interaction term is significant."
        extractor = DatasetExtractor()
        result = extractor.extract_from_text(english_text, use_classifier=True)

        assert result["datasets"] == []
        assert result["skip_reason"] == "no_data"
        # GLiNER2 must not have been called
        mock_gliner_model.extract_json.assert_not_called()

    def test_with_data_prediction_proceeds_to_extraction(
        self, mock_model_manager, mock_gliner_model, mock_classifier_pipeline
    ):
        """When BERT returns WITH_DATA, GLiNER2 is called and no skip_reason is set."""
        mock_classifier_pipeline.return_value = [{"label": "WITH_DATA", "score": 0.96}]

        english_text = "We used the Demographic and Health Survey (DHS) for this analysis."
        extractor = DatasetExtractor()
        result = extractor.extract_from_text(english_text, use_classifier=True)

        assert "skip_reason" not in result
        mock_gliner_model.extract.assert_called_once()

    def test_non_english_skips_before_classifier(
        self, mock_model_manager, mock_gliner_model, mock_classifier_pipeline
    ):
        """Non-English text is rejected by is_english (stage 1) before BERT is called."""
        non_english = "Este es un análisis de los datos de pobreza en América Latina."
        extractor = DatasetExtractor()
        result = extractor.extract_from_text(non_english, use_classifier=True)

        assert result["skip_reason"] == "non_english"
        # BERT classifier must not have been called
        mock_classifier_pipeline.assert_not_called()
        mock_gliner_model.extract.assert_not_called()

    def test_skip_reason_absent_when_classifier_disabled(
        self, mock_model_manager, mock_gliner_model, mock_classifier_pipeline
    ):
        """When use_classifier=False, skip_reason is never set."""
        non_english = "Este es un análisis de los datos de pobreza en América Latina."
        extractor = DatasetExtractor()
        result = extractor.extract_from_text(non_english, use_classifier=False)

        assert "skip_reason" not in result
        mock_classifier_pipeline.assert_not_called()
        mock_gliner_model.extract.assert_called_once()

    def test_no_data_verbose_prints_skip_message(
        self, mock_model_manager, mock_classifier_pipeline, capsys
    ):
        """When verbose=True and BERT predicts NO_DATA, a SKIP message is printed."""
        mock_classifier_pipeline.return_value = [{"label": "NO_DATA", "score": 0.99}]

        extractor = DatasetExtractor()
        extractor.extract_from_text(
            "The coefficient is statistically significant.",
            use_classifier=True,
            verbose=True,
            _page_label="page 5",
        )

        captured = capsys.readouterr()
        assert "SKIP page 5" in captured.out
        assert "NO_DATA" in captured.out


class TestExtractFromDocumentOutputFormat:
    """Tests for classifier_skipped and skip_reason fields in extract_from_document output."""

    def test_skipped_page_has_classifier_skipped_true(self, monkeypatch, mock_model_manager):
        """Pages skipped by the classifier must have classifier_skipped=True and skip_reason set."""
        from ai4data.data_use.extractors import dataset_extractor as de_module

        monkeypatch.setattr(
            de_module.DocumentParser,
            "load_pdf_chunks",
            lambda *a, **kw: [{"text": "Some text", "pages": [0]}],
        )

        def fake_extract(self, text, **kwargs):
            return {"input_text": text, "datasets": [], "skip_reason": "no_data"}

        monkeypatch.setattr(de_module.DatasetExtractor, "extract_from_text", fake_extract)

        extractor = de_module.DatasetExtractor()
        results = extractor.extract_from_document("dummy.pdf", use_classifier=True)

        assert len(results) == 1
        assert results[0]["classifier_skipped"] is True
        assert results[0]["skip_reason"] == "no_data"

    def test_non_skipped_page_has_classifier_skipped_false(self, monkeypatch, mock_model_manager):
        """Pages processed normally must have classifier_skipped=False and skip_reason=None."""
        from ai4data.data_use.extractors import dataset_extractor as de_module

        monkeypatch.setattr(
            de_module.DocumentParser,
            "load_pdf_chunks",
            lambda *a, **kw: [{"text": "Some text", "pages": [0]}],
        )

        def fake_extract(self, text, **kwargs):
            return {"input_text": text, "datasets": []}

        monkeypatch.setattr(de_module.DatasetExtractor, "extract_from_text", fake_extract)

        extractor = de_module.DatasetExtractor()
        results = extractor.extract_from_document("dummy.pdf", use_classifier=True)

        assert len(results) == 1
        assert results[0]["classifier_skipped"] is False
        assert results[0]["skip_reason"] is None

    def test_output_contains_required_keys(self, monkeypatch, mock_model_manager):
        """Every page result must contain all documented output keys."""
        from ai4data.data_use.extractors import dataset_extractor as de_module

        monkeypatch.setattr(
            de_module.DocumentParser,
            "load_pdf_chunks",
            lambda *a, **kw: [{"text": "Some text", "pages": [0]}],
        )

        def fake_extract(self, text, **kwargs):
            return {"input_text": text, "datasets": []}

        monkeypatch.setattr(de_module.DatasetExtractor, "extract_from_text", fake_extract)

        extractor = de_module.DatasetExtractor()
        results = extractor.extract_from_document("dummy.pdf")

        required_keys = (
            "page",
            "input_text",
            "datasets",
            "classifier_skipped",
            "skip_reason",
            "document",
        )
        for page_result in results:
            for key in required_keys:
                assert key in page_result, f"Missing key '{key}' in page result"

    def test_ignore_contained_acronym(self, mock_model_manager, mock_gliner_model):
        """Test that acronyms contained in the dataset name are ignored/cleared."""
        prefix_offset = 83
        mock_gliner_model.extract.return_value = {
            "entities": {
                "name": [
                    {
                        "text": "Demographic and Health Survey (DHS)",
                        "confidence": 0.95,
                        "start": prefix_offset,
                        "end": prefix_offset + 35,
                    }
                ]
            },
            "relation_extraction": {
                "has_acronym": [
                    {
                        "head": {
                            "text": "Demographic and Health Survey (DHS)",
                            "start": prefix_offset,
                            "end": prefix_offset + 35,
                        },
                        "tail": {
                            "text": "DHS",
                            "start": prefix_offset + 31,
                            "end": prefix_offset + 34,
                            "confidence": 0.95,
                        },
                        "label": "has_acronym",
                        "score": 0.95,
                    }
                ]
            },
        }
        mock_gliner_model.batch_extract.return_value = [
            {
                "entities": {
                    "specificity": [{"text": "named", "confidence": 0.95, "start": 13, "end": 18}],
                    "usage": [{"text": "primary", "confidence": 0.95, "start": 47, "end": 54}],
                }
            }
        ]
        extractor = DatasetExtractor()
        results = extractor.extract_from_text("Dummy text", include_confidence=True)
        assert len(results["datasets"]) == 1
        acro = results["datasets"][0]["acronym"]
        assert acro["text"] == ""
        assert acro["clean_text"] == ""
        assert acro["start"] is None
        assert acro["end"] is None

    def test_extract_from_document_forwards_pages(self, monkeypatch, mock_model_manager):
        """extract_from_document passes pages parameter down to load_pdf_chunks."""
        captured = {}

        def fake_load_pdf_chunks(
            source, n_pages=1, skip_references=False, verbose=False, pages=None, **kw
        ):
            captured["pages"] = pages
            return [{"text": "Some page text", "pages": [0]}]

        from ai4data.data_use.extractors import dataset_extractor as de_module

        monkeypatch.setattr(de_module.DocumentParser, "load_pdf_chunks", fake_load_pdf_chunks)

        extractor = de_module.DatasetExtractor()
        extractor.extract_from_document("dummy.pdf", pages=[2, 3])

        assert captured.get("pages") == [2, 3]

    def test_deduplicate_overlapping_entities(self, mock_model_manager, mock_gliner_model):
        """Test that overlapping entities are suppressed, favoring specificity > confidence > length."""
        # Overlapping entities:
        # A: "Ghana Living Standard Survey (GLSS)" at [0, 35], named, conf 0.9
        # B: "Ghana Living Standard Survey" at [0, 28], named, conf 0.9
        # C: "Ghana" at [0, 5], vague, conf 0.9
        prefix_offset = 83
        mock_gliner_model.extract.return_value = {
            "entities": {
                "name": [
                    {
                        "text": "Ghana Living Standard Survey (GLSS)",
                        "confidence": 0.9,
                        "start": prefix_offset,
                        "end": prefix_offset + 35,
                    },
                    {
                        "text": "Ghana Living Standard Survey",
                        "confidence": 0.9,
                        "start": prefix_offset,
                        "end": prefix_offset + 28,
                    },
                    {
                        "text": "Ghana",
                        "confidence": 0.9,
                        "start": prefix_offset,
                        "end": prefix_offset + 5,
                    },
                ]
            },
            "relation_extraction": {},
        }
        mock_gliner_model.batch_extract.return_value = [
            {
                "entities": {
                    "specificity": [{"text": "named", "confidence": 0.9, "start": 13, "end": 18}],
                    "usage": [{"text": "primary", "confidence": 0.9, "start": 47, "end": 54}],
                }
            },
            {
                "entities": {
                    "specificity": [{"text": "named", "confidence": 0.9, "start": 13, "end": 18}],
                    "usage": [{"text": "primary", "confidence": 0.9, "start": 47, "end": 54}],
                }
            },
            {
                "entities": {
                    "specificity": [{"text": "vague", "confidence": 0.9, "start": 13, "end": 18}],
                    "usage": [{"text": "primary", "confidence": 0.9, "start": 47, "end": 54}],
                }
            },
        ]
        extractor = DatasetExtractor()
        results = extractor.extract_from_text(
            "Ghana Living Standard Survey (GLSS)", include_confidence=True
        )

        # Only the longest named mention should survive the overlap suppression
        assert len(results["datasets"]) == 1
        name = results["datasets"][0]["mention_name"]["text"]
        assert name == "Ghana Living Standard Survey (GLSS)"


class TestHeuristicFilters:
    """Tests for enhanced postprocessing heuristic filters in DatasetExtractor."""

    def test_table_figure_ref_filter(self):
        """Test filtering of complex and compound table/figure references."""
        extractor = DatasetExtractor()

        # Simple and complex table/figure refs should return True (meaning they are refs)
        assert extractor._is_table_figure_ref("Table A/1") is True
        assert extractor._is_table_figure_ref("Panel X") is True
        assert extractor._is_table_figure_ref("Figure Y") is True
        assert extractor._is_table_figure_ref("Table A-1") is True
        assert extractor._is_table_figure_ref("Figure 1a") is True
        assert extractor._is_table_figure_ref("Table A/1, Panel X, Figure Y") is True
        assert extractor._is_table_figure_ref("Figure 2 and Table 3") is True
        assert extractor._is_table_figure_ref("Table 1.1") is True
        assert extractor._is_table_figure_ref("Table 10") is True

        # Legitimate names/descriptions should return False
        assert extractor._is_table_figure_ref("World Development Indicators") is False
        assert extractor._is_table_figure_ref("Map of Ghana") is False
        assert extractor._is_table_figure_ref("Table 1 and other datasets") is False

    def test_all_caps_document_title_filter(self):
        """Test filtering of long all-caps document titles."""
        extractor = DatasetExtractor()

        # All-caps long titles should return True (even if they contain dataset keywords)
        assert (
            extractor._is_all_caps_document_title(
                "A POVERTY ASSESSMENT OF UKRAINIAN REFUGEES IN NEIGHBORING COUNTRIES UKRAINE REFUGEE"
            )
            is True
        )
        assert extractor._is_all_caps_document_title("GHANA LIVING STANDARDS SURVEY") is True
        assert extractor._is_all_caps_document_title("WORLD DEVELOPMENT INDICATORS") is True

        # Non-all-caps or short titles should return False
        assert (
            extractor._is_all_caps_document_title("A Poverty Assessment of Ukrainian Refugees")
            is False
        )
        assert extractor._is_all_caps_document_title("SHORT TITLE") is False
        assert extractor._is_all_caps_document_title("Ghana Living Standards Survey") is False

    def test_apply_heuristic_filters(self):
        """Test apply_heuristic_filters method removes correct false positives."""
        extractor = DatasetExtractor()

        datasets = [
            {"mention_name": {"text": "Table A/1"}},
            {"mention_name": {"text": "Panel X"}},
            {"mention_name": {"text": "World Development Indicators"}},
            {"mention_name": {"text": "World Bank Policy Research Working Paper No. 10145"}},
            {"mention_name": {"text": "Fiduciary Systems Assessment"}},
            {
                "mention_name": {
                    "text": "A POVERTY ASSESSMENT OF UKRAINIAN REFUGEES IN NEIGHBORING COUNTRIES UKRAINE REFUGEE"
                }
            },
            {"mention_name": {"text": "GHANA LIVING STANDARDS SURVEY"}},
            {"mention_name": {"text": "Ghana Living Standards Survey"}},
            {"mention_name": {"text": "Konstantin Fastovets (UNHCR)"}},
            {"mention_name": {"text": "Erin Neale (IOM)"}},
        ]

        filtered = extractor._apply_heuristic_filters(datasets)
        filtered_names = [d["mention_name"]["text"] for d in filtered]

        assert "World Development Indicators" in filtered_names
        assert "Ghana Living Standards Survey" in filtered_names
        assert "Table A/1" not in filtered_names
        assert "Panel X" not in filtered_names
        assert "World Bank Policy Research Working Paper No. 10145" not in filtered_names
        assert "Fiduciary Systems Assessment" not in filtered_names
        assert (
            "A POVERTY ASSESSMENT OF UKRAINIAN REFUGEES IN NEIGHBORING COUNTRIES UKRAINE REFUGEE"
            not in filtered_names
        )
        assert "GHANA LIVING STANDARDS SURVEY" not in filtered_names
        assert "Konstantin Fastovets (UNHCR)" not in filtered_names
        assert "Erin Neale (IOM)" not in filtered_names

    def test_is_personal_name(self):
        """Test detection of personal names using probablepeople and heuristics."""
        extractor = DatasetExtractor()

        # Legitimate personal names (should return True)
        assert extractor._is_personal_name("Konstantin Fastovets (UNHCR)") is True
        assert extractor._is_personal_name("Erin Neale (IOM)") is True
        assert extractor._is_personal_name("Susanne Klink") is True
        assert extractor._is_personal_name("Mihail Peleah") is True
        assert extractor._is_personal_name("Marco Delogu") is True
        assert extractor._is_personal_name("Steven Bunce") is True
        assert extractor._is_personal_name("Jane Doe") is True
        assert extractor._is_personal_name("John Smith") is True

        # Legitimate datasets (should return False)
        assert extractor._is_personal_name("Demographic and Health Survey") is False
        assert extractor._is_personal_name("Living Standards Measurement Study") is False
        assert extractor._is_personal_name("health status of refugees") is False
        assert extractor._is_personal_name("integrated household survey") is False
        assert extractor._is_personal_name("UNHCR Microdata Library") is False
        assert extractor._is_personal_name("labor force survey") is False
        assert extractor._is_personal_name("OECD") is False
        assert extractor._is_personal_name("MSNA") is False
        assert extractor._is_personal_name("Multi-Sectoral Needs Assessments (MSNA)") is False
