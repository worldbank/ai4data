import json
import pytest

from dsa.constants import ROOT
from dsa.utils import load_json
from dsa.adapters.labelstudio import (
    convert_labelstudio_export_to_eval_v13,
    LS_EXPORT_JSON_PATH,
)
from dsa.adapters.doclayoutyolo import (
    DocLayoutYOLOConfig,
    run_doclayout_yolo_adapter_directory,
)
from dsa.adapters.tfid import TFIDConfig, run_tfid_adapter_directory
from dsa.adapters.yolo26 import YOLO26Config, run_yolo26_adapter_directory
from dsa.adapters.yolo11 import YOLO11Config, run_yolo11_adapter_directory


@pytest.mark.skip(reason="For debugging purposes only.")
def test_labelstudio():
    ref_path = ROOT / "tests/data/ground_truth.json"
    test_path = ROOT / "tests/data/ground_truth_test.json"

    convert_labelstudio_export_to_eval_v13(LS_EXPORT_JSON_PATH, test_path)

    ref = load_json(ref_path)
    test = load_json(test_path)

    assert json.dumps(ref) == json.dumps(test)

    # Delete test file
    test_path.unlink()


@pytest.mark.skip(reason="For debugging purposes only.")
def test_doclayoutyolo():
    ref_path = ROOT / "tests/data/doclayout-yolo.json"
    test_path = ROOT / "tests/data/doclayout-yolo_test.json"

    cfg = DocLayoutYOLOConfig()
    run_doclayout_yolo_adapter_directory(
        input_pdf_dir=ROOT / "pdf_input",
        output_json_path=test_path,
        run_id=None,
        config=cfg,
    )

    ref = load_json(ref_path)
    del ref["info"]
    test = load_json(test_path)
    del test["info"]

    assert json.dumps(ref) == json.dumps(test)

    # Delete test file
    test_path.unlink()


@pytest.mark.skip(reason="For debugging purposes only.")
def test_tfid():
    ref_path = ROOT / "tests/data/tfid-large.json"
    test_path = ROOT / "tests/data/tfid-large_test.json"

    cfg = TFIDConfig()
    run_tfid_adapter_directory(
        ROOT / "pdf_input",
        test_path,
        config=cfg,
    )

    ref = load_json(ref_path)
    del ref["info"]
    test = load_json(test_path)
    del test["info"]

    assert json.dumps(ref) == json.dumps(test)

    # Delete test file
    test_path.unlink()


@pytest.mark.skip(reason="For debugging purposes only.")
def test_yolo26():
    ref_path = ROOT / "tests/data/yolo26.json"
    test_path = ROOT / "tests/data/yolo26_test.json"

    cfg = YOLO26Config()
    run_yolo26_adapter_directory(
        input_pdf_dir=ROOT / "pdf_input",
        output_json_path=test_path,
        run_id=None,
        config=cfg,
    )

    ref = load_json(ref_path)
    del ref["info"]
    test = load_json(test_path)
    del test["info"]

    assert json.dumps(ref) == json.dumps(test)

    # Delete test file
    test_path.unlink()


@pytest.mark.skip(reason="For debugging purposes only.")
def test_yolo11():
    ref_path = ROOT / "tests/data/yolo11.json"
    test_path = ROOT / "tests/data/yolo11_test.json"

    cfg = YOLO11Config()
    run_yolo11_adapter_directory(
        input_pdf_dir=ROOT / "pdf_input",
        output_json_path=test_path,
        run_id=None,
        config=cfg,
    )

    ref = load_json(ref_path)
    del ref["info"]
    test = load_json(test_path)
    del test["info"]

    assert json.dumps(ref) == json.dumps(test)

    # Delete test file
    test_path.unlink()
