import json
import pytest

from data_snapshot.constants import ROOT
from data_snapshot.utils import load_json
from data_snapshot.annotation.labelstudio_adapter import (
    convert_labelstudio_export_to_eval_v13,
)
from data_snapshot.inference.doclayoutyolo import (
    DocLayoutYOLOConfig,
    run_doclayout_yolo_adapter_directory,
)
from data_snapshot.inference.tfid import TFIDConfig, run_tfid_adapter_directory
from data_snapshot.inference.yolo26 import YOLO26Config, run_yolo26_adapter_directory
from data_snapshot.inference.yolo11 import YOLO11Config, run_yolo11_adapter_directory

INPUT_PDF_DIR = ROOT / "tests/data/pdf_input"


@pytest.mark.skip(reason="For debugging purposes only.")
def test_labelstudio():
    raw_path = ROOT / "tests/data/labelstudio_raw.json"
    ref_path = ROOT / "tests/data/labelstudio_ref.json"
    test_path = ROOT / "tests/data/labelstudio_test.json"

    convert_labelstudio_export_to_eval_v13(raw_path, test_path)

    ref = load_json(ref_path)
    del ref["info"]
    test = load_json(test_path)
    del test["info"]

    assert json.dumps(ref) == json.dumps(test)

    # Delete test file
    test_path.unlink()


@pytest.mark.skip(reason="For debugging purposes only.")
def test_doclayoutyolo():
    ref_path = ROOT / "tests/data/doclayoutyolo_ref.json"
    test_path = ROOT / "tests/data/doclayoutyolo_test.json"

    cfg = DocLayoutYOLOConfig()
    run_doclayout_yolo_adapter_directory(
        input_pdf_dir=INPUT_PDF_DIR,
        output_json_path=test_path,
        run_id=None,
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
    ref_path = ROOT / "tests/data/tfid_ref.json"
    test_path = ROOT / "tests/data/tfid_test.json"

    cfg = TFIDConfig()
    run_tfid_adapter_directory(
        INPUT_PDF_DIR,
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
    ref_path = ROOT / "tests/data/yolo26_ref.json"
    test_path = ROOT / "tests/data/yolo26_test.json"

    cfg = YOLO26Config()
    run_yolo26_adapter_directory(
        input_pdf_dir=INPUT_PDF_DIR,
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
    ref_path = ROOT / "tests/data/yolo11_ref.json"
    test_path = ROOT / "tests/data/yolo11_test.json"

    cfg = YOLO11Config()
    run_yolo11_adapter_directory(
        input_pdf_dir=INPUT_PDF_DIR,
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
