import json

from dsa.constants import ROOT
from dsa.evaluate_model import evaluate
from dsa.utils import load_json


def test_evaluate_model():
    ref_path = ROOT / "tests/data/report_chatgpt3.json"
    gt_path = ROOT / "tests/data/ground_truth.json"
    pred_path = ROOT / "tests/data/chatgpt3.json"

    ref = load_json(ref_path)
    res = evaluate(
        gt_json_path=gt_path,
        pred_json_path=pred_path,
        iou_thresholds=(0.5, 0.75),
        labels=["Figure", "Table"],
        output_path=None,
    )
    assert json.dumps(ref) == json.dumps(res)
