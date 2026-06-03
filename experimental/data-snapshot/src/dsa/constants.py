"""
Project-wide constants and path configurations.

This module provides a centralized location for directory paths, file paths,
evaluation thresholds, and other settings used throughout the project.
"""

from pathlib import Path

# Directories
ROOT = Path(__file__).parent.parent.parent
INPUT_PDF_DIR = ROOT / "pdf_input"
MODELS_DIR = ROOT / "models"

# Canonical label map (Unified Evaluation Schema v1.3)
LABEL_MAP: dict[str, str] = {
    "1": "Figure",
    "2": "Table",
}

# evaluate_model.py
GT_JSON_PATH = ROOT / "data/evaluation_input/ground_truth.json"
PRED_JSON_PATH = ROOT / "data/evaluation_input/DocLayout-YOLO-DocStructBench.json"
OUTPUT_REPORT_PATH = ROOT / "data/evaluation_output/DocLayout-YOLO-DocStructBench.json"
IOU_THRESHOLDS = [0.5, 0.75]
LABELS_TO_CONSIDER = ["Figure", "Table"]

# visualize_pages.py
VP_GT_JSON_PATH = ROOT / "data/evaluation_input/ground_truth.json"
VP_PRED_JSON_PATH = ROOT / "data/evaluation_input/DocLayout-YOLO-DocStructBench.json"
GT_COLOR_BGR = (64, 150, 27)  # green
PRED_COLOR_BGR = (64, 64, 255)  # red
VP_OUTPUT_DIR = ROOT / "data/visualize_pages/"

# Adapters
MIN_PREDICTION_AREA = 0.008
