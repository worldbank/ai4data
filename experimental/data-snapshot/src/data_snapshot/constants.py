"""
Project-wide constants and path configurations.

This module provides a centralized location for directory paths,
evaluation thresholds, and other settings used throughout the package.
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

# Evaluation defaults
IOU_THRESHOLDS = [0.5, 0.75]
LABELS_TO_CONSIDER = ["Figure", "Table"]

# Visualization colors (BGR for OpenCV)
GT_COLOR_BGR = (64, 150, 27)  # green
PRED_COLOR_BGR = (64, 64, 255)  # red

# Adapters
MIN_PREDICTION_AREA = 0.008
