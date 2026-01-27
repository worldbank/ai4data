"""ai4data - AI for Data package.

This package provides various AI-powered tools for data analysis:

- data_use: Dataset mention extraction from text and documents
  Install with: uv pip install ai4data[datause]

- anomaly_detection: Anomaly and outlier detection in data
  Install with: uv pip install ai4data[anomaly]

For all capabilities:
  uv pip install ai4data[all]
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ai4data")
except PackageNotFoundError:
    # package is not installed
    pass

# Make submodules available
# These use lazy imports internally, so they won't fail
# if dependencies are not installed until you actually use the features
from . import data_use
from . import anomaly_detection

__all__ = [
    "__version__",
    "data_use",
    "anomaly_detection",
]
