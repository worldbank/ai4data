"""ai4data - AI for Data package.

This package provides various AI-powered tools for data analysis:

- data_use: Dataset mention extraction from text and documents
  Install with: uv pip install ai4data[datause]

- anomaly: Anomaly detection and explanation in data
  Install with: uv pip install ai4data[anomaly]

- metadata: Metadata quality assessment and augmentation
  Install with: uv pip install ai4data[metadata]

- discovery: NADA catalog, metadata templates, and related discovery helpers
  Install with: uv pip install ai4data[discovery]

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
from . import anomaly
from . import discovery
from . import metadata

__all__ = [
    "__version__",
    "data_use",
    "anomaly",
    "discovery",
    "metadata",
]
