"""ai4data.metadata — Metadata quality, augmentation, and enrichment.

Install with: uv pip install ai4data[metadata]

Sub-modules
-----------
augmentation   : Data dictionary augmentation via LLM-powered theme generation.
error_scanner  : Async AI-powered metadata error detection (requires ai4data[error_scanner]).
"""

from . import augmentation

try:
    from . import error_scanner
    __all__ = ["augmentation", "error_scanner"]
except ImportError:
    __all__ = ["augmentation"]
