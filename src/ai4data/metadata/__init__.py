"""ai4data.metadata — Metadata quality, augmentation, and enrichment.

Install with: uv pip install ai4data[metadata]

Sub-modules
-----------
augmentation   : Data dictionary augmentation via LLM-powered theme generation.
reviewer       : Async AI-powered metadata reviewer (requires ai4data[metadata_reviewer]).
"""

from . import augmentation

try:
    from . import reviewer
    __all__ = ["augmentation", "reviewer"]
except ImportError:
    __all__ = ["augmentation"]
