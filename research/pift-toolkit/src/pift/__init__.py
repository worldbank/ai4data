"""
pift: a reusable toolkit for permutation-invariant fine-tuning of embedding
models for structured-metadata retrieval.

Adapt it to a catalogue by writing one YAML config (see ``configs/example.yaml``)
and running the stages: generate -> mine -> finetune -> evaluate -> search.

Public API:
    from pift import load_config, serialize, build_segments, render_segments
"""

from .config import Config, FieldSpec, load_config
from .serialize import serialize, build_segments, render_segments

__all__ = [
    "Config", "FieldSpec", "load_config",
    "serialize", "build_segments", "render_segments",
]

__version__ = "0.1.0"
