"""ai4data.metadata.augmentation — LLM-powered data dictionary augmentation.

Automatically generates DDI-style variable groups for microdata or administrative
data dictionary variables using semantic clustering and LLM-elicited curation.

Install
-------
.. code-block:: bash

    uv pip install ai4data[metadata]

Quick Start
-----------
.. code-block:: python

    from ai4data.metadata.augmentation import DataDictionaryAugmentor

    augmentor = DataDictionaryAugmentor()
    result = augmentor.augment("variables.csv")
    augmentor.export("augmented.json")

The pipeline: Load → Embed → Cluster → Generate Variable Groups → Export.

See :class:`DataDictionaryAugmentor` for the full API.
"""

from . import adapters, clustering, embeddings, prompts, qa, schemas
from .adapters import (
    ConfigurableDictionaryAdapter,
    NADACatalogAdapter,
    adapter_from_config,
)
from .augmentor import DEFAULT_MODEL, DataDictionaryAugmentor
from .schemas import (
    AugmentedDictionary,
    DictionaryVariable,
    VariableGroup,
    VariableGroupAssignment,
    VariableGroupCurationResult,
    VariableGroupQAResult,
    make_vgid,
)

__all__ = [
    # Main class
    "DataDictionaryAugmentor",
    "DEFAULT_MODEL",
    # Adapters
    "ConfigurableDictionaryAdapter",
    "NADACatalogAdapter",
    "adapter_from_config",
    # Schemas
    "AugmentedDictionary",
    "DictionaryVariable",
    "VariableGroup",
    "VariableGroupAssignment",
    "VariableGroupCurationResult",
    "VariableGroupQAResult",
    "make_vgid",
    # Submodules
    "adapters",
    "clustering",
    "embeddings",
    "prompts",
    "qa",
    "schemas",
]
