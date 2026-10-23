"""Resolve model settings from environment overrides and packaged JSON defaults."""

import json
import os
from functools import cache
from importlib.resources import files


@cache
def _load_model_defaults() -> dict[str, dict[str, str]]:
    """Read the bundled model defaults once per process."""
    resource = files("ai4data.data_snapshot").joinpath("config", "models.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def get_model_setting(model: str, setting: str) -> str:
    """Resolve one model setting, preferring a nonempty environment override.

    Parameters
    ----------
    model : str
        Extractor identifier used in the packaged configuration.
    setting : str
        Model location field, such as ``repo_id`` or ``cache_dir``.

    Returns
    -------
    str
        Environment value or the corresponding packaged JSON default.

    Raises
    ------
    KeyError
        If the model or setting has no packaged default and no override is set.
    """
    variable = f"AI4DATA_SNAPSHOT_{model.upper()}_{setting.upper()}"
    return os.getenv(variable) or _load_model_defaults()[model][setting]
