"""Check namespace imports and installed command entry points without models."""

import json
import subprocess
import sys
from importlib import import_module
from importlib.resources import files
from pathlib import Path

import pytest


def test_package_import_does_not_load_snapshot_dependencies(tmp_path: Path) -> None:
    """Keep base imports independent of snapshot and model runtimes."""
    script = """
import importlib.abc
import sys

class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'PIL', 'pymupdf', 'openai', 'dotenv', 'torch', 'transformers',
            'ultralytics', 'doclayout_yolo', 'huggingface_hub',
        }:
            raise AssertionError(f'Unexpected optional import: {fullname}')

sys.meta_path.insert(0, BlockOptionalImports())
from ai4data import data_snapshot
assert data_snapshot.discover_pdfs.__module__ == 'ai4data.data_snapshot.inputs'
try:
    data_snapshot.unknown
except AttributeError:
    pass
else:
    raise AssertionError('Unknown attribute was accepted')
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


@pytest.mark.parametrize(
    "module",
    [
        "ai4data.data_snapshot",
        "ai4data.data_snapshot.extract_data_snapshot",
        "ai4data.data_snapshot.extract_metadata",
    ],
)
def test_module_help_runs_outside_checkout(module: str, tmp_path: Path) -> None:
    """Exercise each module entry point without inference or API calls."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert module in result.stdout
    assert "--input" in result.stdout


def test_packaged_resources_are_available() -> None:
    """Read both provider presets and prompt files through the installed package."""
    from importlib.resources import files

    from ai4data.data_snapshot.metadata_extraction import load_extraction_config
    from ai4data.data_snapshot.metadata_extraction.extraction import (
        get_extraction_config_path,
    )

    for provider in ("openai", "openrouter"):
        config = load_extraction_config(get_extraction_config_path(provider))
        assert config["provider"] == provider
    resources = files("ai4data.data_snapshot.metadata_extraction")
    for prompt in ("system.md", "user.md"):
        assert resources.joinpath("prompts", prompt).read_text(encoding="utf-8")
    model_defaults = json.loads(
        files("ai4data.data_snapshot")
        .joinpath("config", "models.json")
        .read_text(encoding="utf-8")
    )
    assert set(model_defaults) == {"doclayoutyolo", "yolo11", "yolo26", "tfid"}
    assert all(
        isinstance(value, str) and value.strip()
        for settings in model_defaults.values()
        for value in settings.values()
    )


@pytest.mark.parametrize(
    ("module", "config_name"),
    [
        ("doclayoutyolo", "DocLayoutYOLOConfig"),
        ("yolo11", "YOLO11Config"),
        ("yolo26", "YOLO26Config"),
        ("tfid", "TFIDConfig"),
    ],
)
def test_model_configuration_environment_precedence(
    module: str, config_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve per-model defaults at construction time, below explicit settings."""
    adapter_module = import_module(f"ai4data.data_snapshot.adapters.{module}")
    config_type = getattr(adapter_module, config_name)
    prefix = f"AI4DATA_SNAPSHOT_{module.upper()}_"
    defaults = json.loads(
        files("ai4data.data_snapshot")
        .joinpath("config", "models.json")
        .read_text(encoding="utf-8")
    )[module]
    defaults["cache_dir"] = Path(defaults["cache_dir"])
    for other in ("DOCLAYOUTYOLO", "YOLO11", "YOLO26", "TFID"):
        for field in defaults:
            monkeypatch.setenv(f"AI4DATA_SNAPSHOT_{other}_{field.upper()}", "other")
    for field in defaults:
        monkeypatch.delenv(prefix + field.upper())
    config = config_type()
    assert {field: getattr(config, field) for field in defaults} == defaults

    for field in defaults:
        monkeypatch.setenv(prefix + field.upper(), "")
    assert config_type() == config

    from ai4data.data_snapshot import model_config

    alternate_defaults = {field: f"preset-{field}" for field in defaults}
    monkeypatch.setattr(
        model_config, "_load_model_defaults", lambda: {module: alternate_defaults}
    )
    config = config_type()
    assert {
        field: str(getattr(config, field)) for field in defaults
    } == alternate_defaults

    for value in ("first", "second"):
        for field in defaults:
            monkeypatch.setenv(prefix + field.upper(), f"{value}-{field}")
        config = config_type()
        assert {field: str(getattr(config, field)) for field in defaults} == {
            field: f"{value}-{field}" for field in defaults
        }

    explicit = {field: f"explicit-{field}" for field in defaults}
    config = config_type(**explicit)
    assert {field: getattr(config, field) for field in defaults} == explicit
