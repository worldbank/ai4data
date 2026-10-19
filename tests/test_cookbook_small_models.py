"""Tests for the scripts shipped with the small-and-open-models cookbook."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "website" / "static" / "cookbook-files" / "small-and-open-models"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts():
    return {
        n: load_script(n)
        for n in [
            "task_fit",
            "model_card_check",
            "hardware_sizing",
            "cost_frontier",
            "verify_model_files",
        ]
    }


def test_task_fit(scripts, capsys):
    assert scripts["task_fit"].main([str(SCRIPTS / "tasks.csv")]) == 0
    out = capsys.readouterr().out
    assert "6 of 10 tasks start with a small open model" in out
    assert (
        "Occupation coding                small open model, local" in out
        and "confidential text stays on local open weights" in out
    )


def test_model_card_check(scripts, tmp_path, capsys):
    assert (
        scripts["model_card_check"].main([str(SCRIPTS / "model_card_example.json")])
        == 0
    )
    assert "result: PASS" in capsys.readouterr().out
    card = json.loads((SCRIPTS / "model_card_example.json").read_text(encoding="utf-8"))
    del card["license_url"]
    card["training_data"] = "Fine-tuned on survey responses."
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    assert scripts["model_card_check"].main([str(path)]) == 1
    out = capsys.readouterr().out
    assert (
        "MISSING license_url" in out
        and "does not say whether confidential or external data were used" in out
    )


def test_hardware_sizing(scripts, capsys):
    assert (
        scripts["hardware_sizing"].main(
            [
                "--params-b",
                "7",
                "--bits",
                "4",
                "--layers",
                "32",
                "--hidden",
                "4096",
                "--kv-heads-ratio",
                "0.25",
                "--context",
                "4096",
                "--concurrency",
                "8",
                "--gpu-gb",
                "24",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert (
        "weights 3.5 GB" in out
        and "total with 15% margin: 9.0 GB" in out
        and "GPU 24 GB: fits" in out
    )
    assert (
        scripts["hardware_sizing"].main(
            [
                "--params-b",
                "70",
                "--bits",
                "4",
                "--layers",
                "80",
                "--hidden",
                "8192",
                "--context",
                "4096",
                "--concurrency",
                "4",
                "--gpu-gb",
                "24",
            ]
        )
        == 0
    )
    assert "does not fit" in capsys.readouterr().out


def test_cost_frontier(scripts, capsys):
    assert (
        scripts["cost_frontier"].main(
            [
                str(SCRIPTS / "candidates.csv"),
                "--required",
                "0.90",
                "--required-national",
                "0.85",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "6 candidates; 4 on the cost-quality frontier" in out
    assert (
        "cheapest candidate meeting 0.90 overall and 0.85 national: mid-8b-q4 at 0.11"
        in out
    )
    assert (
        scripts["cost_frontier"].main(
            [str(SCRIPTS / "candidates.csv"), "--required", "0.99"]
        )
        == 1
    )


def test_verify_model_files(scripts, tmp_path, capsys):
    lock = json.loads((SCRIPTS / "model_lock.json").read_text(encoding="utf-8"))
    contents = {
        "config.json": b'{"architectures":["example"]}',
        "tokenizer.json": b'{"version":"1.0"}',
        "model.safetensors": b"weights",
    }
    for name, data in contents.items():
        (tmp_path / name).write_bytes(data)
    assert (
        scripts["verify_model_files"].main(
            [str(SCRIPTS / "model_lock.json"), str(tmp_path)]
        )
        == 0
    )
    assert "3 verified, 0 mismatched, 0 missing, 0 unlisted" in capsys.readouterr().out
    (tmp_path / "model.safetensors").write_bytes(b"tampered weights")
    (tmp_path / "extra.bin").write_bytes(b"x")
    assert (
        scripts["verify_model_files"].main(
            [str(SCRIPTS / "model_lock.json"), str(tmp_path)]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "MISMATCH model.safetensors" in out and "unlisted extra.bin" in out
    assert set(lock["files"]) == set(contents)
