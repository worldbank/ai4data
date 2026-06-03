---
name: create-adapter
description: Develop a layout detection adapter that converts raw model outputs into the Unified Evaluation Schema v1.3 format.
---

# Create Adapter Skill

This skill provides the complete reference for building a **prediction adapter** that integrates a layout detection tool/library into the Data Snapshot Evaluation project.

---

## 1. What is an Adapter?

An adapter module converts raw model outputs into a JSON file conforming to the **Unified Evaluation Schema v1.3** (`docs/data-snapshot-eval-v1.3.schema.json`).

Adapters live in `src/dsa/adapters/` and follow a consistent architecture described below.

---

## 2. Schema Contract (v1.3)

The output JSON **must** have these top-level keys:

```json
{
  "label_map": { "1": "Figure", "2": "Table" },
  "info": {
    "schema_version": "1.3",
    "type": "prediction",
    "created_at": "<ISO8601>",
    "run_id": "<unique-run-id>",
    "model": {
      "name": "<model-name>",
      "version": "<version>",
      "notes": "<free-text>"
    },
    "coordinate_system": {
      "type": "normalized_xyxy",
      "range": [0.0, 1.0],
      "origin": "top_left"
    }
  },
  "documents": [
    { "doc_id": "<filename>", "doc_name": "<filename>", "doc_path": "<path>" }
  ],
  "predictions": [
    {
      "page_id": "<doc_id>::p<NNN>",
      "doc_id": "<doc_id>",
      "page_index": 0,
      "objects": [
        {
          "id": "<page_id>:<obj_index>",
          "label": "Figure",
          "bbox": [0.1, 0.2, 0.5, 0.6],
          "score": 0.95
        }
      ]
    }
  ]
}
```

### Key rules

| Rule | Detail |
|------|--------|
| Bounding boxes | Normalized `[x1, y1, x2, y2]` in `[0, 1]`, `top_left` origin |
| Labels | Only `"Figure"` and `"Table"` are canonical |
| Score | **Required** for every prediction object, clamped to `[0, 1]` |
| `page_id` format | `"{doc_id}::p{page_index:03d}"` |
| Object `id` format | `"{page_id}:{object_index}"` |
| `doc_id` | Use the PDF filename (e.g. `"report.pdf"`) |
| Pages with no objects | **Skip** — do not add empty page entries |

---

## 3. Adapter Architecture

Every adapter module follows this structure. Use the existing adapters as reference:

- `doclayoutyolo.py` — YOLO-based object detector
- `tfid.py` — HuggingFace generative model

### 3.1 Module Structure (for prediction adapters)

```
src/dsa/adapters/<adapter_name>.py
```

```python
"""
<Model Name> adapter -> Unified Evaluation Schema v1.3.

- Input: directory of PDFs
- Output: single JSON file matching data-snapshot-eval-v1.3.schema.json
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from dsa.constants import LABEL_MAP, ROOT
from dsa.utils import convert_pdf_to_images, normalize_bboxes_xyxy, utc_now_iso

# ── Constants ──────────────────────────────────────────────────────────
MODEL_NAME = "<model-identifier>"
INPUT_PDF_DIR = ROOT / "pdf_input"
OUTPUT_JSON_PATH = ROOT / "data/evaluation_input/<adapter-name>.json"

# Raw model labels -> canonical labels.
# All unmapped labels are silently ignored.
_LABEL_NORMALIZATION: dict[str, str] = {
    "figure": "Figure",
    # ... add model-specific variants
    "table": "Table",
}
_ALLOWED_LABELS = set(LABEL_MAP.values())


# ── Label Coercion ─────────────────────────────────────────────────────
def _coerce_label(raw: Any) -> str | None:
    """Map a raw model label to a canonical label.

    Parameters
    ----------
    raw : Any
        Raw label produced by the model.

    Returns
    -------
    str | None
        ``"Figure"`` or ``"Table"`` if recognized, ``None`` otherwise.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in _ALLOWED_LABELS:
        return s
    s_low = s.lower()
    if s_low in _LABEL_NORMALIZATION:
        return _LABEL_NORMALIZATION[s_low]
    return None


# ── Config ─────────────────────────────────────────────────────────────
class <AdapterName>Config:
    """Configuration for the <Model Name> adapter.

    Parameters
    ----------
    <param> : <type>
        <description>
    ...
    """

    def __init__(
        self,
        # ... adapter-specific parameters
        device: str = "cpu",
        dpi: int = 300,
        store_doc_path_as: str = "relative",
        pdf_backend: str = "pymupdf",
    ) -> None:
        self.device = device
        self.dpi = dpi
        self.store_doc_path_as = store_doc_path_as
        self.pdf_backend = pdf_backend


# ── Main adapter function ─────────────────────────────────────────────
def run_<adapter_name>_adapter_directory(
    input_pdf_dir: str | Path,
    output_json_path: str | Path,
    *,
    run_id: str | None = None,
    config: <AdapterName>Config | None = None,
) -> Path:
    """Run <Model Name> on every PDF in a directory and write predictions.

    Parameters
    ----------
    input_pdf_dir : str | Path
        Directory containing PDF files (searched recursively).
    output_json_path : str | Path
        Destination path for the prediction JSON.
    run_id : str | None
        Optional identifier for this evaluation run.
    config : <AdapterName>Config | None
        Adapter configuration. Uses defaults when ``None``.

    Returns
    -------
    Path
        The path where the prediction JSON was written.

    Raises
    ------
    FileNotFoundError
        If no PDF files are found under *input_pdf_dir*.
    """
    input_pdf_dir = Path(input_pdf_dir)
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = config or <AdapterName>Config()
    # TODO: Initialize the model here

    pdf_files = sorted(input_pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found under: {input_pdf_dir}")

    run_id = run_id or f"<adapter-name>-{uuid.uuid4().hex[:10]}"

    documents: list[dict[str, str]] = []
    predictions: list[dict[str, Any]] = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        doc_id = pdf_path.name

        if cfg.store_doc_path_as == "absolute":
            doc_path = str(pdf_path.resolve())
        else:
            doc_path = str(pdf_path.resolve().relative_to(ROOT))

        documents.append(
            {
                "doc_id": doc_id,
                "doc_name": pdf_path.name,
                "doc_path": doc_path,
            }
        )

        images = convert_pdf_to_images(pdf_path, dpi=cfg.dpi, backend=cfg.pdf_backend)

        for page_index, image in enumerate(
            tqdm(images, desc=f"Pages: {pdf_path.name}", leave=False)
        ):
            # TODO: Run model inference on `image`
            # TODO: Extract bboxes, labels, scores from model output

            bboxes_norm = normalize_bboxes_xyxy(
                boxes_raw, width=image.width, height=image.height
            )

            page_id = f"{doc_id}::p{page_index:03d}"
            objects: list[dict[str, Any]] = []

            for i, bbox in enumerate(bboxes_norm):
                label = _coerce_label(raw_labels[i])
                if label is None:
                    continue

                score = float(raw_scores[i])
                score = max(0.0, min(1.0, score))

                objects.append(
                    {
                        "id": f"{page_id}:{i}",
                        "label": label,
                        "bbox": bbox,
                        "score": score,
                    }
                )

            if not objects:
                continue

            predictions.append(
                {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "page_index": page_index,
                    "objects": objects,
                }
            )

    out = {
        "label_map": LABEL_MAP,
        "info": {
            "schema_version": "1.3",
            "type": "prediction",
            "created_at": utc_now_iso(),
            "run_id": run_id,
            "model": {
                "name": MODEL_NAME,
                "version": "<version>",
                "notes": f"adapter=<adapter_name>; device={cfg.device}; dpi={cfg.dpi}",
            },
            "coordinate_system": {
                "type": "normalized_xyxy",
                "range": [0.0, 1.0],
                "origin": "top_left",
            },
        },
        "documents": documents,
        "predictions": predictions,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)

    return output_json_path


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run <Model Name> over a PDF directory and produce a v1.3 prediction JSON."
    )
    parser.add_argument("--input_pdf_dir", type=str, default=str(INPUT_PDF_DIR))
    parser.add_argument("--output_json_path", type=str, default=str(OUTPUT_JSON_PATH))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--store_doc_path_as",
        type=str,
        choices=["relative", "absolute"],
        default="relative",
    )
    parser.add_argument(
        "--pdf_backend",
        type=str,
        choices=["pymupdf", "pdf2image"],
        default="pymupdf",
        help="PDF-to-image rendering backend (default: pymupdf).",
    )
    # TODO: Add model-specific CLI args
    args = parser.parse_args()

    pdf_dir = Path(args.input_pdf_dir)
    print(f"Input PDF dir : {pdf_dir}")
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    print(f"PDFs found    : {len(pdf_files)}")

    cfg = <AdapterName>Config(
        device=args.device,
        dpi=args.dpi,
        store_doc_path_as=args.store_doc_path_as,
        pdf_backend=args.pdf_backend,
    )

    out_path = run_<adapter_name>_adapter_directory(
        args.input_pdf_dir,
        args.output_json_path,
        run_id=args.run_id,
        config=cfg,
    )
    print(f"Wrote: {out_path}")
```

### 3.2 Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Module filename | `snake_case` (no hyphens) | `doclayoutyolo.py` |
| Config class | `PascalCase` + `Config` | `DocLayoutYOLOConfig` |
| Main function | `run_<name>_adapter_directory` | `run_doclayout_yolo_adapter_directory` |
| Run ID prefix | lowercase with hyphens | `"doclayout-yolo-"` |

---

## 4. Shared Utilities

Always import from `dsa.utils` and `dsa.constants` — **do not duplicate**.

### From `dsa.constants`

| Symbol | Purpose |
|--------|---------|
| `ROOT` | Project root directory (`Path`) |
| `LABEL_MAP` | Canonical label map `{"1": "Figure", "2": "Table"}` |

### From `dsa.utils`

| Function | Purpose |
|----------|---------|
| `convert_pdf_to_images(pdf_path, dpi, backend)` | Convert PDF pages to `list[PIL.Image.Image]`. Supports `"pymupdf"` (default) and `"pdf2image"` backends. |
| `normalize_bboxes_xyxy(bboxes, width, height)` | Convert absolute-pixel or already-normalized bboxes to `[0,1]` xyxy. Handles clamping, reordering, degenerate-box filtering. |
| `utc_now_iso()` | ISO 8601 UTC timestamp with `Z` suffix. |
| `clamp01(x)` | Clamp a float to `[0, 1]`. |
| `sanitize_bbox(b)` | Validate and clamp a 4-element bbox list. |
| `load_json(path)` | Load a JSON file as a dict. |

---

## 5. Integration Checklist

After implementing the adapter module:

### 5.1 `pyproject.toml`

Add a new optional-dependency group under `[project.optional-dependencies]`:

```toml
<adapter_name> = [
    "<pip-package-name>",
    # ... other deps
]
```

Also add it to the `dev` extras list:

```toml
dev = [
    "data-snapshot-annotation[tfid,doclayout_yolo,viz,<adapter_name>]",
    ...
]
```

### 5.2 Test Stub

Add a skipped test function in `tests/test_adapters.py`:

```python
@pytest.mark.skip(reason="For debugging purposes only.")
def test_<adapter_name>():
    ref_path = ROOT / "tests/data/<adapter-name>.json"
    test_path = ROOT / "tests/data/<adapter-name>_test.json"

    cfg = <AdapterName>Config()
    run_<adapter_name>_adapter_directory(
        ROOT / "pdf_input",
        test_path,
        config=cfg,
    )

    ref = load_json(ref_path)
    del ref["info"]
    test = load_json(test_path)
    del test["info"]

    assert json.dumps(ref) == json.dumps(test)

    test_path.unlink()
```

### 5.3 Imports

Add the necessary imports at the top of `tests/test_adapters.py`:

```python
from dsa.adapters.<adapter_name> import (
    <AdapterName>Config,
    run_<adapter_name>_adapter_directory,
)
```

---

## 6. Coding Standards

| Standard | Rule |
|----------|------|
| Formatter | **Black** (88 char line length) |
| Docstrings | **NumPy** style on all public modules, classes, and functions |
| Type hints | Modern Python 3.10+ (`str \| None`, `list[str]`) |
| Naming | PEP 8 (`snake_case` functions/variables, `PascalCase` classes) |
| Constants | Project-wide constants go in `src/dsa/constants.py` |

---

## 7. Model-Specific Considerations

When integrating a new model, pay attention to:

1. **Coordinate format**: The model may output absolute pixels, percentage-based, or already-normalized coordinates. Use `normalize_bboxes_xyxy()` for conversion.
2. **Label vocabulary**: Map every model-specific label to `"Figure"` or `"Table"` via `_LABEL_NORMALIZATION`. Labels that don't map to either are silently skipped.
3. **Score availability**: Some models don't output confidence scores. In that case, default to `1.0` for all detections.
4. **Batch vs. single inference**: Some models support batch processing. Adapt the page loop accordingly but maintain the same output structure.
5. **Dependencies**: Keep model-specific dependencies in the optional-dependency group, not in the base `dependencies`.

---

## 8. Reference Files

| File | Purpose |
|------|---------|
| [data-snapshot-eval-v1.3.schema.json](file:///home/ajd/data-snapshot-annotation/docs/data-snapshot-eval-v1.3.schema.json) | JSON Schema definition |
| [evaluation_spec.md](file:///home/ajd/data-snapshot-annotation/docs/evaluation_spec.md) | Evaluation contract |
| [project_context.md](file:///home/ajd/data-snapshot-annotation/docs/project_context.md) | Project background |
| [constants.py](file:///home/ajd/data-snapshot-annotation/src/dsa/constants.py) | Shared constants |
| [utils.py](file:///home/ajd/data-snapshot-annotation/src/dsa/utils.py) | Shared utilities |
| [doclayoutyolo.py](file:///home/ajd/data-snapshot-annotation/src/dsa/adapters/doclayoutyolo.py) | Reference adapter (YOLO) |
| [tfid.py](file:///home/ajd/data-snapshot-annotation/src/dsa/adapters/tfid.py) | Reference adapter (TF-ID) |
| [pyproject.toml](file:///home/ajd/data-snapshot-annotation/pyproject.toml) | Project dependencies |
| [test_adapters.py](file:///home/ajd/data-snapshot-annotation/tests/test_adapters.py) | Adapter test stubs |
