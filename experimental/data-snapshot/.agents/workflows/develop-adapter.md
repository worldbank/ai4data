---
description: Develop a layout detection adapter for a given tool/library
---

# Develop Adapter Workflow

Use this workflow when the user provides a **tool or library name** and asks you to develop an adapter for it.

## Steps

1. **Read the skill instructions**
   - Read `.agents/skills/create-adapter/SKILL.md` for architecture patterns, schema contract, and coding standards.

2. **Research the tool/library**
   - Search the web for the tool's documentation, installation instructions, and API reference.
   - Identify:
     - How to install it (`pip install <package>`)
     - How to load/initialize the model
     - How to run inference on a PIL image
     - What the raw output format looks like (bboxes, labels, scores)
     - What coordinate format it uses (absolute pixels, percentage, normalized)
     - What label vocabulary it uses
   - If the user provides a demo/example script, study it carefully.

3. **Plan the adapter**
   - Draft an `implementation_plan.md` covering:
     - Model initialization approach
     - Label normalization mapping (`_LABEL_NORMALIZATION`)
     - Coordinate conversion strategy
     - Score handling (native scores vs. default `1.0`)
     - Config class parameters (model-specific options)
     - Any additional pip dependencies
   - Request user approval via `notify_user` before proceeding.

4. **Implement the adapter module**
   - Create `src/dsa/adapters/<adapter_name>.py` following the template in the skill.
   - Key components:
     - Module docstring
     - Module-level constants (`MODEL_NAME`, `INPUT_PDF_DIR`, `OUTPUT_JSON_PATH`, `_LABEL_NORMALIZATION`)
     - `_coerce_label()` function
     - `<AdapterName>Config` class (plain `__init__`)
     - `run_<adapter_name>_adapter_directory()` function
     - `if __name__ == "__main__":` CLI block with `argparse`

5. **Update `pyproject.toml`**
   - Add a new optional-dependency group for the adapter's pip packages.
   - Add the new group to the `dev` extras.

6. **Add a test stub**
   - Add a `@pytest.mark.skip` test function in `tests/test_adapters.py`.
   - Add the necessary imports.

// turbo
7. **Run Black formatter**
   ```bash
   python -m black src/dsa/adapters/<adapter_name>.py
   ```

// turbo
8. **Verify import**
   ```bash
   python -c "from dsa.adapters.<adapter_name> import <AdapterName>Config; print('OK')"
   ```

// turbo
9. **Run tests**
   ```bash
   python -m pytest tests/ -x -q
   ```

10. **Notify the user**
    - Use `notify_user` to present the adapter for review.
    - Include the adapter module path in `PathsToReview`.
