"""Regenerate assets/recorded_responses.json from live model outputs.

Runs every notebook with a key configured and WORKSHOP_RECORD pointing at a
fresh file; the model client appends each live output under its scenario.
One-step sequences are then collapsed into single items.

    WORKSHOP_API_KEY=... [WORKSHOP_BASE_URL=... WORKSHOP_MODEL=...] \
        python record_responses.py

Review the result before committing it: the recorded outputs are what
participants read when they run the notebooks without a key.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
TARGET = ASSETS / "recorded_responses.json"
TMP = ASSETS / "recorded_responses.live.json"


def main() -> int:
    if not os.environ.get("WORKSHOP_API_KEY"):
        print("WORKSHOP_API_KEY is not set; nothing to record.")
        return 1
    if TMP.exists():
        TMP.unlink()
    os.environ["WORKSHOP_RECORD"] = str(TMP)
    for nb_path in sorted(HERE.glob("0*.ipynb")):
        print("running", nb_path.name)
        nb = nbformat.read(nb_path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name=os.environ.get("WORKSHOP_KERNEL", "python3"), resources={"metadata": {"path": str(HERE)}}).execute()
    store = json.loads(TMP.read_text())
    collapsed = {k: (v[0] if len(v) == 1 else v) for k, v in store.items()}
    TARGET.write_text(json.dumps(collapsed, indent=2, ensure_ascii=False))
    TMP.unlink()
    print(f"wrote {len(collapsed)} scenarios to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
