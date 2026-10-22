"""Regenerate assets/recorded_responses.json from live model outputs.

Runs the notebooks with a key configured and WORKSHOP_RECORD pointing at a
fresh file; the model client appends each live output under its scenario.
One-step sequences are then collapsed into single items.

    WORKSHOP_API_KEY=... [WORKSHOP_BASE_URL=... WORKSHOP_MODEL=...] \
        python record_responses.py            # every notebook; the file is rewritten
    ... python record_responses.py 01 03      # only notebooks 01 and 03; their scenarios
                                              # replace the old ones, the rest are kept

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
    prefixes = tuple(sys.argv[1:])
    paths = [p for p in sorted(HERE.glob("0*.ipynb")) if not prefixes or p.name.startswith(prefixes)]
    if not paths:
        print("no notebook matches", prefixes)
        return 1
    for nb_path in paths:
        print("running", nb_path.name, flush=True)
        nb = nbformat.read(nb_path, as_version=4)
        NotebookClient(nb, timeout=600, kernel_name=os.environ.get("WORKSHOP_KERNEL", "python3"), resources={"metadata": {"path": str(HERE)}}).execute()
    store = json.loads(TMP.read_text())
    collapsed = {k: (v[0] if len(v) == 1 else v) for k, v in store.items()}
    if prefixes and TARGET.exists():
        merged = json.loads(TARGET.read_text())
        merged.update(collapsed)
        print(f"replaced {len(collapsed)} scenarios: {', '.join(sorted(collapsed))}")
        collapsed = merged
    TARGET.write_text(json.dumps(collapsed, indent=2, ensure_ascii=False) + "\n")
    TMP.unlink()
    print(f"wrote {len(collapsed)} scenarios to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
