"""Verify a model directory against its lock file before serving it.

Reads a lock file (JSON: model_id, version, source, files with the
expected SHA-256 digest of each file) and a model directory, computes
the digest of every listed file, and reports matches, mismatches, files
missing from the directory, and files in the directory that the lock
does not list (which should not be loaded). Exit status 1 on any
mismatch or missing file. Standard library only.

Usage:
    python verify_model_files.py model_lock.json /path/to/model-dir

For the running example the directory does not ship with the cookbook;
the script is run with a directory created in the test, and the lock file
shows the format.

What this does not do: a digest proves the files are the ones the
organization recorded; it does not prove the recorded files are
trustworthy. The provenance of the import (publisher, date, licence) is
the lock file's source field and the model card, checked by a person.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("lock", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    with args.lock.open(encoding="utf-8") as fh:
        lock = json.load(fh)
    expected = lock["files"]
    present = {p.name for p in args.directory.iterdir() if p.is_file()}
    ok = mismatched = 0
    missing = []
    print(f"{lock['model_id']} {lock['version']} from {lock.get('source', '?')}\n")
    for name, sha in expected.items():
        path = args.directory / name
        if not path.exists():
            missing.append(name)
            print(f"  MISSING  {name}")
            continue
        actual = digest(path)
        if actual == sha:
            ok += 1
            print(f"  ok       {name}")
        else:
            mismatched += 1
            print(f"  MISMATCH {name}: expected {sha[:12]}..., found {actual[:12]}...")
    extra = sorted(present - set(expected))
    for name in extra:
        print(f"  unlisted {name} (not loaded)")
    print(
        f"\n{ok} verified, {mismatched} mismatched, {len(missing)} missing, {len(extra)} unlisted"
    )
    return 1 if mismatched or missing else 0


if __name__ == "__main__":
    sys.exit(main())
