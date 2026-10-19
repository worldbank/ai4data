"""Write or verify the manifest of a dataset release.

In build mode, writes a manifest (JSON) with the version, the date, and,
for every file of the release, its size and SHA-256. In verify mode,
compares the files on disk with the manifest and reports files that
changed, files that are missing, and files present but not listed; exit
status 1 on any difference. Standard library only.

Usage:
    python release_manifest.py build --version 1.0.0 -o manifest.json \
        lfs_occupation_ml.csv splits.csv dictionary.csv dataset_card.md \
        lfs_occupation_ml.croissant.json
    python release_manifest.py verify manifest.json

What this does not do: a manifest proves that the files a user holds are
the files that were released. It does not say that the release is
correct, and it does not replace the version in the Croissant record and
the dataset card, which the chapter keeps in step with it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(args: argparse.Namespace) -> int:
    files = {}
    for name in args.files:
        p = Path(name)
        files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    manifest = {
        "version": args.version,
        "date": args.date or time.strftime("%Y-%m-%d", time.gmtime()),
        "files": files,
    }
    Path(args.output).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{Path(args.output).name}: version {manifest['version']}, {len(files)} file(s), {sum(f['bytes'] for f in files.values())} bytes"
    )
    return 0


def verify(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    base = Path(args.files_dir) if args.files_dir else path.parent
    changed, missing, ok = [], [], []
    for name, info in manifest["files"].items():
        p = base / name
        if not p.exists():
            missing.append(name)
        elif sha256(p) != info["sha256"]:
            changed.append(name)
        else:
            ok.append(name)
    print(
        f"{path.name}: version {manifest.get('version', '?')} of {manifest.get('date', '?')}; {len(manifest['files'])} file(s) listed"
    )
    for name in ok:
        print(f"  ok        {name}")
    for name in changed:
        print(f"  CHANGED   {name}")
    for name in missing:
        print(f"  MISSING   {name}")
    print(f"result: {'PASS' if not (changed or missing) else 'FAIL'}")
    return 1 if (changed or missing) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build")
    b.add_argument("files", nargs="+")
    b.add_argument("--version", required=True)
    b.add_argument("--date", help="ISO date; default today")
    b.add_argument("-o", "--output", default="manifest.json")
    v = sub.add_parser("verify")
    v.add_argument("manifest")
    v.add_argument("--files-dir")
    args = parser.parse_args(argv)
    return build(args) if args.mode == "build" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
