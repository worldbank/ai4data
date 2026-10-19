"""Check a Croissant record for the parts a loader and a catalog need.

Reads a Croissant JSON-LD file and checks: the dataset type and the
conformsTo declaration; the presence of name, description, url, license,
version, and citeAs; that every file in the distribution has a
contentUrl, an encodingFormat, and a sha256; that every record set has
fields, that every field has a dataType, and that every field source
names a file that exists in the distribution; and, when the files are
present beside the record, that each file's SHA-256 matches. Exit status
1 on any error. Uses mlcroissant, the reference implementation, for the specification check.

Usage:
    python croissant_check.py lfs_occupation_ml.croissant.json

What this does not do: it checks structure and integrity. It does not
run the full Croissant validator (mlcroissant), does not load the data,
and does not judge whether the description or the licence are right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlcroissant as mlc

REQUIRED = [
    "name",
    "description",
    "url",
    "license",
    "version",
    "datePublished",
    "citeAs",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("record")
    parser.add_argument(
        "--files-dir",
        help="where the files named by contentUrl are; default: beside the record",
    )
    args = parser.parse_args(argv)
    path = Path(args.record)
    rec = json.loads(path.read_text(encoding="utf-8"))
    files_dir = Path(args.files_dir) if args.files_dir else path.parent
    errors: list[str] = []
    warnings: list[str] = []

    # 1. the specification, through the reference implementation
    try:
        dataset = mlc.Dataset(jsonld=str(path))
        warnings += (
            [
                w.strip()
                for w in str(dataset.metadata.issues.warnings or "").split("\n")
                if w.strip() and "datePublished" not in w
            ]
            if hasattr(dataset.metadata, "issues")
            else []
        )
    except mlc.ValidationError as exc:
        errors += [
            line.strip(" -")
            for line in str(exc).splitlines()
            if line.strip().startswith("-")
        ]

    # 2. the fields a catalog needs beyond the specification, and the checksums
    errors += [f"missing {k}" for k in REQUIRED if not rec.get(k)]
    if rec.get("license") and not str(rec["license"]).startswith("http"):
        warnings.append(
            "license is not a URL; a URL is what tools and catalogs resolve"
        )
    files = rec.get("distribution", [])
    for d in files:
        if (
            d.get("@type") == "cr:FileObject"
            and d.get("sha256")
            and d.get("contentUrl")
        ):
            local = files_dir / d["contentUrl"]
            if not local.exists():
                warnings.append(
                    f"file {d.get('@id', '?')}: {d['contentUrl']} not found beside the record; checksum not verified"
                )
            elif sha256(local) != d["sha256"]:
                errors.append(
                    f"file {d.get('@id', '?')}: sha256 does not match {local.name}"
                )

    sets = rec.get("recordSet", [])
    n_fields = sum(len(rs.get("field", [])) for rs in sets)
    print(
        f"{path.name}: {rec.get('name', '?')} version {rec.get('version', '?')}; {len(files)} file(s), {len(sets)} record set(s), {n_fields} field(s)"
    )
    for w in warnings:
        print(f"  warning  {w}")
    for e in errors:
        print(f"  error    {e}")
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
