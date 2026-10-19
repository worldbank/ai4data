"""Write a Croissant record for a CSV dataset from its dictionary.

Reads a data file (CSV) and its dictionary (name, label, type,
description, role) and writes a Croissant 1.0 JSON-LD record: the
dataset's name, description, licence, URL, version, and citation; the
file as a FileObject with its SHA-256; and one RecordSet with a Field per
column, typed from the dictionary. With --splits, adds the split
assignment file as a second FileObject and a RecordSet that joins it to
the records. Standard library only.

Usage:
    python build_croissant.py lfs_occupation_ml.csv dictionary.csv \
        --name "LFS occupation coding (example)" \
        --description "Job descriptions with ISCO-08 codes from the labour force survey." \
        --url https://stats.example/datasets/lfs-occupation-ml \
        --license https://creativecommons.org/licenses/by/4.0/ \
        --version 1.0.0 --date-published 2026-10-09 --cite-as "National Statistics Office (example). 2026. ..." \
        --splits splits.csv -o lfs_occupation_ml.croissant.json

What this does not do: it writes the structural layers (metadata, files,
fields, splits). It does not validate the record against the full
specification (mlcroissant does), and the description, licence, and
citation are only as right as the arguments given.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}

TYPES = {
    "numeric": "sc:Float",
    "integer": "sc:Integer",
    "categorical": "sc:Text",
    "text": "sc:Text",
    "id": "sc:Text",
    "boolean": "sc:Boolean",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_object(path: Path, ident: str, description: str) -> dict:
    return {
        "@type": "cr:FileObject",
        "@id": ident,
        "name": path.name,
        "description": description,
        "contentUrl": path.name,
        "encodingFormat": "text/csv",
        "sha256": sha256(path),
    }


def field(
    record_set: str, name: str, description: str, data_type: str, file_id: str
) -> dict:
    return {
        "@type": "cr:Field",
        "@id": f"{record_set}/{name}",
        "name": f"{record_set}/{name}",
        "description": description,
        "dataType": data_type,
        "source": {"fileObject": {"@id": file_id}, "extract": {"column": name}},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data")
    parser.add_argument("dictionary")
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--license", required=True, help="a URL")
    parser.add_argument("--version", required=True)
    parser.add_argument("--cite-as", required=True)
    parser.add_argument("--creator", default="National Statistics Office (example)")
    parser.add_argument(
        "--date-published", required=True, help="ISO date of the release"
    )
    parser.add_argument(
        "--splits", help="CSV with the record identifier and a split column"
    )
    parser.add_argument("-o", "--output", default="dataset.croissant.json")
    args = parser.parse_args(argv)

    data = Path(args.data)
    with open(args.dictionary, newline="", encoding="utf-8") as f:
        dictionary = list(csv.DictReader(f))
    with open(data, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    names = {d["name"] for d in dictionary}
    missing = [c for c in header if c not in names]
    if missing:
        print(
            f"columns without a dictionary row: {', '.join(missing)}", file=sys.stderr
        )
        return 1

    records = "records"
    fields = []
    for d in dictionary:
        desc = d["description"] + (f" Role: {d['role']}." if d.get("role") else "")
        fields.append(
            field(
                records, d["name"], desc, TYPES.get(d["type"], "sc:Text"), "data-file"
            )
        )
    record_sets = [
        {
            "@type": "cr:RecordSet",
            "@id": records,
            "name": records,
            "description": f"One record per row of {data.name}.",
            "key": {"@id": f"{records}/{header[0]}"},
            "field": fields,
        }
    ]
    distribution = [file_object(data, "data-file", "The data, one record per row.")]

    if args.splits:
        splits = Path(args.splits)
        with open(splits, newline="", encoding="utf-8") as f:
            split_names = sorted({r["split"] for r in csv.DictReader(f)})
        distribution.append(
            file_object(
                splits, "splits-file", "The split assignment, one row per record."
            )
        )
        record_sets.append(
            {
                "@type": "cr:RecordSet",
                "@id": "split_names",
                "name": "split_names",
                "description": "The named splits of the dataset.",
                "dataType": "cr:Split",
                "key": {"@id": "split_names/name"},
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "split_names/name",
                        "name": "name",
                        "description": "The split name.",
                        "dataType": "sc:Text",
                    }
                ],
                "data": [{"split_names/name": s} for s in split_names],
            }
        )
        record_sets.append(
            {
                "@type": "cr:RecordSet",
                "@id": "record_splits",
                "name": "record_splits",
                "description": "The split of each record, joined on the record identifier.",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "record_splits/record",
                        "name": "record_splits/record",
                        "description": "The record identifier.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "splits-file"},
                            "extract": {"column": header[0]},
                        },
                        "references": {"field": {"@id": f"{records}/{header[0]}"}},
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "record_splits/split",
                        "name": "record_splits/split",
                        "description": "The split name.",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "splits-file"},
                            "extract": {"column": "split"},
                        },
                        "references": {"field": {"@id": "split_names/name"}},
                    },
                ],
            }
        )

    record = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": args.name,
        "description": args.description,
        "url": args.url,
        "license": args.license,
        "version": args.version,
        "datePublished": args.date_published,
        "citeAs": args.cite_as,
        "creator": {"@type": "Organization", "name": args.creator},
        "distribution": distribution,
        "recordSet": record_sets,
    }
    Path(args.output).write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{Path(args.output).name}: {len(distribution)} file(s), {len(record_sets)} record set(s), {len(fields)} field(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
