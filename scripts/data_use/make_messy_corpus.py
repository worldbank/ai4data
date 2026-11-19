#!/usr/bin/env python
"""Generate a deliberately messy PDF corpus to exercise ``ai4data.data_use.batch``.

    python scripts/data_use/make_messy_corpus.py /tmp/messy

Layout (every case the discovery has to handle):

    plain PDFs, an upper-case ``.PDF``, accents and spaces in names
    folders 3-4 levels deep, empty folders
    a byte-identical copy of a PDF in another folder and inside an archive
    non-PDF files (txt, csv, docx), hidden files, ``__MACOSX``, ``~$`` temp files
    .zip (with folders, a nested .zip, a macOS resource fork), .tar.gz, .pdf.gz, .csv.gz
    a corrupt .zip, a .zip with path-escaping members, a 7z, a corrupt PDF
    archives nested 3 deep (deeper than the default limit of 2)

Prints the expected counts so a run can be checked against them.
"""

import gzip
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pymupdf

TEXT = ("We estimate poverty with the 2015 Household Survey collected by the statistics office. "
        "Results are cross-checked against household data from administrative records. ")


def pdf_bytes(tag: str, pages: int = 2) -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        pg = doc.new_page()
        pg.insert_textbox(pymupdf.Rect(50, 50, 550, 780), f"{tag} page {i}. " + TEXT * 3, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def zip_bytes(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def make(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)

    def w(rel: str, data):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())

    a = pdf_bytes("A")
    # --- plain / folders ---
    w("plain1.pdf", pdf_bytes("plain1"))
    w("UPPER.PDF", pdf_bytes("upper"))
    w("Résumé données.pdf", pdf_bytes("accents"))
    w("reports/2023/a.pdf", a)
    w("reports/2023/b.pdf", pdf_bytes("B", 3))
    w("reports/2024/deep/er/c.pdf", pdf_bytes("C"))
    w("reports/2024/dup_of_a.pdf", a)  # identical bytes, other folder
    (root / "empty_dir").mkdir(exist_ok=True)
    w("corrupt.pdf", b"%PDF-1.4 this is not really a pdf")
    # --- ignored / hidden ---
    w("notes.txt", "hello")
    w("data.csv", "a,b\n1,2\n")
    w("report.docx", b"PK fake docx")
    w(".hidden/secret.pdf", pdf_bytes("hidden"))
    w(".DS_Store", b"\0")
    w("~$temp.pdf", b"lock file")
    w("old.7z", b"7z\xbc\xaf\x27\x1c")
    # --- archives ---
    nested = zip_bytes({"n1.pdf": pdf_bytes("N1")})
    w("bundle.zip", zip_bytes({
        "z1.pdf": pdf_bytes("Z1"),
        "folder/z2.pdf": pdf_bytes("Z2"),
        "__MACOSX/._z1.pdf": b"fork",
        "readme.txt": "x",
        "a_copy.pdf": a,  # identical to reports/2023/a.pdf
        "nested.zip": nested,
    }))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in (("t1.pdf", pdf_bytes("T1")), ("sub/t2.pdf", pdf_bytes("T2"))):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    w("batch.tar.gz", buf.getvalue())
    w("single.pdf.gz", gzip.compress(pdf_bytes("G1")))
    w("data.csv.gz", gzip.compress(b"a,b\n1,2\n"))
    w("broken.zip", b"PK\x03\x04 truncated garbage")
    w("evil.zip", zip_bytes({"../../escape.pdf": pdf_bytes("EVIL1"), "/abs.pdf": pdf_bytes("EVIL2"),
                             "ok.pdf": pdf_bytes("OK")}))
    l3 = zip_bytes({"deep.pdf": pdf_bytes("DEEP")})
    l2 = zip_bytes({"l3.zip": l3})
    w("l1.zip", zip_bytes({"l2.zip": l2}))

    return {
        # unique PDFs at the default archive depth (2): see the notes below
        "unique_pdfs": 14,
        "expected": [
            "plain1, UPPER, accents, a, b, c, corrupt (discovered; fails later in parse)",
            "bundle.zip: z1, z2, nested.zip!n1 (a_copy is a duplicate of a)",
            "batch.tar.gz: t1, t2   single.pdf.gz: g1   evil.zip: ok (2 unsafe members rejected)",
            "l1.zip!l2.zip!l3.zip: NOT opened at depth 2 -> reported, deep.pdf not found",
            "problems: corrupt broken.zip, 2 unsafe members, 7z unsupported, l3.zip too deep",
            "duplicates dropped: 2 (reports/2024/dup_of_a.pdf, bundle.zip!/a_copy.pdf)",
            "never found: .hidden/secret.pdf, ~$temp.pdf, __MACOSX/*",
        ],
    }


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "messy_corpus")
    info = make(out)
    print(f"wrote {out}\nexpected unique PDFs at --max-archive-depth 2: {info['unique_pdfs']}")
    print("\n".join("  - " + line for line in info["expected"]))
