"""Discovery of PDFs in a messy tree: folders, archives, duplicates, unsafe members."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("pymupdf4llm")

from ai4data.data_use.batch import manifest as mf  # noqa: E402
from ai4data.data_use.batch.cli import main as cli_main  # noqa: E402
from ai4data.data_use.batch.discover import Limits, classify, discover  # noqa: E402
from ai4data.data_use.batch.parse import read_manifest, run_parse  # noqa: E402

_GEN = Path(__file__).resolve().parents[1] / "scripts" / "data_use" / "make_messy_corpus.py"
_spec = importlib.util.spec_from_file_location("make_messy_corpus", _GEN)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


@pytest.fixture(scope="module")
def messy(tmp_path_factory):
    root = tmp_path_factory.mktemp("messy") / "corpus"
    gen.make(root)
    return root


def origins(inv):
    return {f.origin for f in inv.found}


def test_classify_is_case_insensitive_and_knows_archives():
    assert classify("A.PDF") == "pdf" and classify("x.Pdf") == "pdf"
    assert classify("a.zip") == "zip" and classify("a.tar.gz") == "tar" and classify("a.TGZ") == "tar"
    assert classify("a.pdf.gz") == "gz" and classify("a.7z") == "unsupported_archive"
    assert classify("a.txt") == "other"


def test_every_folder_level_and_case_is_crawled(messy, tmp_path):
    inv = discover(messy, tmp_path / "u")
    direct = {f.origin for f in inv.found if f.container is None}
    assert {"plain1.pdf", "UPPER.PDF", "Résumé données.pdf", "corrupt.pdf",
            "reports/2023/a.pdf", "reports/2023/b.pdf", "reports/2024/deep/er/c.pdf",
            "reports/2024/dup_of_a.pdf"} == direct


def test_hidden_and_system_files_are_never_found(messy, tmp_path):
    inv = discover(messy, tmp_path / "u")
    assert not any(("secret" in o) or ("~$" in o) or ("MACOSX" in o) or o.endswith("DS_Store") for o in origins(inv))
    assert inv.counts["hidden"] >= 3


def test_archives_are_opened_including_nested_and_single_gz(messy, tmp_path):
    inv = discover(messy, tmp_path / "u", max_archive_depth=2)
    o = origins(inv)
    assert {"bundle.zip!/z1.pdf", "bundle.zip!/folder/z2.pdf", "bundle.zip!/nested.zip!/n1.pdf",
            "batch.tar.gz!/t1.pdf", "batch.tar.gz!/sub/t2.pdf", "single.pdf.gz!/single.pdf",
            "evil.zip!/ok.pdf"} <= o
    n1 = next(f for f in inv.found if f.origin.endswith("n1.pdf"))
    assert n1.container == "bundle.zip" and n1.archive_depth == 2
    assert n1.pdf_url == (messy / "bundle.zip").as_uri() + "!/nested.zip!/n1.pdf"
    assert not any("deep.pdf" in x for x in o)  # three levels down: beyond the default limit


def test_archive_depth_is_configurable(messy, tmp_path):
    assert any("deep.pdf" in f.origin for f in discover(messy, tmp_path / "u", max_archive_depth=3).found)
    inv0 = discover(messy, tmp_path / "u0", max_archive_depth=0)
    assert not any(f.container for f in inv0.found)
    assert sum(p["kind"] == "archive_not_opened" for p in inv0.problems) >= 5


def test_archives_disabled_when_no_unpack_dir(messy):
    inv = discover(messy, None)
    assert not any(f.container for f in inv.found)
    assert all(p["kind"] in ("archive_not_opened", "unsupported_archive") for p in inv.problems)


def test_problems_are_reported_not_silent(messy, tmp_path):
    inv = discover(messy, tmp_path / "u")
    kinds = [p["kind"] for p in inv.problems]
    assert kinds.count("corrupt_archive") == 1 and "broken.zip" in next(
        p["path"] for p in inv.problems if p["kind"] == "corrupt_archive")
    assert kinds.count("unsafe_member") == 2
    assert kinds.count("unsupported_archive") == 1  # old.7z
    assert kinds.count("archive_not_opened") == 1  # l3.zip, too deep
    assert inv.ignored_ext["(none)"] == 0 and inv.ignored_ext[".txt"] >= 1 and inv.ignored_ext[".docx"] == 1
    assert "problems" in inv.report() and "PDFs found" in inv.report()


def test_zip_slip_members_never_escape_the_unpack_dir(messy, tmp_path):
    work = tmp_path / "work"
    unpack = work / "unpacked"
    discover(messy, unpack)
    assert not list(tmp_path.rglob("escape.pdf")) and not list(tmp_path.rglob("abs.pdf"))
    assert not (Path("/abs.pdf")).exists()
    for p in unpack.rglob("*"):
        assert unpack in p.parents or p == unpack


def test_archive_unpack_is_cached_and_reused(messy, tmp_path):
    u = tmp_path / "u"
    discover(messy, u)
    marks = {p: p.stat().st_mtime_ns for p in u.glob("*/.done")}
    assert marks
    inv = discover(messy, u)
    assert {p: p.stat().st_mtime_ns for p in u.glob("*/.done")} == marks
    assert len(inv.found) == 16


def test_corrupt_archive_is_not_cached_as_done(messy, tmp_path):
    u = tmp_path / "u"
    discover(messy, u)
    assert not any(".tmp" in d.name for d in u.iterdir())
    discover(messy, u)
    assert sum(p["kind"] == "corrupt_archive" for p in discover(messy, u).problems) == 1


def test_size_caps_stop_archive_bombs(messy, tmp_path):
    inv = discover(messy, tmp_path / "u", limits=Limits(max_member_bytes=500))
    assert any(p["kind"] == "corrupt_archive" and "larger than" in p["detail"] for p in inv.problems)
    inv = discover(messy, tmp_path / "u2", limits=Limits(max_members=1))
    assert any(p["kind"] == "corrupt_archive" and "members" in p["detail"] for p in inv.problems)


def test_symlinked_folders_are_not_followed(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    real = tmp_path / "elsewhere"
    real.mkdir()
    gen.pdf_bytes  # noqa: B018
    (real / "x.pdf").write_bytes(gen.pdf_bytes("x"))
    os.symlink(real, root / "link")
    (root / "loop").symlink_to(root)
    assert discover(root, tmp_path / "u").found == []


def test_duplicates_keep_the_copy_outside_archives_and_are_logged(messy, tmp_path):
    recs, inv = mf.scan_input(messy, "c", tmp_path / "u")
    assert len(recs) == 14 and len(inv.found) == 16 and len(inv.duplicates) == 2
    assert {d["kept"] for d in inv.duplicates} == {(messy / "reports/2023/a.pdf").as_uri()}
    assert {d["dropped"] for d in inv.duplicates} == {
        (messy / "reports/2024/dup_of_a.pdf").as_uri(), (messy / "bundle.zip").as_uri() + "!/a_copy.pdf"}


def test_records_carry_provenance_in_metadata(messy, tmp_path):
    recs = {r["title"]: r for r in mf.input_records(messy, "c", tmp_path / "u")}
    assert recs["a"]["metadata"] == {"origin": "reports/2023/a.pdf", "folder": "reports"}
    assert recs["n1"]["metadata"]["container"] == "bundle.zip" and recs["n1"]["metadata"]["archive_depth"] == 2
    assert recs["n1"]["source_path"].startswith(str(tmp_path / "u"))  # parsed from the unpacked copy


def test_hash_cache_avoids_rereading_pdfs(messy, tmp_path, monkeypatch):
    cache = {}
    mf.input_records(messy, "c", tmp_path / "u", hash_cache=cache)
    assert cache
    reads = []
    real = mf.sha256_file
    monkeypatch.setattr(mf, "sha256_file", lambda p, *a, **k: reads.append(p) or real(p, *a, **k))
    mf.input_records(messy, "c", tmp_path / "u", hash_cache=cache)
    assert reads == []


def test_parse_stage_on_the_messy_corpus(messy, tmp_path):
    parsed = tmp_path / "parsed"
    s = run_parse(messy, parsed, "c", workers=2)
    assert (s["done"], s["failed"]) == (13, 1)  # only corrupt.pdf fails, at parse
    inv = json.loads((parsed / "inventory.json").read_text())
    assert inv["summary"]["unique_pdfs"] == 14 and inv["summary"]["duplicates_dropped"] == 2
    man = read_manifest(parsed, "c")
    assert len(man) == 13 and all(Path(r["file"]).exists() for r in man.values())
    assert any("!/" in r["pdf_url"] for r in man.values())
    again = run_parse(messy, parsed, "c", workers=2)
    assert (again["done"], again["skipped_done"], again["skipped_failed"]) == (0, 13, 1)
    start = next(json.loads(l) for l in (parsed / "events.jsonl").read_text().splitlines() if '"start"' in l)
    assert start["inventory"]["pdfs_found"] == 16


def test_no_archives_flag_skips_archives(messy, tmp_path):
    s = run_parse(messy, tmp_path / "p", "c", workers=1, archives=False, use_processes=False)
    assert s["done"] + s["failed"] == 8 - 1  # direct PDFs minus the identical copy


def test_scan_command_prints_and_writes_the_inventory(messy, tmp_path, capsys):
    assert cli_main(["scan", "--source", str(messy), "--work-dir", str(tmp_path / "wd")]) == 0
    out = capsys.readouterr().out
    assert "PDFs found: 16" in out and "unique documents to process: 14" in out
    assert (tmp_path / "wd" / "parsed" / "inventory.json").exists()
    assert cli_main(["scan", "--source", str(messy), "--work-dir", str(tmp_path / "wd"), "--no-archives"]) == 0
