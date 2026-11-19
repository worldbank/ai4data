"""Two-stage batch pipeline (``ai4data.data_use.batch``) with fake models.

Stage 1 runs on real (generated) PDFs; stage 2 uses a fake GLiNER and fake LFM so
the tests need neither a GPU nor a model download.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("pymupdf4llm")
pytest.importorskip("pydantic")

from ai4data.data_use.batch import manifest as mf  # noqa: E402
from ai4data.data_use.batch.chunking import chunk_text  # noqa: E402
from ai4data.data_use.batch.cli import main as cli_main  # noqa: E402
from ai4data.data_use.batch.cli import status  # noqa: E402
from ai4data.data_use.batch.extract import Models, run_extract  # noqa: E402
from ai4data.data_use.batch.footnotes import WORD_RE  # noqa: E402
from ai4data.data_use.batch.markdown import md_to_pages, pages_to_md  # noqa: E402
from ai4data.data_use.batch.parse import PARSE_COMPLETE, read_manifest, run_parse  # noqa: E402
from ai4data.data_use.batch.schemas import EnrichmentRecord, ExtractionRecord  # noqa: E402

TEXT = (
    "We estimate poverty with the 2015 Household Survey collected by the statistics office. "
    "Results are cross-checked against household data from administrative records. "
)


def make_pdf(path: Path, pages=2, text=TEXT):
    doc = pymupdf.open()
    for i in range(pages):
        pg = doc.new_page()
        pg.insert_textbox(pymupdf.Rect(50, 50, 550, 780), f"Page {i}. {text * 3}", fontsize=11)
    doc.save(path)
    return path


@pytest.fixture
def pdf_dir(tmp_path):
    d = tmp_path / "pdfs"
    d.mkdir()
    make_pdf(d / "a.pdf", text=TEXT)
    make_pdf(d / "b.pdf", pages=3, text=TEXT + "Another sentence. ")
    return d


class FakeGliner:
    """Finds 'Household Survey' (named) and 'household data' (descriptive)."""

    def __init__(self):
        self.calls = 0

    def inference(self, texts, labels, flat_ner=True, threshold=0.5, batch_size=32):
        self.calls += 1
        assert list(labels) == ["NAMED_DATA", "DESCRIPTIVE_DATA", "VAGUE_DATA"]
        out = []
        for t in texts:
            ents = []
            for needle, label in (("Household Survey", "NAMED_DATA"), ("household data", "DESCRIPTIVE_DATA")):
                i = t.find(needle)
                while i != -1:
                    ents.append({"text": needle, "label": label, "score": 0.9, "start": i, "end": i + len(needle)})
                    i = t.find(needle, i + 1)
            out.append(ents)
        return out


@pytest.fixture
def fake_lfm(monkeypatch):
    import ai4data.data_use.models.metadata_model as mm

    calls = []

    def predict(tok, model, system, items, batch_size=16, **kw):
        calls.append((system[:20], len(items), batch_size))
        if system == mm.USAGE_SYSTEM:
            return [{"role": "primary", "usage_action": "analyze", "summary": "Used."}] * len(items)
        return [{"tags": ["poverty"], "attributes": {"year": "2015", "data_type": "survey"}}] * len(items)

    monkeypatch.setattr(mm, "predict_metadata", predict)
    return calls


def models():
    return Models(FakeGliner(), None, metadata=(None, None))


def extract(parsed, out, **kw):
    kw.setdefault("config", "c")
    kw.setdefault("poll_interval", 0.05)
    kw.setdefault("models", models())
    return run_extract(parsed, out, **kw)


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# ── markdown / chunking / manifest ─────────────────────────────────────────────


def test_markdown_roundtrip_keeps_real_page_numbers():
    pages = [{"page": 0, "text": "a"}, {"page": 2, "text": " b\n\nc "}, {"page": 5, "text": ""}, {"page": 7, "text": "d"}]
    assert md_to_pages(pages_to_md(pages)) == [
        {"page": 0, "text": "a"}, {"page": 2, "text": "b\n\nc"}, {"page": 7, "text": "d"}]


def test_markdown_reader_accepts_playground_format():
    assert md_to_pages("first\n\n<!-- page 1 -->\n\nsecond") == [
        {"page": 0, "text": "first"}, {"page": 1, "text": "second"}]


def test_word_chunks_never_exceed_the_gliner_window():
    text = " ".join(["**word**", "a,", "b;", "<br>"] * 400)
    chunks = chunk_text(text, None, max_tokens=384, overlap=50, unit="words")
    assert len(chunks) > 1
    for c in chunks:
        assert len(WORD_RE.findall(c["text"])) <= 384
        assert text[c["start"]:c["end"]] == c["text"]
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == len(text.rstrip())


def test_doc_ids_are_content_based_and_stable_when_files_are_added(pdf_dir):
    first = {r["pdf_url"]: r["corpus_id"] for r in mf.input_records(pdf_dir, "c")}
    make_pdf(pdf_dir / "0_new.pdf", text="Brand new text. ")
    second = {r["pdf_url"]: r["corpus_id"] for r in mf.input_records(pdf_dir, "c")}
    assert all(second[u] == i for u, i in first.items())
    assert len(second) == 3 and all(i.startswith("c:") and len(i) == 2 + 16 for i in second.values())


def test_identical_pdf_listed_twice_is_one_document(pdf_dir):
    (pdf_dir / "copy.pdf").write_bytes((pdf_dir / "a.pdf").read_bytes())
    assert len(mf.input_records(pdf_dir, "c")) == 2


def test_jsonl_manifest_with_title_and_metadata(pdf_dir, tmp_path):
    src = tmp_path / "m.jsonl"
    src.write_text(
        json.dumps({"path": str(pdf_dir / "a.pdf"), "title": "Paper A", "country": "KE"}) + "\n"
        + json.dumps({"pdf_url": "https://example.org/x/report.pdf"}) + "\n")
    recs = mf.input_records(src, "c")
    by = {r["title"]: r for r in recs}
    assert by["Paper A"]["metadata"] == {"country": "KE"} and by["Paper A"]["source_path"]
    assert by["report"]["source_path"] is None and by["report"]["pdf_url"].startswith("https://")


def test_checkpoint_last_row_wins_and_reconcile_removes_partial_rows(tmp_path):
    ck = tmp_path / "ck.jsonl"
    mf.append_checkpoint(ck, "x", "failed", "boom")
    mf.append_checkpoint(ck, "x", "done")
    mf.append_checkpoint(ck, "y", "failed", "bad")
    assert mf.load_done(ck) == {"x"}
    f = tmp_path / "out.jsonl"
    f.write_text('{"corpus_id": "x", "n": 1}\n{"corpus_id": "z", "n": 2}\n')
    assert mf.reconcile_outputs([f], {"x"}) == {"out.jsonl": 1}
    assert rows(f) == [{"corpus_id": "x", "n": 1}]


# ── stage 1 ────────────────────────────────────────────────────────────────────


def test_parse_writes_md_manifest_and_survives_a_corrupt_pdf(pdf_dir, tmp_path):
    (pdf_dir / "broken.pdf").write_bytes(b"this is not a pdf")
    parsed = tmp_path / "parsed"
    s = run_parse(pdf_dir, parsed, "c", workers=2)
    assert (s["done"], s["failed"]) == (2, 1)
    man = read_manifest(parsed, "c")
    assert len(man) == 2
    for r in man.values():
        assert Path(r["file"]).exists() and r["sha256"] and r["parser"]["pymupdf4llm"]
        assert r["pages"] == r["n_pdf_pages"] > 0
    st = mf.load_status(parsed / "parse_checkpoint.jsonl")
    assert [v["status"] for v in st.values()].count("failed") == 1
    assert (parsed / PARSE_COMPLETE).exists()
    ev = rows(parsed / "events.jsonl")
    assert {e["event"] for e in ev} >= {"start", "doc_done", "doc_failed", "finish"}
    assert len({e["run_id"] for e in ev}) == 1


def test_parse_resume_skips_done_and_retry_failed(pdf_dir, tmp_path):
    bad = pdf_dir / "broken.pdf"
    bad.write_bytes(b"nope")
    parsed = tmp_path / "parsed"
    run_parse(pdf_dir, parsed, "c", use_processes=False)
    again = run_parse(pdf_dir, parsed, "c", use_processes=False)
    assert (again["done"], again["skipped_done"], again["skipped_failed"]) == (0, 2, 1)
    make_pdf(bad, text="Now it is valid. ")  # same path, new bytes -> new id, parsed
    fixed = run_parse(pdf_dir, parsed, "c", use_processes=False)
    assert fixed["done"] == 1 and len(read_manifest(parsed, "c")) == 3


def test_scanned_pdf_without_text_fails_as_empty(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    doc = pymupdf.open()
    doc.new_page()
    doc.save(d / "blank.pdf")
    s = run_parse(d, tmp_path / "parsed", "c", use_processes=False)
    assert s["failed"] == 1
    err = next(iter(mf.load_status(tmp_path / "parsed" / "parse_checkpoint.jsonl").values()))["error"]
    assert "EMPTY" in err


def test_stage1_does_not_import_torch_or_models():
    code = ("import sys\nfrom ai4data.data_use.batch import parse, cli, manifest\n"
            "bad=[m for m in ('torch','transformers','gliner','gliner2') if m in sys.modules]\n"
            "assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ── stage 2 ────────────────────────────────────────────────────────────────────


@pytest.fixture
def parsed(pdf_dir, tmp_path):
    p = tmp_path / "parsed"
    run_parse(pdf_dir, p, "c", use_processes=False)
    return p


def test_extract_outputs_are_compatible_and_joinable(parsed, tmp_path, fake_lfm):
    out = tmp_path / "out"
    t = extract(parsed, out)
    assert t["processed"] == 2 and t["failed"] == 0
    man = read_manifest(parsed, "c")
    ext = rows(out / "extractions.jsonl")
    assert ext and all(ExtractionRecord.model_validate(r) for r in ext)  # strict, playground shape
    assert {r["corpus_id"] for r in ext} == set(man)
    for r in ext:
        for e in r["entities"]:
            assert r["input_text"][e["start"]:e["end"]] == e["text"]
    enr = rows(out / "enrichment.jsonl")
    assert enr and all(EnrichmentRecord.model_validate(r) for r in enr)
    by_chunk = {(r["corpus_id"], r["page"], r["chunk"]): r for r in ext}
    for e in enr:
        chunk = by_chunk[(e["corpus_id"], e["page"], e["chunk"])]
        assert any((x["start"], x["end"], x["label"]) == (e["start"], e["end"], "NAMED_DATA") for x in chunk["entities"])
        assert e["usage_action"] == "analyze" and e["theme_tags"] == ["poverty"]
        assert e["source_attributes"]["year"] == "2015"
    ds = rows(out / "datasets.jsonl")
    assert {d["corpus_id"] for d in ds} == set(man)
    merged = next(d for d in ds if d["datasets"])["datasets"]
    named = next(m for m in merged if m["dataset_name"] == "Household Survey")
    assert named["usage_action"] == "analyze" and named["theme_tags"] == ["poverty"]
    assert named["source_attributes"]["data_type"] == "survey" and named["count"] >= 2
    assert {r["corpus_id"] for r in rows(out / "checkpoint.jsonl")} == set(man)


def test_lfm_is_called_in_batches_once_per_task(parsed, tmp_path, fake_lfm):
    extract(parsed, tmp_path / "out", lfm_batch_size=7)
    assert len(fake_lfm) == 2 * 1  # usage + attributes, one batch call each (both docs in one batch)
    assert all(bs == 7 for _, _, bs in fake_lfm)
    assert fake_lfm[0][1] == fake_lfm[1][1] > 0


def test_no_enrich_skips_the_lfm(parsed, tmp_path, fake_lfm):
    out = tmp_path / "out"
    extract(parsed, out, enrich=False, models=Models(FakeGliner(), None, metadata=None))
    assert fake_lfm == [] and not (out / "enrichment.jsonl").read_text().strip()


def test_resume_does_nothing_when_all_done(parsed, tmp_path, fake_lfm):
    out = tmp_path / "out"
    extract(parsed, out)
    before = (out / "extractions.jsonl").read_text()
    t = extract(parsed, out)
    assert t["processed"] == 0 and (out / "extractions.jsonl").read_text() == before


def test_crash_between_rows_and_checkpoint_leaves_no_duplicates(parsed, tmp_path, fake_lfm):
    out = tmp_path / "out"
    extract(parsed, out)
    ck = (out / "checkpoint.jsonl").read_text().splitlines()
    n_ext = len(rows(out / "extractions.jsonl"))
    (out / "checkpoint.jsonl").write_text(ck[0] + "\n")  # as if the second doc's checkpoint was never written
    t = extract(parsed, out)
    assert t["processed"] == 1
    assert len(rows(out / "extractions.jsonl")) == n_ext
    ds = rows(out / "datasets.jsonl")
    assert len(ds) == 2 and len({d["corpus_id"] for d in ds}) == 2
    assert any(e["event"] == "reconcile" for e in rows(out / "events.jsonl"))


def test_resuming_with_other_settings_is_refused(parsed, tmp_path, fake_lfm):
    out = tmp_path / "out"
    extract(parsed, out)
    with pytest.raises(SystemExit, match="different settings"):
        extract(parsed, out, threshold=0.7)
    extract(parsed, out, threshold=0.7, allow_stamp_mismatch=True)


def test_run_json_records_models_and_every_run(parsed, tmp_path, fake_lfm):
    out = tmp_path / "out"
    extract(parsed, out)
    extract(parsed, out)
    meta = json.loads((out / "run.json").read_text())
    assert meta["stamp"]["entity_model_id"] == "rafmacalaba/gliner_agents_v3"
    assert meta["stamp"]["chunk_unit"] == "words" and meta["stamp"]["threshold"] == 0.5
    assert len(meta["runs"]) == 2 and meta["runs"][0]["run_id"] != meta["runs"][1]["run_id"]
    assert "torch" in meta["runs"][0]["libraries"]


def test_unreadable_md_fails_one_document_and_continues(parsed, tmp_path, fake_lfm):
    man = read_manifest(parsed, "c")
    victim = sorted(man)[0]
    Path(man[victim]["file"]).write_text("   \n")  # no pages with text
    out = tmp_path / "out"
    t = extract(parsed, out)
    assert (t["processed"], t["failed"]) == (1, 1)
    st = mf.load_status(out / "checkpoint.jsonl")
    assert st[victim]["status"] == "failed" and "no pages" in st[victim]["error"]
    assert victim not in {r["corpus_id"] for r in rows(out / "extractions.jsonl")}
    ev = [e for e in rows(out / "events.jsonl") if e["event"] == "doc_failed"]
    assert ev and ev[0]["corpus_id"] == victim


def test_limit_counts_pending_documents(parsed, tmp_path, fake_lfm):
    out = tmp_path / "out"
    assert extract(parsed, out, limit=1)["processed"] == 1
    assert extract(parsed, out, limit=1)["processed"] == 1
    assert extract(parsed, out, limit=1)["processed"] == 0


def test_poll_picks_up_documents_that_stage1_adds_later(pdf_dir, tmp_path, fake_lfm, monkeypatch):
    """Stage 2 starts with nothing to do, waits, and processes docs as a slow stage 1 produces them."""
    import ai4data.data_use.batch.parse as parse_mod

    real = parse_mod.parse_one

    def slow(*a, **kw):
        time.sleep(0.6)  # stage 1 is the bottleneck
        return real(*a, **kw)

    monkeypatch.setattr(parse_mod, "parse_one", slow)
    parsed, out = tmp_path / "parsed", tmp_path / "out"
    parsed.mkdir()
    th = threading.Thread(target=lambda: run_parse(pdf_dir, parsed, "c", use_processes=False))
    th.start()
    t = extract(parsed, out, idle_exit=30, stop_when=lambda: not th.is_alive(), batch_docs=1)
    th.join()
    assert t["processed"] == 2 and t["failed"] == 0
    assert len(rows(out / "datasets.jsonl")) == 2
    batches = [e for e in rows(out / "events.jsonl") if e["event"] == "batch"]
    assert len(batches) == 2  # one document per loop: it really waited for the second one


def test_extract_exits_after_idle_timeout_if_stage1_never_finishes(tmp_path, fake_lfm):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    t0 = time.monotonic()
    t = extract(parsed, tmp_path / "out", idle_exit=0.3)
    assert t["processed"] == 0 and time.monotonic() - t0 < 10


def test_no_resume_starts_over(parsed, tmp_path, fake_lfm):
    out = tmp_path / "out"
    extract(parsed, out)
    n = len(rows(out / "extractions.jsonl"))
    t = extract(parsed, out, no_resume=True)
    assert t["processed"] == 2 and len(rows(out / "extractions.jsonl")) == n


def test_status_and_cli(parsed, tmp_path, fake_lfm, capsys):
    out = tmp_path / "out"
    extract(parsed, out)
    info = status(parsed, out, "c")
    assert info["parse"]["done"] == 2 and info["extract"] == {"done": 2, "failed": 0, "waiting": 0}
    assert info["rows"]["datasets.jsonl"] == 2 and info["failures"] == {}
    assert cli_main(["status", "--parsed-dir", str(parsed), "--output-dir", str(out), "--config", "c"]) == 0
    assert json.loads(capsys.readouterr().out)["extract"]["done"] == 2


def test_cli_parse_then_dry_run_extract(pdf_dir, tmp_path):
    wd = tmp_path / "wd"
    assert cli_main(["parse", "--source", str(pdf_dir), "--work-dir", str(wd), "--no-processes"]) == 0
    assert cli_main(["extract", "--work-dir", str(wd), "--dry-run"]) == 0
