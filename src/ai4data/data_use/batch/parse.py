"""Stage 1: fetch + parse PDFs to page-delimited markdown (CPU only).

Workers are spawned processes (one PDF each) because pymupdf4llm is CPU-bound.
They import neither torch nor any model code.
"""

import concurrent.futures as cf
import importlib.metadata
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import manifest as mf
from .logging_utils import EventLog, Progress, fmt_secs, new_run_id, setup_logging, utcnow
from .markdown import md_to_pages, pages_to_md
from .schemas import ManifestRecord

log = logging.getLogger("ai4data.batch.parse")

PARSE_CHECKPOINT = "parse_checkpoint.jsonl"
PARSE_COMPLETE = ".parse_complete"


class ParseTimeout(Exception):
    pass


def library_versions() -> Dict[str, str]:
    out = {}
    for mod in ("pymupdf", "pymupdf4llm"):
        try:
            out[mod] = importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError:
            out[mod] = "not-installed"
    return out


def parse_pdf_to_pages(pdf_path: str, max_pages: Optional[int] = None):
    """Parse a PDF into ``[{"page": 0-based idx, "text": markdown}]`` and the page count.

    Pages whose markdown render is blank but that have extractable text fall
    back to plain ``get_text()`` (covers cover pages whose text sits inside
    vector-graphics boxes).
    """
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(pdf_path)
    try:
        n_pdf = len(doc)
        if max_pages and n_pdf > max_pages:
            raise ValueError(f"SKIP:pages={n_pdf}>{max_pages}")
        chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
        pages = []
        for i, c in enumerate(chunks):
            meta = c.get("metadata") or {}
            num = meta.get("page_number", meta.get("page"))
            idx = (num - 1) if num is not None else i
            text = c.get("text", "")
            if not text.strip() and 0 <= idx < n_pdf:
                plain = doc[idx].get_text()
                if plain.strip():
                    text = plain
            pages.append({"page": idx, "text": text})
        return pages, n_pdf
    finally:
        doc.close()


def _parse_with_timeout(pdf_path: str, timeout: Optional[int], max_pages: Optional[int]):
    """Per-document wall-clock budget (SIGALRM; POSIX main thread only)."""
    use_alarm = (
        timeout
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not use_alarm:
        return parse_pdf_to_pages(pdf_path, max_pages)

    def _raise(_sig, _frm):
        raise ParseTimeout(f"TIMEOUT:parse>{timeout}s")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(int(timeout))
    try:
        return parse_pdf_to_pages(pdf_path, max_pages)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def parse_one(rec: Dict[str, Any], parsed_dir: str, timeout: Optional[int], max_pages: Optional[int]):
    """Worker: fetch (if URL) + parse one document and write its ``.md``.

    Module level so it pickles under ``spawn``. Returns ``(rec, manifest_row_or_None,
    error_or_None, timings)``.
    """
    t_fetch = t_parse = 0.0
    tmp_path = None
    try:
        path = rec["source_path"]
        if path is None:
            from ..utils.document_parser import DocumentParser

            t0 = time.perf_counter()
            tmp_path = DocumentParser._download_temp_pdf(rec["pdf_url"])
            t_fetch = time.perf_counter() - t0
            path = tmp_path
        t0 = time.perf_counter()
        pages, n_pdf = _parse_with_timeout(path, timeout, max_pages)
        t_parse = time.perf_counter() - t0
        if rec["source_path"] is None:  # hash the downloaded bytes now that we have them
            rec = {**rec, "size_bytes": os.path.getsize(path), "sha256": mf.sha256_file(Path(path))}
        md = pages_to_md(pages)
        if not md.strip():
            raise ValueError("EMPTY:no extractable text (scanned PDF? OCR is not run)")
        md_path = Path(parsed_dir) / rec["config"] / mf.doc_filename(rec["corpus_id"])
        md_path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(md_path, md)
        row = ManifestRecord(
            corpus_id=rec["corpus_id"],
            config=rec["config"],
            pdf_url=rec["pdf_url"],
            title=rec["title"],
            file=str(md_path),
            pages=len(md_to_pages(md)),
            n_pdf_pages=n_pdf,
            source_path=rec["source_path"],
            sha256=rec["sha256"],
            size_bytes=rec.get("size_bytes"),
            parser=library_versions(),
            parsed_at=utcnow(),
            metadata=rec.get("metadata") or {},
        ).model_dump(mode="json")
        return rec, row, None, {"fetch": t_fetch, "parse": t_parse}
    except Exception as exc:  # noqa: BLE001 - a bad PDF must not stop the batch
        return rec, None, f"{type(exc).__name__}: {exc}", {"fetch": t_fetch, "parse": t_parse}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def manifest_path(parsed_dir: Path, config: str) -> Path:
    return Path(parsed_dir) / f"{config}.jsonl"


def read_manifest(parsed_dir: Path, config: str) -> Dict[str, Dict[str, Any]]:
    """``corpus_id -> manifest row`` (append-only file; the last row of an id wins)."""
    p = manifest_path(parsed_dir, config)
    return {r["corpus_id"]: r for r in mf.jsonl_rows(p)} if p.exists() else {}


def run_parse(
    source: Path,
    parsed_dir: Path,
    config: str = "local",
    workers: Optional[int] = None,
    limit: Optional[int] = None,
    retry_failed: bool = False,
    no_resume: bool = False,
    timeout: Optional[int] = 600,
    max_pages: Optional[int] = None,
    use_processes: bool = True,
    run_id: Optional[str] = None,
    events: Optional[EventLog] = None,
    archives: bool = True,
    max_archive_depth: int = 2,
) -> Dict[str, int]:
    """Parse every pending document under ``source`` into ``parsed_dir``."""
    parsed_dir = Path(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id("parse")
    events = events or EventLog(parsed_dir / "events.jsonl", run_id, "parse")
    workers = workers or min(32, os.cpu_count() or 1)
    (parsed_dir / PARSE_COMPLETE).unlink(missing_ok=True)

    t0 = time.monotonic()
    cache_path = parsed_dir / "hash_cache.json"
    hash_cache = mf.load_hash_cache(cache_path)
    docs, inv = mf.scan_input(Path(source), config, parsed_dir / "unpacked" if archives else None,
                              max_archive_depth, hash_cache)
    mf.save_hash_cache(cache_path, hash_cache)
    if inv is not None:
        _write_atomic(parsed_dir / "inventory.json", json.dumps(inv.to_json(), indent=1))
        for line in inv.report().splitlines():
            log.info("[inventory] %s", line)
        for d in inv.duplicates:
            log.info("[inventory] duplicate dropped: %s (same bytes as %s)", d["dropped"], d["kept"])
        if inv.problems:
            log.warning("[inventory] %d problem(s); see %s", len(inv.problems), parsed_dir / "inventory.json")
        events_inv = {"inventory": inv.summary()}
    else:
        events_inv = {}
    log.info("[start] run=%s source=%s config=%s docs=%d workers=%d", run_id, source, config, len(docs), workers)
    status = {} if no_resume else mf.load_status(parsed_dir / PARSE_CHECKPOINT)
    todo = []
    skipped_done = skipped_failed = 0
    for rec in docs:
        st = status.get(rec["corpus_id"], {}).get("status")
        md_ok = (parsed_dir / config / mf.doc_filename(rec["corpus_id"])).exists()
        if st == "done" and md_ok:
            skipped_done += 1
        elif st == "failed" and not retry_failed:
            skipped_failed += 1
        else:
            todo.append(rec)
    if limit:
        todo = todo[:limit]
    log.info("[plan] total=%d todo=%d skip_done=%d skip_failed=%d (use --retry-failed)", len(docs), len(todo), skipped_done, skipped_failed)
    events.emit("start", source=str(source), config=config, total=len(docs), todo=len(todo), workers=workers, **events_inv)

    prog = Progress(len(todo))
    mpath = manifest_path(parsed_dir, config)
    if no_resume:
        mpath.unlink(missing_ok=True)

    def handle(rec, row, err, timing):
        if err is None:
            # manifest row first, then checkpoint: stage 2 may be reading the manifest live
            mf.append_jsonl(mpath, [json.dumps(row)])
            mf.append_checkpoint(parsed_dir / PARSE_CHECKPOINT, rec["corpus_id"], "done")
            prog.step(True)
            log.info("[parse] ok %s %r pages=%d/%d fetch=%s parse=%s | %s", rec["corpus_id"], rec["title"][:60], row["pages"], row["n_pdf_pages"], fmt_secs(timing["fetch"]), fmt_secs(timing["parse"]), prog.line())
            events.emit("doc_done", corpus_id=rec["corpus_id"], title=rec["title"], pages=row["pages"], n_pdf_pages=row["n_pdf_pages"], sha256=rec["sha256"], timing=timing)
        else:
            mf.append_checkpoint(parsed_dir / PARSE_CHECKPOINT, rec["corpus_id"], "failed", err[:500])
            prog.step(False)
            log.warning("[parse] FAILED %s %r: %s | %s", rec["corpus_id"], rec["title"][:60], err[:200], prog.line())
            events.emit("doc_failed", corpus_id=rec["corpus_id"], title=rec["title"], error=err[:500], timing=timing)

    if use_processes and workers > 1 and len(todo) > 1:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(parse_one, r, str(parsed_dir), timeout, max_pages) for r in todo]
            for fut in cf.as_completed(futs):
                try:
                    handle(*fut.result())
                except Exception as exc:  # worker died (OOM, segfault)
                    log.error("[parse] worker crashed: %s", exc)
                    prog.step(False)
    else:
        for r in todo:
            handle(*parse_one(r, str(parsed_dir), timeout, max_pages))

    (parsed_dir / PARSE_COMPLETE).write_text(run_id)
    summary = {"done": prog.done, "failed": prog.failed, "skipped_done": skipped_done, "skipped_failed": skipped_failed, "manifest_rows": len(read_manifest(parsed_dir, config))}
    log.info("[finish] %s elapsed=%s manifest=%s", summary, fmt_secs(time.monotonic() - t0), mpath)
    events.emit("finish", elapsed=time.monotonic() - t0, **summary)
    return summary
