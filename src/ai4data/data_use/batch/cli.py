"""Command line: ``python -m ai4data.data_use.batch {parse,extract,run,status}``."""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from . import manifest as mf
from .logging_utils import fmt_secs, new_run_id, setup_logging


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--work-dir", type=Path, default=Path("data_use_batch"),
                   help="holds parsed/ and output/ unless overridden (default: ./data_use_batch)")
    p.add_argument("--parsed-dir", type=Path, help="default: <work-dir>/parsed")
    p.add_argument("--config", default="local", help="corpus name; prefix of every corpus_id (default: local)")
    p.add_argument("--limit", type=int,
                   help="process at most N pending documents in this run (already-done ones do not count)")
    p.add_argument("--no-resume", action="store_true", help="ignore checkpoints and redo everything")
    p.add_argument("-v", "--verbose", action="store_true")


def _extract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-dir", type=Path, help="default: <work-dir>/output")
    p.add_argument("--device", help="cuda | cuda:1 | mps | cpu (default: best available)")
    p.add_argument("--entity-model", help="GLiNER model id (default: rafmacalaba/gliner_agents_v3)")
    p.add_argument("--metadata-model", help="LFM model id (default: rafmacalaba/lfm2.5-350M-focus-metadata)")
    p.add_argument("--require-gpu", action="store_true",
                   help="abort if a model would run on the CPU (use on the GPU server)")
    p.add_argument("--no-enrich", action="store_true", help="skip the LFM usage/attributes step")
    p.add_argument("--batch-docs", type=int, default=8, help="documents per loop (default 8)")
    p.add_argument("--extract-batch-size", type=int, default=32, help="chunks per GLiNER forward pass")
    p.add_argument("--lfm-batch-size", type=int, default=16, help="prompts per LFM generate call")
    p.add_argument("--chunk-tokens", type=int, default=384, help="chunk size in --chunk-unit (default 384)")
    p.add_argument("--chunk-unit", choices=("words", "tokens"), default="words",
                   help="words = GLiNER's real 384-word window (default); tokens = playground-style subword chunks, "
                        "which can exceed that window and get truncated")
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.5, help="GLiNER score threshold (default 0.5)")
    p.add_argument("--no-footnote-link", action="store_true")
    p.add_argument("--no-dedupe-overlap", action="store_true")
    p.add_argument("--poll-interval", type=float, default=30.0)
    p.add_argument("--idle-exit", type=float, default=600.0,
                   help="exit after N idle seconds when stage 1 is not finished (0 = wait forever; use 0 when "
                        "starting extract before parse)")
    p.add_argument("--allow-stamp-mismatch", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="count what would be processed and exit")


def _parse_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", type=Path, required=True,
                   help="directory of PDFs, or a .jsonl manifest (pdf_url|path, optional title and metadata)")
    p.add_argument("--workers", type=int, help="parse processes (default min(32, cpus))")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--timeout", type=int, default=600, help="per-document parse budget in seconds (0 = none)")
    p.add_argument("--max-pages", type=int, help="skip PDFs with more pages")
    p.add_argument("--no-archives", action="store_true", help="do not open .zip/.tar*/.gz files")
    p.add_argument("--max-archive-depth", type=int, default=2,
                   help="archive-in-archive nesting to open (default 2; 0 = none)")
    p.add_argument("--no-processes", action="store_true", help="parse in-process (debugging)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ai4data.data_use.batch", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse", help="stage 1: PDFs -> markdown (CPU)")
    _common(p); _parse_args(p)
    p = sub.add_parser("extract", help="stage 2: markdown -> mentions (GPU)")
    _common(p); _extract_args(p)
    p = sub.add_parser("run", help="both stages at once (parse in a background thread)")
    _common(p); _parse_args(p); _extract_args(p)
    p = sub.add_parser("scan", help="inventory a source tree (folders, archives, duplicates, problems); no parsing")
    _common(p)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--no-archives", action="store_true")
    p.add_argument("--max-archive-depth", type=int, default=2)
    p.add_argument("--json", action="store_true", help="print the full inventory as JSON")
    p = sub.add_parser("status", help="progress of both stages")
    _common(p)
    p.add_argument("--output-dir", type=Path)
    return ap


def _dirs(a):
    parsed = a.parsed_dir or a.work_dir / "parsed"
    out = getattr(a, "output_dir", None) or a.work_dir / "output"
    return parsed, out


def _parse_kwargs(a):
    return dict(source=a.source, config=a.config, workers=a.workers, limit=a.limit,
                retry_failed=a.retry_failed, no_resume=a.no_resume, timeout=a.timeout or None,
                max_pages=a.max_pages, use_processes=not a.no_processes,
                archives=not a.no_archives, max_archive_depth=a.max_archive_depth)


def _extract_kwargs(a):
    return dict(config=a.config, device=a.device, entity_model_id=a.entity_model,
                metadata_model_id=a.metadata_model, enrich=not a.no_enrich, batch_docs=a.batch_docs,
                extract_batch_size=a.extract_batch_size, lfm_batch_size=a.lfm_batch_size,
                chunk_tokens=a.chunk_tokens, overlap=a.overlap, chunk_unit=a.chunk_unit, threshold=a.threshold,
                footnote_link=not a.no_footnote_link, dedupe_overlap=not a.no_dedupe_overlap,
                limit=a.limit, no_resume=a.no_resume, poll_interval=a.poll_interval,
                idle_exit=a.idle_exit, dry_run=a.dry_run, allow_stamp_mismatch=a.allow_stamp_mismatch, require_gpu=a.require_gpu,
                verbose=a.verbose)


def _count(path: Path) -> int:
    return sum(1 for _ in path.open()) if path.exists() else 0


def status(parsed: Path, out: Path, config: str) -> dict:
    from .parse import PARSE_CHECKPOINT, read_manifest

    man = read_manifest(parsed, config)
    pst = mf.load_status(parsed / PARSE_CHECKPOINT)
    est = mf.load_status(out / "checkpoint.jsonl")
    ready = [c for c, m in man.items() if Path(m["file"]).exists()]
    info = {
        "parse": {"done": sum(s["status"] == "done" for s in pst.values()),
                  "failed": sum(s["status"] == "failed" for s in pst.values()),
                  "complete_marker": (parsed / ".parse_complete").exists()},
        "extract": {"done": sum(s["status"] == "done" for s in est.values()),
                    "failed": sum(s["status"] == "failed" for s in est.values()),
                    "waiting": sum(1 for c in ready if est.get(c, {}).get("status") != "done")},
        "rows": {n: _count(out / n) for n in ("extractions.jsonl", "enrichment.jsonl", "datasets.jsonl")},
        "failures": {c: s.get("error") for c, s in {**pst, **est}.items() if s["status"] == "failed"},
    }
    return info


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    parsed, out = _dirs(a)
    if a.cmd == "scan":
        cache_path = parsed / "hash_cache.json"
        cache = mf.load_hash_cache(cache_path) if parsed.exists() else {}
        parsed.mkdir(parents=True, exist_ok=True)
        recs, inv = mf.scan_input(a.source, a.config, None if a.no_archives else parsed / "unpacked",
                                  a.max_archive_depth, cache)
        mf.save_hash_cache(cache_path, cache)
        (parsed / "inventory.json").write_text(json.dumps(inv.to_json(), indent=1))
        print(json.dumps(inv.to_json(), indent=1) if a.json else inv.report())
        print(f"\nunique documents to process: {len(recs)}  (inventory: {parsed / 'inventory.json'})")
        return 0
    if a.cmd == "status":
        print(json.dumps(status(parsed, out, a.config), indent=2))
        return 0

    from .extract import run_extract
    from .parse import run_parse

    if a.cmd == "parse":
        setup_logging(parsed / "logs", new_run_id("parse"), a.verbose)
        s = run_parse(parsed_dir=parsed, **_parse_kwargs(a))
        return 1 if s["failed"] and not s["done"] else 0
    if a.cmd == "extract":
        t = run_extract(parsed, out, **_extract_kwargs(a))
        return 1 if t.get("failed") and not t.get("processed") else 0

    # run: stage 1 in a background thread (its own process pool), stage 2 consumes
    result = {}
    parse_run = new_run_id("parse")

    def _stage1():
        try:
            setup_logging(parsed / "logs", parse_run, a.verbose)
            result["parse"] = run_parse(parsed_dir=parsed, run_id=parse_run, **_parse_kwargs(a))
        except BaseException as exc:  # noqa: BLE001
            result["parse_error"] = exc

    th = threading.Thread(target=_stage1, name="stage1-parse")
    th.start()
    t = run_extract(parsed, out, stop_when=lambda: not th.is_alive(), **_extract_kwargs(a))
    th.join()
    if "parse_error" in result:
        print(f"stage 1 crashed: {result['parse_error']!r}", file=sys.stderr)
        return 2
    return 0
