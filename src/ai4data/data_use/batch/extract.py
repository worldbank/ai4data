"""Stage 2: chunk the parsed markdown and extract mentions (GPU consumer).

Per loop it takes up to ``batch_docs`` parsed documents that are not yet
checkpointed, chunks them (tokenizer-aware, footnote-linked), runs GLiNER over
all their chunks in batches, runs the LFM metadata model over every named
mention in batches, and writes per document:

    extractions.jsonl   one row per chunk (playground-compatible, strict)
    enrichment.jsonl    LFM usage/attributes per named mention
    datasets.jsonl      one deduplicated record per document
    checkpoint.jsonl    done / failed (written last)

Rows of a document are appended together and the checkpoint last; on start,
rows of documents without a ``done`` checkpoint are removed so a crash never
leaves duplicates.
"""

import importlib.metadata
import json
import logging
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..schemas.dataset_schema import (
    ENTITY_LABELS,
    build_call2_fields,
    empty_call2_fields,
)
from . import manifest as mf
from .chunking import chunk_text, dedupe_overlap_entities
from .footnotes import WORD_RE, attach_footnotes, map_span_to_page
from .logging_utils import EventLog, Progress, fmt_secs, new_run_id, setup_logging, utcnow
from .markdown import md_to_pages
from .parse import PARSE_COMPLETE, read_manifest
from .schemas import EnrichmentRecord, EntitySpan, ExtractionRecord

log = logging.getLogger("ai4data.batch.extract")

LABELS = tuple(ENTITY_LABELS)
OUTPUT_FILES = ("extractions.jsonl", "enrichment.jsonl", "datasets.jsonl")
CHECKPOINT = "checkpoint.jsonl"
RUN_JSON = "run.json"
MIN_MENTION_CHARS = 3  # enrichment / datasets skip spans of 1-2 characters


# ── model handles ──────────────────────────────────────────────────────────────


def model_revision(model_id: str) -> Optional[str]:
    """Snapshot hash of a cached Hub model (None if unknown or a local path)."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True)).name
    except Exception:  # noqa: BLE001
        return None


def library_versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "gliner", "pymupdf", "pymupdf4llm"):
        try:
            out[mod] = importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError:
            out[mod] = "not-installed"
    return out


class Models:
    """The two models plus the chunk tokenizer. Tests can inject fakes."""

    def __init__(self, entity, chunk_tokenizer, metadata=None):
        self.entity = entity
        self.chunk_tokenizer = chunk_tokenizer
        self.metadata = metadata  # (tokenizer, model) or None


def load_models(entity_model_id, metadata_model_id, device, enrich=True, require_gpu=False) -> Models:
    import torch

    from ..models.model_manager import ModelManager

    mm = ModelManager(device=device)
    resolved = mm._map_location()
    if require_gpu and resolved == "cpu":
        raise SystemExit("--require-gpu: no CUDA/MPS device available (resolved to cpu)")
    entity = mm.load_entity_model(entity_model_id)
    tok = entity.data_processor.transformer_tokenizer
    meta = mm.load_metadata_model(metadata_model_id) if enrich else None
    where = {"entity": str(next(entity.parameters()).device)}
    if meta is not None:
        where["metadata"] = str(meta[1].device)
    if torch.cuda.is_available() and resolved.startswith("cuda"):
        props = torch.cuda.get_device_properties(torch.device(resolved))
        log.info("[gpu] %s %.0f GB | models on %s | torch %s cuda %s", props.name,
                 props.total_memory / 2**30, where, torch.__version__, torch.version.cuda)
    else:
        log.info("[gpu] no CUDA in use | models on %s", where)
    if require_gpu and any(d == "cpu" for d in where.values()):
        raise SystemExit(f"--require-gpu: a model is on cpu: {where}")
    return Models(entity, tok, meta)


# ── run stamp ──────────────────────────────────────────────────────────────────

_STAMP_KEYS = (
    "entity_model_id", "entity_revision", "metadata_model_id", "metadata_revision",
    "labels", "threshold", "chunk_tokens", "overlap", "chunk_unit", "footnote_link",
    "dedupe_overlap", "enrich",
)


def check_stamp(out: Path, stamp: Dict[str, Any], run_id: str, allow_mismatch: bool) -> None:
    """Write ``run.json`` or refuse to resume with different settings.

    Rows of one output directory must come from one configuration; mixing chunk
    sizes or models would make them silently inconsistent.
    """
    path = out / RUN_JSON
    entry = {"run_id": run_id, "started_at": utcnow(), "host": socket.gethostname(),
             "platform": platform.platform(), "argv": sys.argv, "libraries": library_versions()}
    if path.exists():
        prev = json.loads(path.read_text())
        old = {k: prev["stamp"].get(k) for k in _STAMP_KEYS}
        new = {k: stamp.get(k) for k in _STAMP_KEYS}
        if old != new and not allow_mismatch:
            diff = {k: (old[k], new[k]) for k in _STAMP_KEYS if old[k] != new[k]}
            raise SystemExit(
                f"{path} was written with different settings {diff}. Use a fresh --output-dir "
                f"(or --allow-stamp-mismatch to mix them)."
            )
        prev["runs"].append(entry)
        prev["stamp"] = stamp if allow_mismatch else prev["stamp"]
        doc = prev
    else:
        doc = {"stamp": stamp, "runs": [entry]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


# ── chunking ───────────────────────────────────────────────────────────────────


def build_tasks(pages, tokenizer, chunk_tokens, overlap, footnote_link, chunk_unit="words"):
    """Chunks of a document: ``tasks`` = (page, chunk_idx, text), ``chunks`` = chunk dicts,
    ``groups`` = (page, start, end) index ranges per page (for overlap dedupe)."""
    tasks, chunks, groups = [], [], []
    for page in pages:
        page_chunks = list(chunk_text(page["text"], tokenizer, max_tokens=chunk_tokens, overlap=overlap, unit=chunk_unit))
        if footnote_link:
            page_chunks = attach_footnotes(page["text"], page_chunks, tokenizer, max_tokens=chunk_tokens)
        s0 = len(tasks)
        tasks.extend((page["page"], i, c["text"]) for i, c in enumerate(page_chunks))
        chunks.extend(page_chunks)
        groups.append((page["page"], s0, len(tasks)))
    return tasks, chunks, groups


def to_spans(entities, threshold) -> List[EntitySpan]:
    return [
        EntitySpan(text=e["text"], label=e["label"], score=float(e["score"]),
                   start=int(e["start"]), end=int(e["end"]))
        for e in entities or []
        if e.get("score", 0.0) >= threshold and e.get("label") in ENTITY_LABELS
    ]


# ── per-batch processing ───────────────────────────────────────────────────────


class DocWork:
    """Everything computed for one document before anything is written."""

    def __init__(self, meta, pages):
        self.meta, self.pages = meta, pages
        self.tasks, self.chunks, self.groups = [], [], []
        self.spans: List[List[EntitySpan]] = []
        self.dropped_overlap = 0
        self.enrich: Dict[Tuple[int, int, int], Dict[str, Any]] = {}  # (task, ent idx) -> usage/attrs/passage
        self.error: Optional[str] = None


def _enrich_batch(works, models, lfm_batch_size, usage_system, attributes_system, predict):
    """One batched LFM pass (usage + attributes) over the named mentions of ``works``."""
    # The LFM model was trained on passage chunks, so the passage is the whole chunk
    # (the same text as ``input_text`` in extractions.jsonl). The same mention twice
    # in one chunk is asked once.
    keys, items, index = [], [], {}
    for w in works:
        if w.error:
            continue
        for ti, spans in enumerate(w.spans):
            for ei, sp in enumerate(spans):
                if sp.label == "NAMED_DATA" and len(sp.text.strip()) >= MIN_MENTION_CHARS:
                    passage = w.tasks[ti][2]
                    k = (passage, sp.text)
                    if k not in index:
                        index[k] = len(items)
                        items.append({"passage": passage, "mention": sp.text})
                    keys.append((w, ti, ei, index[k]))
    if not items:
        return 0
    tok, lfm = models.metadata
    usage = predict(tok, lfm, usage_system, items, batch_size=lfm_batch_size)
    attrs = predict(tok, lfm, attributes_system, items, batch_size=lfm_batch_size)
    for w, ti, ei, i in keys:
        w.enrich[(ti, ei)] = {"passage": items[i]["passage"], "usage": usage[i], "attributes": attrs[i]}
    return len(items)


def _doc_rows(w: DocWork, config_name: str):
    """Serialized rows for one finished document: extractions, enrichment, dataset record."""
    meta = w.meta
    cid = meta["corpus_id"]
    ext_rows, enr_rows = [], []
    page_text = {p["page"]: p["text"] for p in w.pages}
    per_page: Dict[int, List[Dict[str, Any]]] = {}
    n_spans = n_named = 0
    for ti, ((page, chunk_idx, text), spans) in enumerate(zip(w.tasks, w.spans)):
        ext_rows.append(
            ExtractionRecord(
                corpus_id=cid, config=meta["config"], pdf_url=meta["pdf_url"], title=meta["title"],
                page=page, chunk=chunk_idx, input_text=text,
                has_data_score=1.0 if spans else 0.0, entities=spans,
            ).model_dump_json()
        )
        for ei, sp in enumerate(spans):
            n_spans += 1
            n_named += sp.label == "NAMED_DATA"
            if len(sp.text.strip()) < MIN_MENTION_CHARS:
                continue
            mapped = map_span_to_page(w.chunks[ti], sp.start, sp.end)
            ps, pe = (mapped[0], mapped[1]) if mapped else (None, None)
            rec = {
                "mention_name": {"text": sp.text, "confidence": sp.score, "start": ps, "end": pe},
                "specificity_tag": ENTITY_LABELS[sp.label],
            }
            e = w.enrich.get((ti, ei))
            if e is not None:
                s0, e0 = (ps, pe) if mapped else (sp.start, sp.end)
                rec.update(build_call2_fields(e["usage"], e["attributes"], sp.text, s0, e0))
                enr = rec
                enr_rows.append(
                    EnrichmentRecord(
                        corpus_id=cid, page=page, chunk=chunk_idx, start=sp.start, end=sp.end,
                        text=sp.text,
                        usage_context=enr["usage_context"]["text"], is_used=enr["is_used"]["text"],
                        usage_action=enr["usage_action"]["text"], usage_summary=enr["usage_summary"]["text"],
                        purpose_action=enr["purpose_action"]["text"], theme_tags=enr["theme_tags"],
                        source_attributes=enr["source_attributes"],
                        raw_usage=e["usage"], raw_attributes=e["attributes"],
                    ).model_dump_json()
                )
            else:
                rec.update(empty_call2_fields(ps or 0, pe or 0))
            per_page.setdefault(page, []).append(rec)

    # document-level merge, with the same function users call on extract_from_document output
    from ..extractors.deduplication import deduplicate_extraction

    page_results = [
        {"input_text": page_text.get(pg, ""), "datasets": recs,
         "document": {"source": cid, "pages": [pg]}}
        for pg, recs in sorted(per_page.items())
    ]
    merged = deduplicate_extraction(page_results)["datasets"] if page_results else []
    ds_row = json.dumps({
        "corpus_id": cid, "config": meta["config"], "pdf_url": meta["pdf_url"], "title": meta["title"],
        "sha256": meta.get("sha256"), "n_pages": len(w.pages), "n_chunks": len(w.tasks),
        "n_spans": n_spans, "n_named": n_named, "datasets": merged,
    }, default=str)
    return ext_rows, enr_rows, ds_row, {"spans": n_spans, "named": n_named, "enriched": len(enr_rows), "datasets": len(merged)}


def process_batch(metas, models, cfg, predict_metadata_fn, usage_system, attributes_system):
    """Compute (not write) results for ``metas``. Returns a list of DocWork."""
    works = []
    t0 = time.perf_counter()
    for meta in metas:
        try:
            pages = md_to_pages(Path(meta["file"]).read_text(encoding="utf-8"))
            if not pages:
                raise ValueError("md has no pages with text")
            w = DocWork(meta, pages)
            w.tasks, w.chunks, w.groups = build_tasks(
                pages, models.chunk_tokenizer, cfg["chunk_tokens"], cfg["overlap"], cfg["footnote_link"],
                cfg["chunk_unit"],
            )
        except Exception as exc:  # noqa: BLE001
            w = DocWork(meta, [])
            w.error = f"{type(exc).__name__}: {exc}"
        works.append(w)
    t_prep = time.perf_counter() - t0
    n_over = sum(
        1 for w in works if not w.error for _p, _i, text in w.tasks
        if len(WORD_RE.findall(text)) > cfg["chunk_tokens"]
    )
    if n_over:  # GLiNER silently truncates these, losing mentions in their tail
        log.warning("[chunks] %d chunk(s) exceed the %d-word GLiNER window and will be truncated "
                    "(use --chunk-unit words)", n_over, cfg["chunk_tokens"])

    # GLiNER over the chunks of every document in the batch
    t0 = time.perf_counter()
    flat = [(w, ti) for w in works if not w.error for ti in range(len(w.tasks))]
    texts = [w.tasks[ti][2] for w, ti in flat]
    ents = []
    for i in range(0, len(texts), cfg["extract_batch_size"]):
        ents.extend(models.entity.inference(
            texts[i : i + cfg["extract_batch_size"]], list(LABELS), flat_ner=True,
            threshold=cfg["threshold"], batch_size=cfg["extract_batch_size"],
        ))
    t_gliner = time.perf_counter() - t0
    for w in works:
        w.spans = [[] for _ in w.tasks]
    for (w, ti), e in zip(flat, ents):
        w.spans[ti] = to_spans(e, cfg["threshold"])
    for w in works:
        if w.error or not cfg["dedupe_overlap"]:
            continue
        for _page, s0, s1 in w.groups:
            kept, dropped = dedupe_overlap_entities(w.chunks[s0:s1], w.spans[s0:s1])
            for ti, k in enumerate(kept):
                w.spans[s0 + ti] = k
            w.dropped_overlap += len(dropped)

    # LFM usage + attributes over the named mentions
    t0 = time.perf_counter()
    n_enriched = 0
    if cfg["enrich"] and models.metadata is not None:
        n_enriched = _enrich_batch(works, models, cfg["lfm_batch_size"], usage_system, attributes_system, predict_metadata_fn)
    t_lfm = time.perf_counter() - t0
    gpu_mem = ""
    peak = None
    try:
        import torch

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            gpu_mem = f" gpu_peak={peak:.1f}GB"
    except Exception:  # noqa: BLE001
        pass
    log.info(
        "[batch] docs=%d chunks=%d prep=%s gliner=%s (%.0f chunks/s) lfm=%s (%d mentions)%s",
        len(works), len(texts), fmt_secs(t_prep), fmt_secs(t_gliner),
        len(texts) / t_gliner if t_gliner else 0, fmt_secs(t_lfm), n_enriched, gpu_mem,
    )
    return works, {"prep": t_prep, "gliner": t_gliner, "lfm": t_lfm, "chunks": len(texts),
                   "enriched": n_enriched, "gpu_peak_gb": peak}


# ── driver ─────────────────────────────────────────────────────────────────────


def run_extract(
    parsed_dir: Path,
    output_dir: Path,
    config: str = "local",
    device: Optional[str] = None,
    entity_model_id: Optional[str] = None,
    metadata_model_id: Optional[str] = None,
    enrich: bool = True,
    batch_docs: int = 8,
    extract_batch_size: int = 32,
    lfm_batch_size: int = 16,
    chunk_tokens: int = 384,
    overlap: int = 50,
    chunk_unit: str = "words",
    threshold: float = 0.5,
    footnote_link: bool = True,
    dedupe_overlap: bool = True,
    limit: Optional[int] = None,
    no_resume: bool = False,
    poll_interval: float = 30.0,
    idle_exit: float = 600.0,
    stop_when: Optional[Callable[[], bool]] = None,
    dry_run: bool = False,
    allow_stamp_mismatch: bool = False,
    require_gpu: bool = False,
    run_id: Optional[str] = None,
    models: Optional[Models] = None,
    verbose: bool = False,
) -> Dict[str, int]:
    """Extract from every parsed document under ``parsed_dir``; returns counters."""
    from ..models.metadata_model import ATTRIBUTES_SYSTEM, USAGE_SYSTEM, predict_metadata
    from ..models.model_manager import ModelManager

    parsed_dir, out = Path(parsed_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = run_id or new_run_id("extract")
    setup_logging(out / "logs", run_id, verbose)
    events = EventLog(out / "events.jsonl", run_id, "extract")
    entity_model_id = entity_model_id or ModelManager.DEFAULT_ENTITY_MODEL_ID
    metadata_model_id = metadata_model_id or ModelManager.DEFAULT_METADATA_MODEL_ID

    stamp = {
        "entity_model_id": entity_model_id, "entity_revision": model_revision(entity_model_id),
        "metadata_model_id": metadata_model_id if enrich else None,
        "metadata_revision": model_revision(metadata_model_id) if enrich else None,
        "labels": list(LABELS), "threshold": threshold, "chunk_tokens": chunk_tokens, "overlap": overlap,
        "chunk_unit": chunk_unit, "footnote_link": footnote_link, "dedupe_overlap": dedupe_overlap, "enrich": enrich,
        "extract_batch_size": extract_batch_size, "lfm_batch_size": lfm_batch_size, "device": device,
    }
    if no_resume:
        for name in (*OUTPUT_FILES, CHECKPOINT, RUN_JSON):
            (out / name).unlink(missing_ok=True)
        log.info("[reset] --no-resume: outputs in %s removed", out)
    if not dry_run:
        check_stamp(out, stamp, run_id, allow_stamp_mismatch)

    done = mf.load_done(out / CHECKPOINT)
    removed = mf.reconcile_outputs([out / n for n in OUTPUT_FILES], done)
    if removed:
        log.warning("[reconcile] removed rows of un-checkpointed documents (crash recovery): %s", removed)
        events.emit("reconcile", removed=removed)

    def ready_docs(attempted):
        rows = read_manifest(parsed_dir, config)
        return [m for cid, m in rows.items()
                if cid not in done and cid not in attempted and Path(m["file"]).exists()]

    total = len(read_manifest(parsed_dir, config))
    first = ready_docs(set())
    log.info("[start] run=%s parsed=%d extracted=%d ready=%d device=%s entity=%s@%s metadata=%s@%s enrich=%s",
             run_id, total, len(done), len(first), device or "auto", entity_model_id,
             (stamp["entity_revision"] or "?")[:8], metadata_model_id, (stamp["metadata_revision"] or "?")[:8], enrich)
    events.emit("start", parsed=total, extracted=len(done), ready=len(first), stamp=stamp)
    if dry_run:
        return {"processed": 0, "failed": 0, "remaining": len(first)}

    cfg = {"chunk_tokens": chunk_tokens, "overlap": overlap, "chunk_unit": chunk_unit, "threshold": threshold,
           "extract_batch_size": extract_batch_size, "lfm_batch_size": lfm_batch_size,
           "footnote_link": footnote_link, "dedupe_overlap": dedupe_overlap, "enrich": enrich}
    t_load = time.perf_counter()
    if models is None:
        models = load_models(entity_model_id, metadata_model_id, device, enrich, require_gpu)
        log.info("[models] loaded in %s", fmt_secs(time.perf_counter() - t_load))

    attempted: set = set()
    n_done_at_start = len(done)
    totals = {"processed": 0, "failed": 0, "spans": 0, "named": 0, "enriched": 0, "datasets": 0}
    target = len(first) if not limit else min(limit, len(first))
    prog = Progress(target)
    idle_since = None
    stop = False
    while not stop:
        ready = ready_docs(attempted)
        if limit:
            ready = ready[: max(0, limit - len(attempted))]
        # stage 1 may still be adding documents: the total grows with the manifest
        prog.total = len(attempted) + len(ready) if limit else len(read_manifest(parsed_dir, config)) - n_done_at_start
        if not ready:
            parse_done = (parsed_dir / PARSE_COMPLETE).exists() or (stop_when is not None and stop_when())
            if parse_done or limit and len(attempted) >= limit:
                log.info("[idle] nothing left and stage 1 finished")
                break
            idle_since = idle_since or time.monotonic()
            if idle_exit and time.monotonic() - idle_since > idle_exit:
                log.info("[idle] no new documents for %s, exiting", fmt_secs(idle_exit))
                break
            time.sleep(poll_interval)
            continue
        idle_since = None
        batch = ready[:batch_docs]
        attempted.update(m["corpus_id"] for m in batch)
        try:
            works, timing = process_batch(batch, models, cfg, predict_metadata, USAGE_SYSTEM, ATTRIBUTES_SYSTEM)
        except Exception as exc:  # noqa: BLE001 - e.g. CUDA OOM: retry one by one
            log.error("[batch] failed (%s: %s); retrying documents one at a time", type(exc).__name__, exc)
            works, timing = [], {}
            for m in batch:
                try:
                    ws, _ = process_batch([m], models, cfg, predict_metadata, USAGE_SYSTEM, ATTRIBUTES_SYSTEM)
                    works.extend(ws)
                except Exception as exc1:  # noqa: BLE001
                    w = DocWork(m, [])
                    w.error = f"{type(exc1).__name__}: {exc1}"
                    works.append(w)
        for w in works:
            cid = w.meta["corpus_id"]
            if w.error is None:
                try:
                    ext_rows, enr_rows, ds_row, stats = _doc_rows(w, config)
                    mf.append_jsonl(out / "extractions.jsonl", ext_rows)
                    mf.append_jsonl(out / "enrichment.jsonl", enr_rows)
                    mf.append_jsonl(out / "datasets.jsonl", [ds_row])
                except Exception as exc:  # noqa: BLE001
                    w.error = f"{type(exc).__name__}: {exc}"
            if w.error is None:
                mf.append_checkpoint(out / CHECKPOINT, cid, "done")
                done.add(cid)
                prog.step(True)
                totals["processed"] += 1
                for k in ("spans", "named", "enriched", "datasets"):
                    totals[k] += stats[k]
                log.info("[extract] ok %s %r pages=%d chunks=%d spans=%d named=%d enriched=%d datasets=%d dropped_overlap=%d | %s",
                         cid, w.meta["title"][:60], len(w.pages), len(w.tasks), stats["spans"], stats["named"],
                         stats["enriched"], stats["datasets"], w.dropped_overlap, prog.line())
                events.emit("doc_done", corpus_id=cid, title=w.meta["title"], sha256=w.meta.get("sha256"),
                            pages=len(w.pages), chunks=len(w.tasks), dropped_overlap=w.dropped_overlap, **stats)
            else:
                mf.append_checkpoint(out / CHECKPOINT, cid, "failed", w.error[:500])
                prog.step(False)
                totals["failed"] += 1
                log.warning("[extract] FAILED %s %r: %s | %s", cid, w.meta["title"][:60], w.error[:200], prog.line())
                events.emit("doc_failed", corpus_id=cid, title=w.meta["title"], error=w.error[:500])
        events.emit("batch", docs=len(batch), **timing)
    log.info("[finish] %s | %s", totals, prog.line())
    events.emit("finish", **totals)
    return totals
