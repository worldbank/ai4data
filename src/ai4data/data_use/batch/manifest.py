"""Input manifest, document ids, checkpoints.

Document ids are content based (``<config>:<sha256[:16]>``), not an index after
sorting, so adding files never changes the ids of documents already processed.
Local PDFs hash their bytes; ``pdf_url`` rows hash the URL (the bytes are not
known before the download).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

from .schemas import CheckpointRecord

ID_HEX_LEN = 16
_CORE_COLUMNS = {"pdf_url", "path", "file", "title", "corpus_id", "config"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_file_cached(path: Path, cache: Optional[Dict[str, str]]) -> str:
    """``sha256_file`` with a cache keyed by (path, size, mtime) so restarts do not re-read PDFs."""
    if cache is None:
        return sha256_file(path)
    st = Path(path).stat()
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
    if key not in cache:
        cache[key] = sha256_file(path)
    return cache[key]


def load_hash_cache(path: Path) -> Dict[str, str]:
    p = Path(path)
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_hash_cache(path: Path, cache: Dict[str, str]) -> None:
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    os.replace(tmp, p)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_filename(corpus_id: str) -> str:
    return corpus_id.replace(":", "_") + ".md"


def jsonl_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON ({exc})") from exc


def _title_from(name: str) -> str:
    return Path(name).stem


def input_records(source: Path, config: str, unpack_dir: Optional[Path] = None,
                  max_archive_depth: int = 2, hash_cache: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """``scan_input(...)[0]``: the documents only."""
    return scan_input(source, config, unpack_dir, max_archive_depth, hash_cache)[0]


def scan_input(
    source: Path,
    config: str,
    unpack_dir: Optional[Path] = None,
    max_archive_depth: int = 2,
    hash_cache: Optional[Dict[str, str]] = None,
):
    """Documents to process, from a directory tree or a JSONL manifest.

    A directory is walked at every level for ``.pdf`` (any case); with
    ``unpack_dir`` set, ``.zip``/``.tar*``/``.gz`` archives are unpacked there and
    their PDFs included (see ``discover``). A ``.jsonl`` file has one row per
    document with ``pdf_url`` (http(s) or a local path) or ``path``; ``title`` is
    optional and every other column is kept as ``metadata``. A directory may also
    hold a ``manifest.jsonl`` with such rows keyed by ``path`` (relative to the
    directory) to attach titles/metadata to its direct (non-archive) PDFs.

    Returns ``(records, inventory)``. Records are dicts with ``corpus_id, config,
    pdf_url, title, source_path, sha256, size_bytes, metadata`` sorted by ``pdf_url``
    (stable output order). Identical PDFs (same bytes, any folder or archive) are one
    document; the dropped copies are listed in ``inventory.duplicates``. The
    inventory is None for ``.jsonl`` sources.
    """
    from .discover import discover

    source = Path(source)
    recs: List[Dict[str, Any]] = []
    if source.is_dir():
        extra: Dict[str, Dict[str, Any]] = {}
        mpath = source / "manifest.jsonl"
        if mpath.exists():
            for row in jsonl_rows(mpath):
                key = row.get("path") or row.get("file")
                if key:
                    extra[str(Path(key))] = row
        inv = discover(source, unpack_dir, max_archive_depth, hash_cache=hash_cache)
        for f in inv.found:
            row = extra.get(f.origin, {}) if f.container is None else {}
            rec = _local_record(f.path, config, row, hash_cache)
            rec["pdf_url"] = f.pdf_url
            rec["metadata"] = {**rec["metadata"], "origin": f.origin, "folder": f.folder,
                               **({"container": f.container, "archive_depth": f.archive_depth} if f.container else {})}
            recs.append(rec)
    elif source.suffix == ".jsonl":
        inv = None
        for row in jsonl_rows(source):
            ref = row.get("pdf_url") or row.get("path") or row.get("file")
            if not ref:
                raise ValueError(f"{source}: row without pdf_url/path: {row}")
            if ref.startswith(("http://", "https://")):
                recs.append(_url_record(ref, config, row))
            else:
                p = Path(ref)
                if not p.is_absolute():
                    p = source.parent / p
                recs.append(_local_record(p, config, row, hash_cache))
    else:
        raise ValueError(f"source must be a directory of PDFs or a .jsonl manifest: {source}")

    # identical bytes/url listed twice: keep the copy outside an archive, then the first by path
    seen: Dict[str, str] = {}
    out = []
    for r in sorted(recs, key=lambda r: ("container" in r["metadata"], r["pdf_url"])):
        if r["corpus_id"] in seen:
            if inv is not None:
                inv.duplicates.append({"kept": seen[r["corpus_id"]], "dropped": r["pdf_url"], "corpus_id": r["corpus_id"]})
            continue
        seen[r["corpus_id"]] = r["pdf_url"]
        out.append(r)
    out.sort(key=lambda r: r["pdf_url"])
    return out, inv


def _meta(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in _CORE_COLUMNS}


def _local_record(pdf: Path, config: str, row: Dict[str, Any],
                  hash_cache: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    pdf = Path(pdf).resolve()
    digest = sha256_file_cached(pdf, hash_cache)
    return {
        "corpus_id": f"{config}:{digest[:ID_HEX_LEN]}",
        "config": config,
        "pdf_url": pdf.as_uri(),
        "title": row.get("title") or _title_from(pdf.name),
        "source_path": str(pdf),
        "sha256": digest,
        "size_bytes": pdf.stat().st_size,
        "metadata": _meta(row),
    }


def _url_record(url: str, config: str, row: Dict[str, Any]) -> Dict[str, Any]:
    digest = sha256_text(url)
    return {
        "corpus_id": row.get("corpus_id") or f"{config}:{digest[:ID_HEX_LEN]}",
        "config": config,
        "pdf_url": url,
        "title": row.get("title") or _title_from(url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]),
        "source_path": None,
        "sha256": digest,
        "size_bytes": None,
        "metadata": _meta(row),
    }


# ── checkpoints ────────────────────────────────────────────────────────────────


def load_done(path: Path) -> Set[str]:
    """Corpus ids whose last checkpoint row is ``done``."""
    return {cid for cid, st in load_status(path).items() if st["status"] == "done"}


def load_status(path: Path) -> Dict[str, Dict[str, Any]]:
    """Last checkpoint row per corpus id."""
    p = Path(path)
    if not p.exists():
        return {}
    last: Dict[str, Dict[str, Any]] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            last[rec["corpus_id"]] = rec
    return last


def append_checkpoint(
    path: Path, corpus_id: str, status: str, error: Optional[str] = None
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = CheckpointRecord(corpus_id=corpus_id, status=status, error=error)
    with p.open("a") as f:
        f.write(rec.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path: Path, rows: Iterable[str]) -> int:
    """Append pre-serialized JSON lines, flushed to disk. Returns rows written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a") as f:
        for line in rows:
            f.write(line + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    return n


def reconcile_outputs(paths: Iterable[Path], done: Set[str]) -> Dict[str, int]:
    """Drop rows of documents that are not checkpointed ``done``.

    A crash between writing a document's rows and its checkpoint leaves partial
    rows that a resume would duplicate. Files are rewritten atomically.
    """
    removed: Dict[str, int] = {}
    for path in map(Path, paths):
        if not path.exists():
            continue
        keep, dropped = [], 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            if json.loads(line).get("corpus_id") in done:
                keep.append(line)
            else:
                dropped += 1
        if dropped:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text("".join(l + "\n" for l in keep))
            os.replace(tmp, path)
            removed[path.name] = dropped
    return removed
