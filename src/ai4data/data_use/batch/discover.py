"""Find the PDFs under a messy input tree, including inside archives.

Walks every folder level of the source, matches ``.pdf`` case-insensitively, and
unpacks ``.zip``, ``.tar(.gz|.bz2|.xz)``, ``.tgz`` and single-file ``.gz`` archives
(nested archives down to ``max_archive_depth``). Archives are unpacked into a
content-addressed cache (``<unpack_dir>/<sha256[:16]>/``), so a restart reuses
them. Nothing is ever written outside ``unpack_dir``: members with absolute paths
or ``..`` parts, links and devices are rejected, and size caps stop archive bombs.

The result is an ``Inventory``: the PDFs found (with their origin) plus everything
that was skipped and why, so a run never silently drops input.
"""

import gzip
import logging
import os
import shutil
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from .manifest import sha256_file_cached

log = logging.getLogger("ai4data.batch.discover")

TAR_SUFFIXES = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar")
UNSUPPORTED_ARCHIVES = (".7z", ".rar", ".zst", ".bz2", ".xz", ".z")
GiB = 1 << 30


@dataclass
class Limits:
    max_member_bytes: int = 2 * GiB
    max_total_bytes: int = 50 * GiB
    max_members: int = 200_000


@dataclass
class Found:
    """A PDF on disk (possibly unpacked) and where it came from."""

    path: Path  # file to parse
    origin: str  # path under the source root; ``archive.zip!/inner/a.pdf`` inside archives
    pdf_url: str  # file URI of the original file, ``!/`` joins archive members
    container: Optional[str]  # outermost archive (relative to the source), if any
    archive_depth: int
    folder: str  # first folder level under the source ("" at the top)


@dataclass
class Inventory:
    source: str
    found: List[Found] = field(default_factory=list)
    dirs: int = 0
    files_seen: int = 0
    counts: Counter = field(default_factory=Counter)  # by kind
    ignored_ext: Counter = field(default_factory=Counter)
    problems: List[Dict[str, Any]] = field(default_factory=list)
    duplicates: List[Dict[str, Any]] = field(default_factory=list)
    max_archive_depth_seen: int = 0

    def problem(self, kind: str, path: str, detail: str = "") -> None:
        self.problems.append({"kind": kind, "path": path, "detail": detail})

    def summary(self) -> Dict[str, Any]:
        by_folder = Counter(f.folder or "." for f in self.found)
        return {
            "source": self.source,
            "folders": self.dirs,
            "files_seen": self.files_seen,
            "pdfs_found": len(self.found),
            "pdfs_direct": sum(1 for f in self.found if f.container is None),
            "pdfs_in_archives": sum(1 for f in self.found if f.container is not None),
            "unique_pdfs": len(self.found) - len(self.duplicates),
            "duplicates_dropped": len(self.duplicates),
            "archives": {k: v for k, v in self.counts.items() if k.startswith("archive_")},
            "hidden_skipped": self.counts.get("hidden", 0),
            "ignored_by_extension": dict(self.ignored_ext.most_common(15)),
            "max_archive_depth_seen": self.max_archive_depth_seen,
            "problems": Counter(p["kind"] for p in self.problems),
            "pdfs_by_top_folder": dict(by_folder.most_common(20)),
        }

    def to_json(self) -> Dict[str, Any]:
        s = self.summary()
        s["problems"] = dict(s["problems"])
        return {
            "summary": s,
            "problem_list": self.problems,
            "duplicates": self.duplicates,
            "pdfs": [
                {"origin": f.origin, "pdf_url": f.pdf_url, "container": f.container,
                 "archive_depth": f.archive_depth, "folder": f.folder}
                for f in self.found
            ],
        }

    def report(self) -> str:
        s = self.summary()
        lines = [
            f"source: {s['source']}",
            f"folders={s['folders']} files={s['files_seen']} hidden/system skipped={s['hidden_skipped']}",
            f"PDFs found: {s['pdfs_found']} (direct {s['pdfs_direct']}, in archives {s['pdfs_in_archives']}) "
            f"-> unique {s['unique_pdfs']} after dropping {s['duplicates_dropped']} identical copies",
            f"archives: {s['archives'] or 'none'}  (deepest nesting seen: {s['max_archive_depth_seen']})",
            f"ignored by extension: {s['ignored_by_extension'] or 'none'}",
        ]
        if s["problems"]:
            lines.append(f"problems: {dict(s['problems'])}")
            for p in self.problems[:20]:
                lines.append(f"  - {p['kind']}: {p['path']} {p['detail']}".rstrip())
            if len(self.problems) > 20:
                lines.append(f"  ... {len(self.problems) - 20} more (see inventory.json)")
        lines.append(f"PDFs per top folder: {s['pdfs_by_top_folder']}")
        return "\n".join(lines)


# ── classification ─────────────────────────────────────────────────────────────


def is_hidden(parts) -> bool:
    return any(p.startswith(".") or p == "__MACOSX" or p.startswith("~$") for p in parts)


def classify(name: str) -> str:
    """``pdf`` | ``zip`` | ``tar`` | ``gz`` | ``unsupported_archive`` | ``other``."""
    low = name.lower()
    if low.endswith(".pdf"):
        return "pdf"
    if low.endswith(".zip"):
        return "zip"
    if low.endswith(TAR_SUFFIXES):
        return "tar"
    if low.endswith(".gz"):
        return "gz"
    if low.endswith(UNSUPPORTED_ARCHIVES):
        return "unsupported_archive"
    return "other"


def _ext(name: str) -> str:
    return Path(name).suffix.lower() or "(none)"


def _safe_member(name: str) -> Optional[PurePosixPath]:
    """Relative POSIX path of an archive member, or None if it could escape the folder."""
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or any(part in ("..", "") for part in p.parts) or (p.parts and ":" in p.parts[0]):
        return None
    return p


# ── unpacking ──────────────────────────────────────────────────────────────────


class _Budget:
    def __init__(self, limits: Limits):
        self.limits, self.members, self.total = limits, 0, 0


def _copy_capped(src, dst: Path, budget: _Budget) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(dst, "wb") as out:
        while block := src.read(1 << 20):
            written += len(block)
            budget.total += len(block)
            if written > budget.limits.max_member_bytes:
                raise ValueError(f"member larger than {budget.limits.max_member_bytes} bytes")
            if budget.total > budget.limits.max_total_bytes:
                raise ValueError(f"archive expands beyond {budget.limits.max_total_bytes} bytes")
            out.write(block)


def _wanted(rel: PurePosixPath, inv: Inventory, container: str) -> bool:
    """Members worth extracting: PDFs and nested archives. Counts the rest."""
    if is_hidden(rel.parts):
        inv.counts["hidden"] += 1
        return False
    kind = classify(rel.name)
    if kind == "other":
        inv.ignored_ext[_ext(rel.name)] += 1
        return False
    if kind == "unsupported_archive":
        inv.problem("unsupported_archive", f"{container}!/{rel}", "7z/rar/zst are not supported")
        return False
    return True


def _unpack(archive: Path, kind: str, dest: Path, inv: Inventory, label: str, limits: Limits) -> None:
    budget = _Budget(limits)
    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = _safe_member(info.filename)
                if rel is None:
                    inv.problem("unsafe_member", f"{label}!/{info.filename}", "path escapes the archive; skipped")
                    continue
                if not _wanted(rel, inv, label):
                    continue
                budget.members += 1
                if budget.members > limits.max_members:
                    raise ValueError(f"more than {limits.max_members} members")
                try:
                    with zf.open(info) as src:
                        _copy_capped(src, dest / rel, budget)
                except (RuntimeError, NotImplementedError) as exc:  # encrypted / unsupported compression
                    inv.problem("unreadable_member", f"{label}!/{rel}", str(exc)[:120])
    elif kind == "tar":
        with tarfile.open(archive, "r:*") as tf:
            for m in tf:
                if not m.isfile():  # links, devices, directories
                    if m.issym() or m.islnk() or m.isdev():
                        inv.problem("unsafe_member", f"{label}!/{m.name}", "link/device skipped")
                    continue
                rel = _safe_member(m.name)
                if rel is None:
                    inv.problem("unsafe_member", f"{label}!/{m.name}", "path escapes the archive; skipped")
                    continue
                if not _wanted(rel, inv, label):
                    continue
                budget.members += 1
                if budget.members > limits.max_members:
                    raise ValueError(f"more than {limits.max_members} members")
                src = tf.extractfile(m)
                if src is not None:
                    _copy_capped(src, dest / rel, budget)
    elif kind == "gz":
        inner = archive.name[:-3]
        rel = _safe_member(inner)
        if rel is not None and _wanted(rel, inv, label):
            with gzip.open(archive, "rb") as src:
                _copy_capped(src, dest / rel, budget)


def _unpack_cached(archive: Path, kind: str, unpack_dir: Path, inv: Inventory, label: str,
                   limits: Limits, hash_cache) -> Optional[Path]:
    dest = unpack_dir / sha256_file_cached(archive, hash_cache)[:16]
    if (dest / ".done").exists():
        return dest
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        _unpack(archive, kind, tmp, inv, label, limits)
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError, ValueError, gzip.BadGzipFile) as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        inv.problem("corrupt_archive", label, f"{type(exc).__name__}: {exc}"[:160])
        return None
    (tmp / ".done").write_text(label)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return dest


# ── discovery ──────────────────────────────────────────────────────────────────


def discover(
    source: Path,
    unpack_dir: Optional[Path] = None,
    max_archive_depth: int = 2,
    limits: Optional[Limits] = None,
    hash_cache: Optional[Dict[str, str]] = None,
) -> Inventory:
    """Walk ``source`` and return the PDFs (direct and inside archives) plus a report.

    ``unpack_dir=None`` does not open archives (they are listed as skipped).
    ``max_archive_depth`` bounds archive-in-archive nesting (0 = do not open archives).
    """
    source = Path(source).resolve()
    inv = Inventory(source=str(source))
    limits = limits or Limits()

    def walk(root: Path, origin_prefix: str, uri_prefix: str, container: Optional[str], depth: int):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            rel_dir = Path(dirpath).relative_to(root)
            hidden_dirs = [d for d in dirnames if is_hidden([d])]
            inv.counts["hidden"] += len(hidden_dirs)  # pruned folders count once
            dirnames[:] = sorted(d for d in dirnames if d not in hidden_dirs)
            inv.dirs += 1
            for name in sorted(filenames):
                if name == ".done":  # our own cache marker
                    continue
                rel = rel_dir / name
                if is_hidden(rel.parts):
                    inv.counts["hidden"] += 1
                    continue
                path = Path(dirpath) / name
                inv.files_seen += 1
                origin = origin_prefix + rel.as_posix()
                uri = uri_prefix + rel.as_posix()
                kind = classify(name)
                if kind == "pdf":
                    inv.counts["pdf"] += 1
                    top_folder = _top_folder(origin)
                    inv.found.append(Found(path, origin, _file_uri(source, uri, depth), container, depth, top_folder))
                elif kind in ("zip", "tar", "gz"):
                    inv.counts[f"archive_{kind}"] += 1
                    if unpack_dir is None or depth >= max_archive_depth:
                        inv.problem("archive_not_opened", origin,
                                    "archives disabled" if unpack_dir is None else f"deeper than {max_archive_depth}")
                        continue
                    dest = _unpack_cached(path, kind, Path(unpack_dir), inv, origin, limits, hash_cache)
                    if dest is None:
                        continue
                    inv.max_archive_depth_seen = max(inv.max_archive_depth_seen, depth + 1)
                    walk(dest, origin + "!/", uri + "!/", container or origin, depth + 1)
                elif kind == "unsupported_archive":
                    inv.counts["archive_unsupported"] += 1
                    inv.problem("unsupported_archive", origin, "7z/rar/zst/bz2/xz are not supported")
                else:
                    inv.ignored_ext[_ext(name)] += 1

    walk(source, "", "", None, 0)
    return inv


def _top_folder(origin: str) -> str:
    first = origin.split("!/", 1)[0]
    parts = PurePosixPath(first).parts
    return parts[0] if len(parts) > 1 else ""


def _file_uri(source: Path, rel_uri: str, depth: int) -> str:
    """``file:///abs/source/a/b.zip!/inner.pdf``: the original file URI, ``!/`` into archives."""
    head, sep, tail = rel_uri.partition("!/")
    return (source / head).as_uri() + (sep + tail if sep else "")
