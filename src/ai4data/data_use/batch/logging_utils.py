"""Logging and traceability helpers: text log, JSONL event log, progress/ETA."""

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER_NAME = "ai4data.batch"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(stage: str) -> str:
    return f"{stage}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def fmt_secs(s: float) -> str:
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def setup_logging(log_dir: Optional[Path], run_id: str, verbose: bool = False) -> logging.Logger:
    """Log to stderr and, if ``log_dir`` is given, to ``<log_dir>/<run_id>.log``."""
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{run_id}.log")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


class EventLog:
    """Append-only JSONL of per-document events (machine-readable trace).

    Every row carries ``run_id``, ``stage``, ``corpus_id`` and a timestamp, so a
    row of any output file can be traced to the run, timings and errors.
    """

    def __init__(self, path: Path, run_id: str, stage: str):
        self.path = Path(path)
        self.run_id = run_id
        self.stage = stage
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        row: Dict[str, Any] = {
            "ts": utcnow(),
            "run_id": self.run_id,
            "stage": self.stage,
            "event": event,
            **fields,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())


class Progress:
    """Counts documents and estimates the time left."""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.failed = 0
        self.t0 = time.monotonic()

    def step(self, ok: bool = True) -> None:
        if ok:
            self.done += 1
        else:
            self.failed += 1

    @property
    def processed(self) -> int:
        return self.done + self.failed

    def line(self) -> str:
        el = time.monotonic() - self.t0
        rate = self.processed / el * 60 if el > 0 and self.processed else 0.0
        remaining = max(0, self.total - self.processed)
        eta = remaining / (self.processed / el) if el > 0 and self.processed else 0.0
        return (
            f"{self.processed}/{self.total} docs (ok={self.done} failed={self.failed}) "
            f"{rate:.1f} docs/min elapsed={fmt_secs(el)} eta={fmt_secs(eta) if self.processed else '?'}"
        )
