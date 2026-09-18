"""
Loading catalogue records and the deterministic train/eval split.

Records can come from a directory of per-record JSON files (``records_glob``) or
a single JSON-Lines file (``records_file``). Each record is an arbitrary dict;
the only requirement is a unique id at ``config.id_field``.

The split is by a hash of the record id, so a record is always on the same side
regardless of order or machine, and the evaluation set is disjoint from training
at the record level (this is what makes the held-out metrics meaningful).
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

from .config import Config
from .serialize import _get_path


def load_records(config: Config) -> dict[str, dict]:
    """Return ``{record_id: record}`` for the whole catalogue."""
    records: dict[str, dict] = {}
    # Relative catalogue paths resolve against the config file's directory, so
    # the toolkit works regardless of the current working directory.
    base = config.path.parent if config.path else Path(".")
    if config.records_file:
        path = Path(config.records_file)
        if not path.is_absolute():
            path = base / path
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[str(_get_path(r, config.id_field))] = r
    elif config.records_glob:
        pattern = config.records_glob
        if not Path(pattern).is_absolute():
            pattern = str(base / pattern)
        for fp in sorted(glob.glob(pattern)):
            r = json.loads(Path(fp).read_text())
            records[str(_get_path(r, config.id_field))] = r
    else:
        raise ValueError("config.catalogue needs records_glob or records_file")
    if not records:
        raise ValueError("no records loaded; check catalogue.records_glob / records_file")
    return records


def is_eval(record_id: str, eval_fraction: float) -> bool:
    """Stable hash-based membership test for the held-out split."""
    bucket = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % 1000
    return bucket < int(eval_fraction * 1000)


def split_ids(record_ids, eval_fraction: float) -> tuple[list[str], list[str]]:
    train = [i for i in record_ids if not is_eval(i, eval_fraction)]
    ev = [i for i in record_ids if is_eval(i, eval_fraction)]
    return train, ev
