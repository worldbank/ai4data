"""Validate and display one Data Snapshot metadata JSON record.

Usage: python inspect_metadata.py metadata.json
Returns exit status 0 for valid metadata and 1 for unreadable or invalid input.
Requires ai4data[data-snapshot-core]. This does not check factual accuracy,
completeness, source grounding, or crop quality, and makes no network requests.
"""

import argparse
from pathlib import Path

from pydantic import ValidationError

from ai4data.data_snapshot import DataSnapshotMetadata


def main(argv: list[str] | None = None) -> int:
    """Validate a local metadata record and print its populated fields.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command arguments, or the process arguments when omitted.

    Returns
    -------
    int
        Zero after successful validation.

    Raises
    ------
    SystemExit
        If arguments are invalid or the input cannot be read or validated.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args(argv)
    try:
        record = DataSnapshotMetadata.model_validate_json(
            args.metadata.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        parser.exit(1, f"Invalid metadata: {error}\n")
    print(record.model_dump_json(indent=2, exclude_none=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
