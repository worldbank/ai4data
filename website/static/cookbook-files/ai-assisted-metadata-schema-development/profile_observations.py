"""Aggregate a small observation excerpt using exact metadata field names.

Usage: python profile_observations.py geographic_observations.json --output outputs/geographic_profiles.json
Uses only the standard library. The input is a JSON array of observations.
Output is created exclusively; an existing file is never replaced. Exit status
is 0 on success or 2 for invalid input or an unavailable output destination.
What this does not check: semantic equivalence, observation correctness, sample
representativeness, concept coverage, or any full-study result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def aggregate_observations(observations: list[dict]) -> list[dict]:
    """Group exact names and retain counts, denominators, and all input evidence.

    Parameters
    ----------
    observations : list[dict]
        Nonempty observations containing snapshot_id, metadata_field,
        observed_value, description, source_level, and reasoning.

    Returns
    -------
    list[dict]
        Profiles sorted by exact field name, with occurrence count, distinct
        snapshot count, total input snapshots, support, and unchanged evidence.

    Raises
    ------
    TypeError
        If an observation is not an object.
    ValueError
        If the input is empty or malformed, required text is absent, or a
        source level is outside snapshot, document, and both.
    """
    if not isinstance(observations, list) or not observations:
        raise ValueError("A nonempty observation array is required.")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in observations:
        if not isinstance(row, dict):
            raise TypeError("Each observation must be an object.")
        for key in [
            "snapshot_id",
            "metadata_field",
            "observed_value",
            "description",
            "source_level",
            "reasoning",
        ]:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing observation value: {key}")
        if row["source_level"] not in {"snapshot", "document", "both"}:
            raise ValueError("Unknown observation source level.")
        groups[row["metadata_field"]].append(row)
    total_snapshots = len({row["snapshot_id"] for row in observations})
    profiles = []
    for field, evidence in sorted(groups.items()):
        n_snapshots = len({row["snapshot_id"] for row in evidence})
        profiles.append(
            {
                "metadata_field": field,
                "count": len(evidence),
                "n_snapshots": n_snapshots,
                "total_snapshots": total_snapshots,
                "support": n_snapshots / total_snapshots,
                "evidence": evidence,
            }
        )
    return profiles


def main(argv: list[str] | None = None) -> int:
    """Write new profiles to an explicit destination and print their counts.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments; None reads the process arguments.

    Returns
    -------
    int
        Zero on success; invalid input or a write failure exits with status 2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        observations = json.loads(args.observations.read_text(encoding="utf-8"))
        profiles = aggregate_observations(observations)
        payload = json.dumps(profiles, ensure_ascii=False, indent=2) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(
        f"observations: {len(observations)}; "
        f"snapshots: {profiles[0]['total_snapshots']}; "
        f"exact-name profiles: {len(profiles)}"
    )
    for profile in profiles:
        print(
            f"{profile['metadata_field']}: count={profile['count']}, "
            f"n_snapshots={profile['n_snapshots']}, "
            f"support={profile['support']:.3f}"
        )
    print(f"Wrote {args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
