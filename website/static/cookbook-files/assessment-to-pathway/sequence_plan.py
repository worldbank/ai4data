"""Order the dimensions with a gap into waves by their dependencies.

Reads the question scores, the targets, and the dependency table (one
row per dimension with the dimensions it depends on) and prints the
waves: wave 1 holds the dimensions with a gap that depend on nothing
with a gap, wave 2 those whose dependencies are all in wave 1, and so
on. Within a wave the dimensions are ordered by gap, then by the
number of questions below target. It also reports whether the
framework's four foundations for early-stage organizations (quality
control, metadata standards, a searchable catalog, API access under a
licence) are in the first two waves. Exit status 1 when the
dependencies contain a cycle.

Usage:
    python sequence_plan.py question_scores.csv targets.csv dependencies.csv

Requires pandas, and readiness_profile.py and gap_table.py in the same folder.

What this does not do: it orders dimensions, not recipes. The effort
estimate of the next chapter puts recipes inside each wave, and the
organization's own constraints (a procurement, a recruitment) can move
a dimension later than its wave.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

FOUNDATIONS = {
    "2.2": "quality control",
    "2.1": "metadata standards",
    "2.3": "searchable catalog",
    "2.4": "API access",
    "2.6": "licensing",
}


def load(name: str):
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def waves(table: pd.DataFrame, deps: dict[str, list[str]]) -> list[list[str]]:
    """Dimensions with a gap, in waves; a dimension enters a wave once its gapped dependencies are placed."""
    gapped = {r["dimension"] for _, r in table.iterrows() if (r["gap"] or 0) > 0}
    order = {
        r["dimension"]: (-(r["gap"] or 0), -(r["below_target"] or 0), r["dimension"])
        for _, r in table.iterrows()
    }
    placed: set[str] = set()
    result = []
    remaining = set(gapped)
    while remaining:
        ready = sorted(
            [
                d
                for d in remaining
                if all(x in placed or x not in gapped for x in deps.get(d, []))
            ],
            key=lambda d: order[d],
        )
        if not ready:
            raise ValueError("dependency cycle among " + ", ".join(sorted(remaining)))
        result.append(ready)
        placed.update(ready)
        remaining.difference_update(ready)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scores", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("dependencies", type=Path)
    args = parser.parse_args(argv)
    gt = load("gap_table")
    scores = pd.read_csv(args.scores, dtype=str).fillna("")
    targets = pd.read_csv(args.targets, dtype=str).fillna("")
    dep_df = pd.read_csv(args.dependencies, dtype=str).fillna("")
    deps = {
        r["dimension"]: [d for d in r["depends_on"].split(";") if d]
        for _, r in dep_df.iterrows()
    }
    table = gt.gaps(scores, targets)
    names = dict(zip(table["dimension"], table["name"]))
    info = {r["dimension"]: r for _, r in table.iterrows()}
    try:
        ws = waves(table, deps)
    except ValueError as e:
        print(f"problem: {e}")
        print("result: FAIL")
        return 1
    for i, w in enumerate(ws, 1):
        print(f"wave {i}:")
        for d in w:
            r = info[d]
            waits = [
                x for x in deps.get(d, []) if x in {y for ww in ws[: i - 1] for y in ww}
            ]
            print(
                f"  {d}  {names[d][:44]:<44} {r['current']} -> {r['target']}  (gap {r['gap']}, {r['below_target']} questions below target)"
                + (f"  after {', '.join(waits)}" if waits else "")
            )
    early = {d for w in ws[:2] for d in w}
    found = [
        f"{d} {FOUNDATIONS[d]}"
        for d in FOUNDATIONS
        if d in early or d not in {x for w in ws for x in w}
    ]
    print(
        f"\nfoundations in the first two waves or already at target: {len(found)} of {len(FOUNDATIONS)}"
    )
    for f in found:
        print(f"  {f}")
    print("result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
