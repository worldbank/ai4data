"""Produce a report card for an evaluation run with intervals.

Reads a run (CSV: question_id, language, slice, pass 0|1) and prints,
overall and per language and slice, the pass rate with a 95 percent
bootstrap interval and the number of questions, in a form that can be
pasted into a release note or compared across organizations that use the
same question set format. Standard library only; fixed seed.

Uses pandas and scipy (percentile bootstrap).

Usage:
    python report_card.py run_scores_v2.csv --run-name "search v2, 2026-10-01"

What this does not do: a report card describes one run on one question
set. Comparability across organizations needs the same set, or sets
built with the same rules, and the same measure definitions; the
reporting chapter states what to publish with the numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def interval(passes: pd.Series, seed: int = 7) -> tuple[float, float]:
    """95% percentile bootstrap interval for a pass rate."""
    if passes.nunique() == 1:
        return float(passes.iloc[0]), float(passes.iloc[0])
    res = bootstrap(
        (passes.to_numpy(),),
        np.mean,
        confidence_level=0.95,
        method="percentile",
        random_state=seed,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args(argv)
    df = pd.read_csv(args.run)
    groups = pd.concat(
        [
            df.assign(group="all"),
            df.assign(group="language=" + df["language"]),
            df.assign(group="slice=" + df["slice"]),
        ]
    )

    print(f"Report card: {args.run_name or Path(args.run).name}\n")
    print(f"{'group':<18} {'n':>3} {'pass rate':>9} {'95% interval':>14}")
    for name, g in groups.groupby("group", sort=False):
        lo, hi = interval(g["pass"])
        print(
            f"{name:<18} {len(g):>3} {g['pass'].mean():>9.2f} {f'[{lo:.2f}, {hi:.2f}]':>14}"
        )
    print(
        "\nPublish with: the question set version, the measure definitions, the model and index versions, and the date."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
