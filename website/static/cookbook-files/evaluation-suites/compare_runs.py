"""Compare two evaluation runs and decide whether a change may ship.

Reads two runs of the same question set (CSV: question_id, language,
slice, pass 0|1) and reports, overall and per language and slice, the
pass rate of each run, the difference, and a bootstrap confidence
interval on the difference over the paired questions. The gate rule:
the change ships when no slice's interval lies entirely below zero
minus the tolerated regression (exit 1 otherwise). A slice that dropped
beyond the tolerance on too few questions for the interval to decide
sends the change to a person (exit 2). Standard library only; the
bootstrap uses a fixed seed so that the result is reproducible.

Uses pandas and scipy (percentile bootstrap).

Usage:
    python compare_runs.py run_scores_v1.csv run_scores_v2.csv --tolerance 0.05

What this does not do: thirty questions give wide intervals, which is
the point: the script shows how little thirty questions can decide. A
slice that the set does not cover (a language with two questions) is
not protected by the gate, and the question chapter is where the set
grows.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def interval(diffs: pd.Series, seed: int = 7) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean difference."""
    if (
        diffs.nunique() == 1
    ):  # bootstrap needs variation; a constant difference has none
        return float(diffs.iloc[0]), float(diffs.iloc[0])
    res = bootstrap(
        (diffs.to_numpy(),),
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
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="regression tolerated on any slice",
    )
    args = parser.parse_args(argv)
    a = pd.read_csv(args.before, dtype={"question_id": str})
    b = pd.read_csv(args.after, dtype={"question_id": str})
    paired = a.merge(
        b[["question_id", "pass"]], on="question_id", suffixes=("_before", "_after")
    )
    paired["diff"] = paired["pass_after"] - paired["pass_before"]
    groups = pd.concat(
        [
            paired.assign(group="all"),
            paired.assign(group="language=" + paired["language"]),
            paired.assign(group="slice=" + paired["slice"]),
        ]
    )

    print(f"{len(paired)} paired questions; tolerance {args.tolerance:.2f}\n")
    print(
        f"{'group':<18} {'n':>3} {'before':>7} {'after':>6} {'diff':>6} {'95% interval':>16}  verdict"
    )
    blocking: list[str] = []
    review: list[str] = []
    for name, g in groups.groupby("group", sort=False):
        before, after = g["pass_before"].mean(), g["pass_after"].mean()
        lo, hi = interval(g["diff"])
        if hi < -args.tolerance:
            verdict = "REGRESSION"
            blocking.append(name)
        elif after - before < -args.tolerance:
            verdict = "possible regression, too few to decide"
            review.append(name)
        else:
            verdict = "ok (small)" if len(g) < 30 else "ok"
        print(
            f"{name:<18} {len(g):>3} {before:>7.2f} {after:>6.2f} {after - before:>+6.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>16}  {verdict}"
        )
    print()
    if blocking:
        print(f"gate: BLOCKED by {', '.join(blocking)}")
        return 1
    if review:
        print(
            f"gate: REVIEW ({', '.join(review)} dropped beyond the tolerance on too few questions to decide; a person decides, and the set grows there)"
        )
        return 2
    print(
        "gate: PASS (no slice shows a regression beyond the tolerance; small slices show direction only)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
