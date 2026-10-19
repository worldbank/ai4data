"""Compare a model's quality scores with curators' scores.

Reads two CSV files with the same columns (record_id and one column per
quality dimension, scores 1 to 5): the model's scores and the curators'
scores on the same records. Reports, per dimension, the exact agreement,
the agreement within one point, and the mean difference (model minus
curator). A dimension whose mean difference is 0.5 or more in either
direction is marked as miscalibrated: the model is systematically more or
less severe than the curators. Uses pandas.

Usage:
    python assess_agreement.py model_scores.csv curator_scores.csv

What this does not do: five records give a demonstration. A calibration
needs 50 to 100 records scored by two curators, with their own agreement
reported (Cohen's kappa or similar) so that the model is compared with a
reliable reference.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

THRESHOLD = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model")
    parser.add_argument("curator")
    args = parser.parse_args(argv)
    model = pd.read_csv(args.model, index_col="record_id")
    curator = pd.read_csv(args.curator, index_col="record_id")
    records = sorted(set(model.index) & set(curator.index))
    if not records:
        sys.exit("no records in common")
    dimensions = [d for d in model.columns if d in curator.columns]
    m, c = model.loc[records, dimensions], curator.loc[records, dimensions]

    exact = (m == c).mean()
    within = ((m - c).abs() <= 1).mean()
    diff = (m - c).mean()
    print(f"{len(records)} records, {len(dimensions)} dimensions\n")
    print(f"{'dimension':<22} {'exact':>6} {'within 1':>9} {'mean diff':>10}  note")
    for d in dimensions:
        note = (
            ""
            if abs(diff[d]) < THRESHOLD
            else ("model more severe" if diff[d] < 0 else "model more lenient")
        )
        print(f"{d:<22} {exact[d]:>6.2f} {within[d]:>9.2f} {diff[d]:>+10.2f}  {note}")
    print(
        f"\n{int((diff.abs() >= THRESHOLD).sum())} dimension(s) miscalibrated at a mean difference of {THRESHOLD} or more"
    )
    print(
        "Scores are compared with the curators' scores as the reference; the curators' own agreement is reported separately."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
