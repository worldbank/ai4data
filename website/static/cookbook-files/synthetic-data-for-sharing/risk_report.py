"""Measure the disclosure risk of a synthetic table against the real one.

Reads the real and the synthetic CSV (same columns) and reports three
checks that a release of synthetic microdata needs:

    exact copies      synthetic records identical to a real record on
                      every column (a synthesizer that memorized)
    closest record    for each synthetic record the normalized distance
                      to its closest real record, compared with the same
                      distance computed between two halves of the real
                      data; synthetic records much closer to real ones
                      than real records are to each other indicate
                      copying
    attribute inference  how well a sensitive attribute (income band) of
                      real records can be predicted from the
                      quasi-identifiers through the synthetic data (a
                      nearest-neighbour lookup), compared with a baseline
                      that always predicts the most common band

Uses pandas and scikit-learn (nearest neighbours). Quasi-identifiers and the sensitive attribute are
arguments; the defaults match the running example.

Usage:
    python risk_report.py real_sample.csv synthetic_sample.csv \
        --quasi region sex age educ --sensitive income

What this does not do: three checks on three hundred records are a
demonstration of the kinds of evidence a release needs. The statistical
disclosure control unit sets the thresholds and may require membership
inference tests and a formal privacy accounting, which the risk chapter
points to. Removing exact copies is the minimum; a synthesizer that
produced them needs a parameter change and a rerun.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def band(value: float) -> str:
    if pd.isna(value):
        return "none"
    return "low" if value < 500 else "mid" if value < 1200 else "high"


def encoder(real: pd.DataFrame, quasi: list[str]) -> ColumnTransformer:
    """Scale numeric quasi-identifiers to [0, 1] and one-hot categoricals (weighted so a mismatch counts 1), fitted on the real file."""
    numeric = [
        q
        for q in quasi
        if pd.to_numeric(real[q], errors="coerce").notna().all()
        and real[q].nunique() > 12
    ]
    categorical = [q for q in quasi if q not in numeric]
    return ColumnTransformer(
        [
            ("num", MinMaxScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        sparse_threshold=0,
    ).fit(real[quasi])


def distances(
    enc: ColumnTransformer,
    quasi: list[str],
    query: pd.DataFrame,
    reference: pd.DataFrame,
) -> np.ndarray:
    """Distance from each query record to its closest reference record, averaged over the quasi-identifiers (0 identical, 1 different on every one)."""
    weights = np.array(
        [1.0] * len(enc.transformers_[0][2])
        + [0.5]
        * (enc.transform(reference[quasi]).shape[1] - len(enc.transformers_[0][2]))
    )
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(
        enc.transform(reference[quasi]) * weights
    )
    d, _ = nn.kneighbors(enc.transform(query[quasi]) * weights)
    return d[:, 0] / len(quasi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("real")
    parser.add_argument("synthetic")
    parser.add_argument("--quasi", nargs="+", default=["region", "sex", "age", "educ"])
    parser.add_argument("--sensitive", default="income")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    real, synth = (
        pd.read_csv(p, dtype=str, keep_default_na=False)
        for p in (args.real, args.synthetic)
    )

    copy_rows = synth.index[
        synth.apply(tuple, axis=1).isin(set(real.apply(tuple, axis=1)))
    ].tolist()
    print(f"real {len(real)} rows, synthetic {len(synth)} rows")
    print(
        f"\nexact copies: {len(copy_rows)} synthetic record(s) identical to a real record"
        + (f" (rows {', '.join(str(i + 1) for i in copy_rows)})" if copy_rows else "")
    )

    for q in args.quasi:
        real[q] = (
            pd.to_numeric(real[q])
            if real[q].str.fullmatch(r"-?\d+(\.\d+)?").all()
            else real[q]
        )
        synth[q] = (
            pd.to_numeric(synth[q])
            if synth[q].str.fullmatch(r"-?\d+(\.\d+)?").all()
            else synth[q]
        )
    enc = encoder(real, args.quasi)
    half_a = real.sample(frac=0.5, random_state=args.seed)
    half_b = real.drop(half_a.index)
    d_syn = distances(enc, args.quasi, synth, real)
    d_real = distances(enc, args.quasi, half_a, half_b)
    print("\nclosest record distance on the quasi-identifiers (0 = identical)")
    print(
        f"  synthetic to real:      median {np.median(d_syn):.3f}, share at 0: {(d_syn < 1e-9).mean():.2f}"
    )
    print(
        f"  real half to other half: median {np.median(d_real):.3f}, share at 0: {(d_real < 1e-9).mean():.2f}"
    )
    print(
        "  a synthetic file much closer to the real one than the real halves are to each other has copied records"
    )

    bands_real = pd.to_numeric(real[args.sensitive], errors="coerce").map(band)
    bands_synth = pd.to_numeric(synth[args.sensitive], errors="coerce").map(band)
    weights = np.array(
        [1.0] * len(enc.transformers_[0][2])
        + [0.5]
        * (enc.transform(synth[args.quasi]).shape[1] - len(enc.transformers_[0][2]))
    )
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(
        enc.transform(synth[args.quasi]) * weights
    )
    _, idx = nn.kneighbors(enc.transform(real[args.quasi]) * weights)
    accuracy = (bands_synth.iloc[idx[:, 0]].to_numpy() == bands_real.to_numpy()).mean()
    majority = bands_real.value_counts(normalize=True).iloc[0]
    print(
        f"\nattribute inference of {args.sensitive} band from the quasi-identifiers through the synthetic file"
    )
    print(
        f"  nearest-neighbour accuracy {accuracy:.2f} vs majority baseline {majority:.2f}"
    )
    print(
        "  an accuracy far above the baseline means the synthetic file reveals the attribute for people like the quasi-identifiers; the disclosure control unit sets the limit"
    )
    return 1 if copy_rows else 0


if __name__ == "__main__":
    sys.exit(main())
