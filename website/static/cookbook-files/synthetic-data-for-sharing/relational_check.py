"""Check a synthetic household-person pair of tables for structural validity.

Reads the household table (CSV: hhid, region, size) and the person table
(CSV: hhid, pid, relationship 1=head 2=spouse 3=child 4=other, age, sex)
of a synthetic release, and optionally the real pair, and reports:

    integrity   persons whose household does not exist; households with
                no persons; households whose size differs from their
                person count
    structure   households with no head or more than one; children older
                than the head; spouses more than 25 years apart from the head
    distributions  household size distribution and persons per household
                compared with the real pair when given (total variation
                distance)

Exit status 1 when any integrity or structure fault is found. Uses pandas.

Usage:
    python relational_check.py households_synth.csv persons_synth.csv \
        --real-households households_real.csv --real-persons persons_real.csv

What this does not do: the rules here are the ones every household
survey has; the organization adds its own (age at marriage, school age,
labour force status by age). A relational synthesizer that models the
parent-child structure (REaLTabFormer, HMA) usually passes integrity and
fails some structure rules, which is what the rules are for.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def tvd(a: pd.Series, b: pd.Series) -> float:
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()


def faults_in(hh: pd.DataFrame, pp: pd.DataFrame) -> list[str]:
    """Integrity (every person has a household, every household has persons, sizes match) and structure (one head, ages)."""
    faults = []
    members = pp.groupby("hhid")
    counts = members.size()
    faults += [
        f"person {p.hhid}/{p.pid} belongs to no household"
        for p in pp[~pp["hhid"].isin(hh["hhid"])].itertuples()
    ]
    faults += [
        f"household {h.hhid} has no persons"
        for h in hh[~hh["hhid"].isin(pp["hhid"])].itertuples()
    ]
    sized = hh[hh["hhid"].isin(counts.index)]
    faults += [
        f"household {h.hhid} size {h.size} but {counts[h.hhid]} persons"
        for h in sized.itertuples()
        if int(h.size) != counts[h.hhid]
    ]
    for hid, m in members:
        heads = m[m["relationship"] == "1"]
        if len(heads) != 1:
            faults.append(f"household {hid} has {len(heads)} head(s)")
            continue
        head_age = int(heads["age"].iloc[0])
        ages = m["age"].astype(int)
        faults += [
            f"household {hid}: child {p.pid} aged {p.age} is not younger than the head ({head_age})"
            for p in m[(m["relationship"] == "3") & (ages >= head_age)].itertuples()
        ]
        faults += [
            f"household {hid}: spouse {p.pid} and head differ by more than 25 years"
            for p in m[
                (m["relationship"] == "2") & ((ages - head_age).abs() > 25)
            ].itertuples()
        ]
    return faults


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("households")
    parser.add_argument("persons")
    parser.add_argument("--real-households")
    parser.add_argument("--real-persons")
    args = parser.parse_args(argv)
    hh, pp = (pd.read_csv(p, dtype=str) for p in (args.households, args.persons))

    faults = faults_in(hh, pp)
    print(f"{len(hh)} households, {len(pp)} persons")
    if faults:
        print(f"\n{len(faults)} fault(s):")
        for f in faults:
            print(f"  {f}")
    else:
        print("\nno integrity or structure faults")
    if args.real_households and args.real_persons:
        rhh, rpp = (
            pd.read_csv(p, dtype=str) for p in (args.real_households, args.real_persons)
        )
        sizes = tvd(rhh["size"], hh["size"])
        persons = tvd(
            rpp.groupby("hhid").size().astype(str),
            pp.groupby("hhid").size().astype(str),
        )
        print(
            f"\nhousehold size distribution: TVD {sizes:.3f}; persons per household: TVD {persons:.3f}"
        )
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main())
