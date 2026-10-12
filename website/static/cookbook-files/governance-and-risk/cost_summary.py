"""Summarize the cost record per component and per provider.

Reads the cost record (one row per component and month, with the
provider, the number of requests, and the cost) and prints the cost of
each component by month with the change from the previous month, the
share of the total that goes to hosted providers, and the components
whose cost rose by more than half from one month to the next. Exit
status is always 0; the summary is for the board, which decides.

Usage:
    python cost_summary.py cost_record.csv providers.csv

Requires pandas.

What this does not check: staff time, which is the largest cost of every
component and is not in the record. The investment view of the cookbook
series gives it per recipe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("costs", type=Path)
    parser.add_argument("providers", type=Path)
    args = parser.parse_args(argv)

    costs = pd.read_csv(
        args.costs, dtype={"component_id": str, "month": str, "provider_id": str}
    )
    providers = pd.read_csv(args.providers, dtype=str).fillna("")
    kind = dict(zip(providers["provider_id"], providers["kind"]))
    costs["kind"] = costs["provider_id"].map(kind).fillna("unknown")

    months = sorted(costs["month"].unique())
    table = costs.pivot_table(
        index="component_id", columns="month", values="cost_usd", aggfunc="sum"
    ).fillna(0)
    print("cost per component (USD):")
    print(f"  {'id':<4} " + " ".join(f"{m:>8}" for m in months) + "   change")
    rises = []
    for cid, row in table.iterrows():
        vals = [row.get(m, 0) for m in months]
        change = ""
        if len(vals) >= 2 and vals[-2] > 0:
            pct = (vals[-1] - vals[-2]) / vals[-2] * 100
            change = f"{pct:+.0f}%"
            if pct > 50:
                rises.append((cid, pct))
        print(f"  {cid:<4} " + " ".join(f"{v:>8.0f}" for v in vals) + f"   {change}")

    total = costs.groupby("month")["cost_usd"].sum()
    hosted = costs[costs["kind"] == "hosted"].groupby("month")["cost_usd"].sum()
    print("\nshare of cost at hosted providers:")
    for m in months:
        t = total.get(m, 0)
        h = hosted.get(m, 0)
        print(f"  {m}: {h:.0f} of {t:.0f} ({(h / t * 100) if t else 0:.0f}%)")

    print("\nrequests at hosted providers by component:")
    for cid, n in (
        costs[costs["kind"] == "hosted"]
        .groupby("component_id")["requests"]
        .sum()
        .items()
    ):
        print(f"  {cid}: {n}")

    if rises:
        print("\ncost rose by more than half from the previous month:")
        for cid, pct in rises:
            print(f"  {cid}: {pct:+.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
