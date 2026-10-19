"""Audit an agent interface's request log against the client registry.

Reads the client registry (CSV: client_id, name, tier, requests_per_minute,
daily_quota, terms_accepted, contact) and a request log (CSV: timestamp,
client_id, tool, status) and reports, per client, the requests in the
period, the busiest minute against the per-minute limit, the share of
requests the server rejected (status 429 or 401), and two policy checks:
clients in the log that the registry does not know, and registered
clients that have not accepted the terms. Uses pandas.

Usage:
    python quota_audit.py client_registry.csv agent_requests.csv

Exit status 1 when an unknown client made requests or a registered client
without accepted terms did, so that the audit can run as a periodic check.

What this does not do: it audits a log after the fact. The limits
themselves are enforced by the server or a gateway in front of it; the
audit shows whether the limits, the tiers, and the registry match what
happens. Quotas protect the service; the terms protect the data's
attribution, and both are stated to clients at registration.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

REJECTED = {"429", "401", "403"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("registry")
    parser.add_argument("log")
    args = parser.parse_args(argv)
    registry = pd.read_csv(args.registry, dtype=str).fillna("").set_index("client_id")
    log = pd.read_csv(args.log, dtype=str)
    log["minute"] = log["timestamp"].str[:16]

    per_client = log.groupby("client_id").agg(
        requests=("tool", "size"),
        busiest=("minute", lambda m: m.value_counts().max()),
        rejected=("status", lambda s: s.isin(REJECTED).sum()),
    )
    per_client = per_client.join(registry, how="left").sort_values(
        "requests", ascending=False, kind="stable"
    )

    print(
        f"{len(log)} requests from {len(per_client)} clients; registry lists {len(registry)}\n"
    )
    print(
        f"{'client':<10} {'tier':<11} {'requests':>8} {'busiest min':>11} {'limit/min':>9} {'rejected':>8}  note"
    )
    unknown, no_terms = [], []
    for client, r in per_client.iterrows():
        if pd.isna(r["tier"]):
            unknown.append(client)
            print(
                f"{client:<10} {'unknown':<11} {r.requests:>8} {r.busiest:>11} {'-':>9} {r.rejected:>8}  not in the registry"
            )
            continue
        limit = int(r["requests_per_minute"])
        notes = []
        if r.busiest > limit:
            notes.append(f"busiest minute exceeds limit ({r.busiest} > {limit})")
        if r["tier"] != "public" and not r["terms_accepted"]:
            no_terms.append(client)
            notes.append("terms not accepted")
        print(
            f"{client:<10} {r['tier']:<11} {r.requests:>8} {r.busiest:>11} {limit:>9} {r.rejected:>8}  {'; '.join(notes)}"
        )

    print()
    if unknown:
        print(
            f"unknown clients: {', '.join(unknown)} (rejected by the server; check for a leaked or revoked key)"
        )
    if no_terms:
        print(
            f"registered without accepted terms: {', '.join(no_terms)} (suspend until the terms are accepted)"
        )
    served = log.loc[log["status"] == "200", "tool"].value_counts()
    print("tool calls served: " + ", ".join(f"{t} {n}" for t, n in served.items()))
    return 1 if unknown or no_terms else 0


if __name__ == "__main__":
    sys.exit(main())
