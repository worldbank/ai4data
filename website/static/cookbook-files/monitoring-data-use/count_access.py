"""Count views, downloads, and API calls per dataset the way the COUNTER
Code of Practice for Research Data counts them.

Reads an access log (CSV: timestamp, dataset_id, action view|download|api,
ip_hash, user_agent, status) and applies three rules before counting:

    robots       requests from known crawlers are excluded entirely
    double-click repeated requests by the same client for the same
                 dataset and action within 30 seconds count once
    status       only successful requests (status 2xx) count

Counts, per dataset and month, the regular requests (browsers), the
machine requests (API clients and scripted tools, reported separately as
the Code requires), and the unique investigations (distinct clients that
touched the dataset). Uses pandas.

Usage:
    python count_access.py access_log.csv

What this does not do: a hashed IP is a weak client identifier (shared
networks, changing addresses); the Code allows a session cookie or a
user identifier where the catalog has one. The robot list here is short;
the Code points to a maintained list. Counts of files are not counts of
users or uses; the use report presents them beside the mention counts.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

ROBOTS = (
    "googlebot",
    "bingbot",
    "yandex",
    "baiduspider",
    "duckduckbot",
    "crawler",
    "spider",
    "slurp",
)
MACHINE = (
    "python-requests",
    "curl/",
    "wget/",
    "httpx",
    "java/",
    "go-http-client",
    "agent/",
)
DOUBLE_CLICK = pd.Timedelta(seconds=30)


def kind(user_agent: str, action: str) -> str:
    ua = user_agent.lower()
    if any(r in ua for r in ROBOTS):
        return "robot"
    if action == "api" or any(m in ua for m in MACHINE):
        return "machine"
    return "regular"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log")
    args = parser.parse_args(argv)
    log = pd.read_csv(args.log, dtype=str)
    log["when"] = pd.to_datetime(log["timestamp"], utc=True)
    log = log.sort_values("when", kind="stable")

    ok = log["status"].str.startswith("2")
    log["kind"] = [kind(ua, a) for ua, a in zip(log["user_agent"], log["action"])]
    kept = log[ok & (log["kind"] != "robot")].copy()
    # a repeat of the same request from the same client within 30 seconds is a double-click
    gap = kept.groupby(["ip_hash", "dataset_id", "action"])["when"].diff()
    double = gap.notna() & (gap <= DOUBLE_CLICK)
    counted = kept[~double].copy()
    counted["month"] = counted["timestamp"].str[:7]

    print(
        f"{len(log)} log rows; excluded: robots {(log['kind'] == 'robot').sum()}, double-clicks {double.sum()}, failed requests {(~ok).sum()}\n"
    )
    print(
        f"{'dataset':<22} {'month':<8} {'regular':>8} {'views':>6} {'downl.':>6} {'machine':>8} {'api':>4} {'clients':>8}"
    )
    for (dataset, month), g in counted.groupby(["dataset_id", "month"]):
        regular, machine = g[g["kind"] == "regular"], g[g["kind"] == "machine"]
        print(
            f"{dataset:<22} {month:<8} {len(regular):>8} {(regular['action'] == 'view').sum():>6} {(regular['action'] == 'download').sum():>6} "
            f"{len(machine):>8} {(machine['action'] == 'api').sum():>4} {g['ip_hash'].nunique():>8}"
        )
    print(
        "\nRegular and machine requests are reported separately; neither counts users or uses."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
