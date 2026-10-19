"""Check a list of URLs against a site's robots.txt before collecting them.

Reads a robots.txt file and a CSV of URLs (url, source, document_type)
and reports, for the crawler's user agent, whether each URL may be
fetched, and the crawl delay or request rate the site asks for. Uses the
standard library's robots.txt parser (RFC 9309 rules: the most specific
user-agent group applies, the longest matching path wins). Exit status 1
when any URL is disallowed, so that a pipeline stops before fetching it.

Uses pandas.

Usage:
    python robots_check.py robots_example.txt web_sources.csv --agent ai4data-monitor

What this does not do: robots.txt states the site owner's rules for
crawlers; the site's terms of use and the content licence are separate
and are recorded in the sources list. A site that allows crawling still
expects identification (a user agent with a contact URL) and a rate the
delay sets.
"""

from __future__ import annotations

import argparse
import sys
import urllib.robotparser
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("robots", type=Path)
    parser.add_argument("urls")
    parser.add_argument(
        "--agent", default="ai4data-monitor", help="the crawler's user-agent token"
    )
    args = parser.parse_args(argv)
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(args.robots.read_text(encoding="utf-8").splitlines())
    urls = pd.read_csv(args.urls, dtype=str).fillna("")

    delay, rate = rp.crawl_delay(args.agent), rp.request_rate(args.agent)
    print(
        f"user agent {args.agent!r}: crawl delay {delay if delay is not None else 'none stated'}"
        + (f", request rate {rate.requests}/{rate.seconds}s" if rate else "")
    )
    urls["allowed"] = urls["url"].map(lambda u: rp.can_fetch(args.agent, u))
    for u in urls.itertuples():
        print(
            f"{'allowed ' if u.allowed else 'DISALLOWED'}  {u.url}  ({u.document_type})"
        )
    if rp.site_maps():
        print("sitemaps: " + ", ".join(rp.site_maps()))
    disallowed = int((~urls["allowed"]).sum())
    print(f"\n{len(urls)} URLs, {disallowed} disallowed")
    return 1 if disallowed else 0


if __name__ == "__main__":
    sys.exit(main())
