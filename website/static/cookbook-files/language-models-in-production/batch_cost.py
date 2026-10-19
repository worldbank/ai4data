"""Estimate the cost and time of a batch model job, hosted or local.

Takes the number of records, the average tokens in and out per record,
and the prices or throughput of two options, and prints the cost and
the wall time of each: a hosted model priced per million tokens with a
batch discount, and a local model with a measured throughput on the
organization's hardware. Standard library only.

Usage:
    python batch_cost.py --records 250000 --tokens-in 180 --tokens-out 12 \
        --hosted-in 0.50 --hosted-out 2.00 --batch-discount 0.5 \
        --local-tokens-per-second 900 --local-hourly-cost 1.20

What this does not do: the estimate uses the organization's own token
counts and throughput measurements, which this script takes as input
and does not measure. Review time per record, the larger cost for
reviewed outputs, is added from the production measurements of the
assurance chapter.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument(
        "--tokens-in",
        type=float,
        required=True,
        help="average input tokens per record (prompt, rules, text)",
    )
    parser.add_argument(
        "--tokens-out",
        type=float,
        required=True,
        help="average output tokens per record",
    )
    parser.add_argument(
        "--hosted-in",
        type=float,
        default=0.0,
        help="hosted price per million input tokens",
    )
    parser.add_argument(
        "--hosted-out",
        type=float,
        default=0.0,
        help="hosted price per million output tokens",
    )
    parser.add_argument(
        "--batch-discount",
        type=float,
        default=1.0,
        help="multiplier for a batch endpoint, for example 0.5",
    )
    parser.add_argument(
        "--hosted-records-per-hour",
        type=float,
        default=50000,
        help="rate the hosted limits allow",
    )
    parser.add_argument(
        "--local-tokens-per-second",
        type=float,
        default=0.0,
        help="measured throughput of the local server",
    )
    parser.add_argument(
        "--local-hourly-cost",
        type=float,
        default=0.0,
        help="server cost per hour including operation",
    )
    args = parser.parse_args(argv)

    total_in = args.records * args.tokens_in
    total_out = args.records * args.tokens_out
    print(
        f"{args.records:,} records, {total_in / 1e6:.1f}M input tokens, {total_out / 1e6:.1f}M output tokens\n"
    )
    if args.hosted_in or args.hosted_out:
        cost = (
            total_in / 1e6 * args.hosted_in + total_out / 1e6 * args.hosted_out
        ) * args.batch_discount
        hours = args.records / args.hosted_records_per_hour
        print(
            f"hosted (batch discount {args.batch_discount:.2f}): cost {cost:,.2f}, about {hours:.1f} hours at the allowed rate, {cost / args.records * 1000:.3f} per 1,000 records"
        )
    if args.local_tokens_per_second:
        seconds = (total_in + total_out) / args.local_tokens_per_second
        hours = seconds / 3600
        cost = hours * args.local_hourly_cost
        print(
            f"local ({args.local_tokens_per_second:.0f} tokens/s): {hours:.1f} hours of server time, cost {cost:,.2f}, {cost / args.records * 1000:.3f} per 1,000 records"
        )
    print(
        "\nAdd review time per record from the assurance chapter's measurements; it is usually the larger cost."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
