"""Estimate the memory a model needs and whether it fits the hardware.

Takes the parameter count, the quantization (bits per weight), the
context length, and the number of concurrent sequences, and estimates:
weight memory, key-value cache memory (from layers, hidden size, and
context), and a working margin; then says whether the total fits the
GPU memory or the system RAM given. Standard library only.

Usage:
    python hardware_sizing.py --params-b 7 --bits 4 --layers 32 --hidden 4096 \
        --context 4096 --concurrency 8 --gpu-gb 24

What this does not do: the estimate uses standard formulas and a margin;
the server's actual use depends on its implementation, attention
variant (grouped-query attention reduces the cache), and batching. The
measured throughput and memory of a test run replace the estimate before
procurement.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--params-b", type=float, required=True, help="parameters in billions"
    )
    parser.add_argument(
        "--bits", type=float, default=16, help="bits per weight: 16, 8, 4"
    )
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True, help="hidden size")
    parser.add_argument(
        "--kv-heads-ratio",
        type=float,
        default=1.0,
        help="key-value heads over attention heads (1.0 for full attention, 0.25 for grouped-query with 4:1)",
    )
    parser.add_argument("--context", type=int, default=4096, help="tokens per sequence")
    parser.add_argument(
        "--concurrency", type=int, default=1, help="sequences served at once"
    )
    parser.add_argument("--kv-bits", type=float, default=16)
    parser.add_argument("--gpu-gb", type=float, default=0.0)
    parser.add_argument("--ram-gb", type=float, default=0.0)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.15,
        help="working margin as a share of the total",
    )
    args = parser.parse_args(argv)

    weights = args.params_b * 1e9 * args.bits / 8 / 1e9
    kv_per_token = (
        2 * args.layers * args.hidden * args.kv_heads_ratio * args.kv_bits / 8
    )
    kv = kv_per_token * args.context * args.concurrency / 1e9
    total = (weights + kv) * (1 + args.margin)
    print(
        f"{args.params_b:g}B parameters at {args.bits:g} bits: weights {weights:.1f} GB"
    )
    print(
        f"KV cache for {args.concurrency} x {args.context} tokens: {kv:.1f} GB ({kv_per_token / 1e3:.0f} KB per token)"
    )
    print(f"total with {args.margin:.0%} margin: {total:.1f} GB")
    if args.gpu_gb:
        print(
            f"GPU {args.gpu_gb:g} GB: {'fits' if total <= args.gpu_gb else 'does not fit'}"
            + (
                ""
                if total <= args.gpu_gb
                else f"; reduce concurrency to {max(1, int((args.gpu_gb / (1 + args.margin) - weights) / (kv_per_token * args.context / 1e9)))} or quantize further"
            )
        )
    if args.ram_gb:
        print(
            f"system RAM {args.ram_gb:g} GB (CPU serving, slower): {'fits' if total <= args.ram_gb else 'does not fit'}"
        )
    print(
        "\nMeasure throughput and memory on a test run before procurement; the estimate sizes the test."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
