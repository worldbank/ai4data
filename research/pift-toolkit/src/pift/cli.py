"""
Command-line interface tying the stages together.

    pift preview   --config CONFIG [--n 3]         # show serialized records (+ a permuted one)
    pift generate  --config CONFIG --split train|eval [--limit N]
    pift mine      --config CONFIG --queries Q.jsonl --out T.jsonl [--miner MODEL]
    pift finetune  --config CONFIG --triplets T.jsonl --out DIR [--base MODEL]
    pift evaluate  --config CONFIG --queries EVAL.jsonl [--model DIR]
    pift search    --config CONFIG [--model DIR]    # interactive
    pift demo      --config CONFIG --out DIR        # offline end-to-end smoke run

All stages read the catalogue and schema from CONFIG (a YAML file). See
``configs/example.yaml`` and the docs/ directory.
"""

from __future__ import annotations

import argparse
import random
import sys

from .config import load_config


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def cmd_preview(args):
    from .records import load_records
    from .serialize import serialize

    cfg = load_config(args.config)
    records = load_records(cfg)
    for rid in list(records)[: args.n]:
        print(f"\n=== {rid} ===")
        print("canonical:", serialize(records[rid], cfg))
        rng = random.Random(7)
        print("augmented:", serialize(records[rid], cfg, rng=rng, permute=True,
                                      field_dropout=cfg.serialization["field_dropout"]))


def cmd_generate(args):
    _load_env()
    from .generate import generate_split

    cfg = load_config(args.config)
    generate_split(cfg, args.split, limit=args.limit, out_path=args.out)


def cmd_mine(args):
    cfg = load_config(args.config)
    from .mine import mine

    mine(cfg, args.queries, args.out, miner_model=args.miner,
         n_negatives=args.n_negatives, device=args.device)


def cmd_finetune(args):
    cfg = load_config(args.config)
    from .finetune import finetune

    finetune(cfg, args.triplets, args.out, base_model=args.base,
             loss=args.loss, epochs=args.epochs, batch_size=args.batch_size,
             device=args.device)


def _make_judge(cfg, args):
    if not getattr(args, "judge", False):
        return None
    _load_env()
    from .judge import Judge
    return Judge(cfg, provider=args.judge_provider, model=args.judge_model,
                 cache_path=args.judge_cache)


def cmd_evaluate(args):
    cfg = load_config(args.config)
    from .evaluate import evaluate

    evaluate(cfg, args.queries, model=args.model,
             robustness=not args.no_robustness, device=args.device,
             judge=_make_judge(cfg, args), judge_top_k=args.judge_top_k)


def cmd_benchmark(args):
    cfg = load_config(args.config)
    from .benchmark import benchmark

    # `base` is sugar for "the config's base model" (None resolves to it).
    models = [None if m in ("base", "BASE") else m for m in args.models]
    benchmark(cfg, args.queries, models,
              robustness=not args.no_robustness, device=args.device,
              judge=_make_judge(cfg, args), judge_top_k=args.judge_top_k,
              out_dir=args.out)


def cmd_search(args):
    cfg = load_config(args.config)
    from .search import interactive

    interactive(cfg, model=args.model, top_k=args.top_k, device=args.device)


def cmd_demo(args):
    """Offline end-to-end run on the sample catalogue (no API, no GPU needed for
    generation/metrics; fine-tuning still needs torch). Validates the wiring."""
    _load_env()
    from pathlib import Path

    from .generate import generate_split
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # force the offline heuristic generator
    cfg.raw.setdefault("generation", {})
    cfg.raw["generation"]["provider"] = "heuristic"
    cfg.raw["generation"]["eval_provider"] = "heuristic"
    tq = generate_split(cfg, "train", out_path=str(out / "train_queries.jsonl"))
    eq = generate_split(cfg, "eval", out_path=str(out / "eval_queries.jsonl"))
    print(f"\nDemo complete. Generated queries in {out}/.")
    print("Next (needs torch + sentence-transformers installed):")
    print(f"  pift mine     --config {args.config} --queries {tq} --out {out}/triplets.jsonl")
    print(f"  pift finetune --config {args.config} --triplets {out}/triplets.jsonl --out {out}/model")
    print(f"  pift evaluate --config {args.config} --queries {eq} --model {out}/model")


def build_parser():
    p = argparse.ArgumentParser(prog="pift", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_config(sp):
        sp.add_argument("--config", required=True, help="path to the catalogue YAML config")

    sp = sub.add_parser("preview", help="show serialized records")
    add_config(sp); sp.add_argument("--n", type=int, default=3); sp.set_defaults(func=cmd_preview)

    sp = sub.add_parser("generate", help="generate synthetic queries")
    add_config(sp)
    sp.add_argument("--split", choices=["train", "eval"], required=True)
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("mine", help="mine hard negatives")
    add_config(sp)
    sp.add_argument("--queries", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--miner", default=None)
    sp.add_argument("--n-negatives", type=int, default=None)
    sp.add_argument("--device", default=None)
    sp.set_defaults(func=cmd_mine)

    sp = sub.add_parser("finetune", help="permutation-invariant fine-tuning")
    add_config(sp)
    sp.add_argument("--triplets", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--base", default=None)
    sp.add_argument("--loss", choices=["cmnrl", "cgist"], default=None)
    sp.add_argument("--epochs", type=float, default=None)
    sp.add_argument("--batch-size", type=int, default=None)
    sp.add_argument("--device", default=None)
    sp.set_defaults(func=cmd_finetune)

    def add_judge(sp):
        sp.add_argument("--judge", action="store_true",
                        help="also score top-k results with an LLM judge (0-3)")
        sp.add_argument("--judge-provider", default="anthropic",
                        help="anthropic | openai | heuristic (offline)")
        sp.add_argument("--judge-model", default="claude-haiku-4-5")
        sp.add_argument("--judge-top-k", type=int, default=10)
        sp.add_argument("--judge-cache", default="data/judge_cache.json",
                        help="shared judgement cache (relevance is per pair, not per model)")

    sp = sub.add_parser("evaluate", help="held-out metrics + order-robustness + optional judge")
    add_config(sp)
    sp.add_argument("--queries", required=True)
    sp.add_argument("--model", default=None, help="fine-tuned dir or HF id (default: config base)")
    sp.add_argument("--no-robustness", action="store_true")
    sp.add_argument("--device", default=None)
    add_judge(sp)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("benchmark", help="compare several models on the same held-out set")
    add_config(sp)
    sp.add_argument("--queries", required=True)
    sp.add_argument("--models", nargs="+", required=True,
                    help="model dirs / HF ids to compare; use 'base' for the config base model")
    sp.add_argument("--no-robustness", action="store_true")
    sp.add_argument("--device", default=None)
    sp.add_argument("--out", default="data/benchmark")
    add_judge(sp)
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("search", help="interactive semantic search")
    add_config(sp)
    sp.add_argument("--model", default=None)
    sp.add_argument("--top-k", type=int, default=10)
    sp.add_argument("--device", default=None)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("demo", help="offline end-to-end smoke run on the sample catalogue")
    add_config(sp)
    sp.add_argument("--out", default="data/demo")
    sp.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
