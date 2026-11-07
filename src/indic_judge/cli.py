"""indic-judge CLI. Run from the repo root.

  python -m indic_judge.cli build-data
  python -m indic_judge.cli run configs/sample20.yaml [--offline] [--yes-spend]
  python -m indic_judge.cli analyze runs/<run_id>
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from .analysis import analyze
from .cache import JsonlCache
from .config import load_config
from .data import build_items, load_items
from .generate import GEN_PROMPT, gen_prompt, generate
from .judge import RUBRIC, build_prompt, judge, pairwise_position_check
from .llm import OfflineError, is_cached

CHARS_PER_TOKEN = 2.0  # conservative for Devanagari/Kannada; only used for the budget gate


def estimate_worst_case_usd(cfg, items, cache) -> float:
    """Upper-bound cost of the uncached Anthropic calls this run would make (output = max_tokens)."""
    def price(model, tin, tout):
        p = cfg.pricing_usd_per_mtok.get(model)
        return 0.0 if p is None else (tin * p.input + tout * p.output) / 1e6

    gtmpl = GEN_PROMPT.read_text("utf-8")
    jtmpl, rubric = Path(cfg.judge.prompt_file).read_text("utf-8"), RUBRIC.read_text("utf-8")
    total, n_gen = 0.0, 0
    for g in cfg.generators:
        if g.provider != "anthropic":
            n_gen += 1
            continue
        n_gen += 1
        for it in items.itertuples():
            pr = gen_prompt(gtmpl, it)
            if not is_cached(cache, g.provider, g.model, pr, g.max_tokens, g.temperature):
                total += price(g.model, len(pr) / CHARS_PER_TOKEN, g.max_tokens)
    if cfg.judge.provider == "anthropic":
        for it in items.itertuples():
            pr = build_prompt(jtmpl, rubric, it, "x" * 600)
            per = price(cfg.judge.model, len(pr) / CHARS_PER_TOKEN, cfg.judge.max_tokens)
            total += per * n_gen + 2 * price(cfg.judge.model, len(pr) / CHARS_PER_TOKEN, cfg.judge.max_tokens)
    return total


def cmd_run(args):
    cfg = load_config(args.config)
    items = load_items(cfg.dataset.path, cfg.dataset.sample_n, cfg.seed)
    cache = JsonlCache(cfg.cache_path)
    if not args.offline:
        est = estimate_worst_case_usd(cfg, items, cache)
        print(f"[run] worst-case estimate for uncached API calls: ${est:.2f}")
        if est > cfg.max_budget_usd and not args.yes_spend:
            sys.exit(f"[run] estimate exceeds max_budget_usd={cfg.max_budget_usd}. "
                     "Re-run with --yes-spend after confirming.")
    run_id = f"{cfg.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    shutil.copy(cfg.judge.prompt_file, run_dir / Path(cfg.judge.prompt_file).name)
    t0 = time.time()
    try:
        _execute(cfg, items, cache, run_dir, args.offline)
    except OfflineError as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        sys.exit(f"[run] {e}. Set ANTHROPIC_API_KEY (see .env.example) or drop --offline.")
    (Path("runs") / f"LATEST_{cfg.run_name}").write_text(run_id, "utf-8")
    print(f"[run] done in {time.time() - t0:.1f}s -> {run_dir}")


def _execute(cfg, items, cache, run_dir, offline):
    answers = generate(cfg, items, cache, offline=offline)
    answers.to_csv(run_dir / "answers.csv", index=False)
    print(f"[run] {len(answers)} answers")
    judg = judge(cfg, items, answers, cache, offline=offline)
    judg.to_csv(run_dir / "judgments.csv", index=False)
    print(f"[run] {len(judg)} judgments, parse failures: {(~judg.parse_ok).sum()}")
    pairwise_position_check(cfg, items, answers, cache, offline=offline) \
        .to_csv(run_dir / "pairwise.csv", index=False)
    items.to_json(run_dir / "items.jsonl", orient="records", lines=True, force_ascii=False)
    s = analyze(run_dir, cfg, items)
    print(f"[run] human labels: A={s['n_human_a']} B={s['n_human_b']}")


def cmd_analyze(args):
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    items = pd.read_json(run_dir / "items.jsonl", lines=True, dtype={"item_id": str})
    analyze(run_dir, cfg, items)
    print((run_dir / "agreement.csv").read_text("utf-8")[:3000])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="indic-judge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-data")
    b.add_argument("--seed", type=int, default=13)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--offline", action="store_true", help="cache only, never call an API")
    r.add_argument("--yes-spend", action="store_true", help="allow runs above max_budget_usd")
    a = sub.add_parser("analyze")
    a.add_argument("run_dir")
    args = ap.parse_args(argv)
    if args.cmd == "build-data":
        items = build_items(seed=args.seed)
        print(f"wrote {len(items)} items to data/items.jsonl")
    elif args.cmd == "run":
        cmd_run(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
