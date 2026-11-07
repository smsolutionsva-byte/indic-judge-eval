from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cache import JsonlCache
from .config import RunConfig
from .llm import OfflineError, complete, pmap

GEN_PROMPT = Path("prompts/generate_v1.md")


def gen_prompt(template: str, item) -> str:
    return template.format(context=item.context, question=item.question)


def generate(cfg: RunConfig, items: pd.DataFrame, cache: JsonlCache, offline=False) -> pd.DataFrame:
    tmpl = GEN_PROMPT.read_text("utf-8")
    rows = []
    for g in cfg.generators:
        g_rows = []
        try:
            its = list(items.itertuples())
            res = pmap(lambda it, g=g: complete(cache, g.provider, g.model, gen_prompt(tmpl, it),
                                                g.max_tokens, g.temperature, offline=offline), its)
            for it, r in zip(its, res):
                g_rows.append({"item_id": it.item_id, "generator": g.name, "gen_model": g.model,
                               "answer": r.text.strip(), "gen_in_tok": r.input_tokens,
                               "gen_out_tok": r.output_tokens, "gen_latency_s": r.latency_s})
        except OfflineError as e:
            if g.optional:
                print(f"[generate] skipping optional generator {g.name}: {e}")
                continue
            raise
        rows += g_rows
    return pd.DataFrame(rows)
