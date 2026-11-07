"""Turn a run folder (answers, judgments, human labels) into agreement/bias/cost tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import CRITERIA
from .config import RunConfig
from .metrics import agreement_row, confusion

KEY = ["item_id", "generator"]


def load_labels(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=KEY + list(CRITERIA))
    df = pd.read_json(p, lines=True, dtype={"item_id": str})
    # last label wins if an item was re-labeled
    return df.sort_values("ts").drop_duplicates(KEY, keep="last")[KEY + list(CRITERIA)]


def _long(df: pd.DataFrame, who: str) -> pd.DataFrame:
    return df.melt(id_vars=KEY, value_vars=list(CRITERIA), var_name="criterion",
                   value_name=who).dropna()


def paired(x: pd.DataFrame, y: pd.DataFrame, nx: str, ny: str, items: pd.DataFrame):
    m = _long(x, nx).merge(_long(y, ny), on=KEY + ["criterion"])
    return m.merge(items[["item_id", "lang"]], on="item_id")


def agreement_tables(m: pd.DataFrame, nx: str, ny: str, cfg: RunConfig) -> pd.DataFrame:
    rows = []
    groups = [("all", "all", m)]
    groups += [(lg, "all", g) for lg, g in m.groupby("lang")]
    groups += [("all", c, g) for c, g in m.groupby("criterion")]
    groups += [(lg, c, g) for (lg, c), g in m.groupby(["lang", "criterion"])]
    for lg, c, g in groups:
        if len(g) < 5:
            continue
        r = agreement_row(g[nx], g[ny], cfg.analysis.bootstrap_iters, cfg.analysis.ci, cfg.seed)
        rows.append({"pair": f"{nx}~{ny}", "lang": lg, "criterion": c, **r})
    return pd.DataFrame(rows)


def confusions(m: pd.DataFrame, nx: str, ny: str) -> dict:
    out = {}
    for (lg, c), g in m.groupby(["lang", "criterion"]):
        out[f"{lg}|{c}"] = confusion(g[nx].astype(int), g[ny].astype(int)).astype(int).tolist()
    return out


def bias_checks(answers, judg, human, items, pair_df) -> dict:
    df = answers.merge(judg, on=KEY).merge(items[["item_id", "lang"]], on="item_id")
    df["ans_len"] = df.answer.str.len()
    df["judge_mean"] = df[list(CRITERIA)].mean(axis=1)
    out = {"length_vs_judge_spearman": float(df.ans_len.corr(df.judge_mean, method="spearman")),
           "position_vs_judge_spearman":
               float(df.judge_position.corr(df.judge_mean, method="spearman"))}
    if len(human):
        h = human.copy()
        h["human_mean"] = h[list(CRITERIA)].mean(axis=1)
        d = df.merge(h[KEY + ["human_mean"]], on=KEY)
        if len(d) >= 5:
            out["length_vs_human_spearman"] = float(d.ans_len.corr(d.human_mean, method="spearman"))
            # does the judge over-score long answers relative to humans?
            out["length_vs_judge_minus_human_spearman"] = float(
                d.ans_len.corr(d.judge_mean - d.human_mean, method="spearman"))
            out["n_length_human"] = len(d)
    if len(pair_df):
        out["pairwise_n"] = len(pair_df)
        out["pairwise_consistency"] = float(pair_df.consistent.mean())
        out["pairwise_first_slot_both"] = float(pair_df.first_slot_both.mean())
        out["pairwise_second_slot_both"] = float(pair_df.second_slot_both.mean())
    return out


def cost_latency(answers, judg, cfg: RunConfig) -> dict:
    p = cfg.pricing_usd_per_mtok.get(cfg.judge.model)
    out = {"judgments": len(judg),
           "judge_parse_fail_rate": float(1 - judg.parse_ok.mean()) if len(judg) else None,
           "judge_latency_s_mean": float(judg.judge_latency_s.mean()),
           "judge_latency_s_p50": float(judg.judge_latency_s.median()),
           "judge_latency_s_p95": float(judg.judge_latency_s.quantile(0.95)),
           "judge_in_tok_mean": float(judg.judge_in_tok.mean()),
           "judge_out_tok_mean": float(judg.judge_out_tok.mean())}
    if p:
        c = (judg.judge_in_tok * p.input + judg.judge_out_tok * p.output) / 1e6
        out["judge_cost_usd_per_judgment"] = float(c.mean())
        out["judge_cost_usd_total"] = float(c.sum())
    return out


def worst_disagreements(answers, judg, human, items, k=10) -> pd.DataFrame:
    if not len(human):
        return pd.DataFrame()
    d = judg.merge(human, on=KEY, suffixes=("_judge", "_human"))
    d["abs_diff"] = sum((d[f"{c}_judge"] - d[f"{c}_human"]).abs() for c in CRITERIA)
    d = d.merge(answers[KEY + ["answer"]], on=KEY).merge(
        items[["item_id", "lang", "question", "reference"]], on="item_id")
    cols = ["item_id", "generator", "lang", "abs_diff", "question", "reference", "answer",
            "rationale"] + [f"{c}_{w}" for c in CRITERIA for w in ("judge", "human")]
    return d.sort_values(["abs_diff", "item_id"], ascending=[False, True]).head(k)[cols]


def analyze(run_dir: Path, cfg: RunConfig, items: pd.DataFrame) -> dict:
    answers = pd.read_csv(run_dir / "answers.csv", dtype={"item_id": str})
    judg = pd.read_csv(run_dir / "judgments.csv", dtype={"item_id": str})
    pair_p = run_dir / "pairwise.csv"
    try:
        pair_df = pd.read_csv(pair_p)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pair_df = pd.DataFrame()
    ha, hb = load_labels(cfg.labels.human_a), load_labels(cfg.labels.human_b)
    jok = judg[judg.parse_ok][KEY + list(CRITERIA)]

    tables, conf = [], {}
    if len(ha):
        m = paired(jok, ha, "judge", "human_a", items)
        tables.append(agreement_tables(m, "judge", "human_a", cfg))
        conf["judge~human_a"] = confusions(m, "judge", "human_a")
    if len(ha) and len(hb):
        m = paired(ha, hb, "human_a", "human_b", items)
        tables.append(agreement_tables(m, "human_a", "human_b", cfg))
        conf["human_a~human_b"] = confusions(m, "human_a", "human_b")
        mj = paired(jok, hb, "judge", "human_b", items)
        tables.append(agreement_tables(mj, "judge", "human_b", cfg))
    agree = pd.concat(tables) if tables else pd.DataFrame()
    agree.to_csv(run_dir / "agreement.csv", index=False)

    summary = {
        "n_items": int(items.item_id.nunique()), "n_answers": len(answers),
        "n_human_a": len(ha), "n_human_b": len(hb),
        "bias": bias_checks(answers, judg, ha, items, pair_df),
        "cost_latency": cost_latency(answers, judg, cfg),
        "confusion": conf,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_np), "utf-8")
    worst_disagreements(answers, jok.merge(judg[KEY + ["rationale"]], on=KEY), ha, items) \
        .to_csv(run_dir / "worst_disagreements.csv", index=False)
    return summary


def _np(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    raise TypeError(type(o))
