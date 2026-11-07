from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from . import CRITERIA
from .cache import JsonlCache
from .config import RunConfig
from .llm import complete, pmap

RUBRIC = Path("prompts/rubric.md")


class Judgment(BaseModel):
    correctness: int = Field(ge=1, le=4)
    faithfulness: int = Field(ge=1, le=4)
    fluency: int = Field(ge=1, le=4)
    completeness: int = Field(ge=1, le=4)
    rationale: str = ""


class JudgeParseError(ValueError):
    pass


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judgment(text: str) -> Judgment:
    """Accept bare JSON, JSON in code fences, or JSON surrounded by stray prose."""
    m = _JSON_RE.search(text or "")
    if not m:
        raise JudgeParseError(f"no JSON object in judge output: {str(text)[:120]!r}")
    try:
        return Judgment.model_validate(json.loads(m.group(0)))
    except (json.JSONDecodeError, ValidationError) as e:
        raise JudgeParseError(str(e)) from e


def build_prompt(template: str, rubric: str, item, answer: str) -> str:
    return template.format(rubric=rubric, context=item.context, question=item.question,
                           reference=item.reference, answer=answer)


def judge(cfg: RunConfig, items: pd.DataFrame, answers: pd.DataFrame, cache: JsonlCache,
          offline=False) -> pd.DataFrame:
    tmpl = Path(cfg.judge.prompt_file).read_text("utf-8")
    rubric = RUBRIC.read_text("utf-8")
    by_id = items.set_index("item_id")
    # Judging order is shuffled with a fixed seed; judge_position lets us test for drift.
    order = list(answers.index)
    random.Random(cfg.judge.shuffle_seed).shuffle(order)
    def call(idx):
        a = answers.loc[idx]
        prompt = build_prompt(tmpl, rubric, by_id.loc[a.item_id], a.answer)
        return complete(cache, cfg.judge.provider, cfg.judge.model, prompt, cfg.judge.max_tokens,
                        cfg.judge.temperature, offline=offline)

    results = pmap(call, order)
    rows = []
    for pos, (idx, r) in enumerate(zip(order, results)):
        a = answers.loc[idx]
        row = {"item_id": a.item_id, "generator": a.generator, "judge_position": pos,
               "judge_in_tok": r.input_tokens, "judge_out_tok": r.output_tokens,
               "judge_latency_s": r.latency_s, "judge_cached": r.cached, "parse_ok": True}
        try:
            j = parse_judgment(r.text)
            row.update({c: getattr(j, c) for c in CRITERIA}, rationale=j.rationale)
        except JudgeParseError as e:
            row.update({c: None for c in CRITERIA}, rationale=f"PARSE_ERROR: {e}", parse_ok=False)
        rows.append(row)
    return pd.DataFrame(rows)


PAIR_PROMPT = """Two answers to the same question are shown. Which one is better overall \
(correct, faithful to the passage, fluent in the question's language, complete)?

Passage:
{context}

Question: {question}

Reference answer: {reference}

Answer A: {a}

Answer B: {b}

Reply with ONLY one letter: A or B."""


def pairwise_position_check(cfg: RunConfig, items, answers, cache, offline=False) -> pd.DataFrame:
    """Ask for a preference in both orders. A position-robust judge picks the same answer."""
    gens = sorted(answers.generator.unique())
    if len(gens) < 2:
        return pd.DataFrame()
    g1, g2 = gens[:2]
    wide = answers.pivot(index="item_id", columns="generator", values="answer").dropna()
    by_id = items.set_index("item_id")
    rows = []
    for item_id, r in wide.iterrows():
        it = by_id.loc[item_id]
        picks = []
        for a, b in ((r[g1], r[g2]), (r[g2], r[g1])):
            p = PAIR_PROMPT.format(context=it.context, question=it.question,
                                   reference=it.reference, a=a, b=b)
            out = complete(cache, cfg.judge.provider, cfg.judge.model, p, cfg.judge.max_tokens,
                           0.0, offline=offline)
            picks.append(out.text.strip()[:1].upper())
        w1 = g1 if picks[0] == "A" else g2
        w2 = g2 if picks[1] == "A" else g1
        rows.append({"item_id": item_id, "pick_order1": picks[0], "pick_order2": picks[1],
                     "consistent": w1 == w2, "first_slot_both": picks == ["A", "A"],
                     "second_slot_both": picks == ["B", "B"]})
    return pd.DataFrame(rows)
