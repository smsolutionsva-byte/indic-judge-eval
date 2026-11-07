"""Build data/items.jsonl from IndicQA (hi, kn; CC-BY-4.0) and SQuAD v1.1 dev (en; CC-BY-SA-4.0)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

MAX_CONTEXT_CHARS = 2000
SOURCES = {
    "indicqa": {"repo": "ai4bharat/IndicQA", "license": "CC-BY-4.0"},
    "squad": {"repo": "rajpurkar/squad", "license": "CC-BY-SA-4.0"},
}


def _download(raw_dir: Path):
    from huggingface_hub import hf_hub_download

    for lang in ("hi", "kn"):
        hf_hub_download("ai4bharat/IndicQA", f"data/indicqa.{lang}.json", repo_type="dataset",
                        local_dir=raw_dir / "indicqa")
    hf_hub_download("rajpurkar/squad", "plain_text/validation-00000-of-00001.parquet",
                    repo_type="dataset", local_dir=raw_dir / "squad")


def _indicqa(raw_dir: Path, lang: str) -> list[dict]:
    d = json.loads((raw_dir / "indicqa" / "data" / f"indicqa.{lang}.json").read_text("utf-8"))
    out = []
    for ai, art in enumerate(d["data"]):
        for pi, p in enumerate(art["paragraphs"]):
            ctx = p["context"].strip()
            if len(ctx) > MAX_CONTEXT_CHARS:
                continue
            for q in p["qas"]:
                ans = q["answers"][0]["text"].strip()
                if not ans:  # drop unanswerable questions
                    continue
                out.append({"src_id": f"{lang}:{ai}:{pi}:{q['id']}", "lang": lang,
                            "context": ctx, "question": q["question"].strip(),
                            "reference": ans, "source": "indicqa"})
    return out


def _squad(raw_dir: Path) -> list[dict]:
    df = pd.read_parquet(raw_dir / "squad" / "plain_text" / "validation-00000-of-00001.parquet")
    df = df[df.context.str.len() <= MAX_CONTEXT_CHARS]
    return [{"src_id": f"squad:{r.id}", "lang": "en", "context": r.context,
             "question": r.question, "reference": r.answers["text"][0], "source": "squad"}
            for r in df.itertuples()]


def build_items(out_path="data/items.jsonl", raw_dir="data/raw", n_per_lang=None, seed=13):
    n_per_lang = n_per_lang or {"hi": 80, "kn": 80, "en": 30}
    raw_dir = Path(raw_dir)
    if not (raw_dir / "indicqa").exists() or not (raw_dir / "squad").exists():
        _download(raw_dir)
    rng = random.Random(seed)
    pools = {"hi": _indicqa(raw_dir, "hi"), "kn": _indicqa(raw_dir, "kn"), "en": _squad(raw_dir)}
    items = []
    for lang, n in n_per_lang.items():
        pool = sorted(pools[lang], key=lambda r: r["src_id"])
        items += rng.sample(pool, n)
    for i, it in enumerate(items):
        it["item_id"] = f"{it['lang']}-{i:03d}"
        it["license"] = SOURCES[it["source"]]["license"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return items


def load_items(path="data/items.jsonl", sample_n=None, seed=13) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, dtype={"item_id": str})
    if sample_n:
        # stratified across languages, deterministic for a given seed
        langs = sorted(df.lang.unique())
        per = [sample_n // len(langs) + (1 if i < sample_n % len(langs) else 0)
               for i in range(len(langs))]
        df = pd.concat([df[df.lang == lg].sample(n=min(k, (df.lang == lg).sum()), random_state=seed)
                        for lg, k in zip(langs, per)])
    return df.sort_values("item_id").reset_index(drop=True)
