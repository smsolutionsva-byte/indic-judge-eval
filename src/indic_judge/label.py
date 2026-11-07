"""Labeling queue logic shared by the Streamlit labeling app. Labels are only ever written
by a human clicking Save; nothing in this repo generates them."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pandas as pd

from . import CRITERIA

KEY = ["item_id", "generator"]


def labeled_keys(path: str | Path) -> set[tuple[str, str]]:
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["item_id"], r["generator"]))
    return out


def queue(answers: pd.DataFrame, out_path: str | Path, seed: int = 13,
          restrict_to: str | Path | None = None, limit: int | None = None) -> pd.DataFrame:
    """Unlabeled answers in a fixed shuffled order. restrict_to: another labeler's file,
    so a second annotator labels the same overlap set as the first."""
    df = answers.sort_values(KEY).reset_index(drop=True)
    idx = list(df.index)
    random.Random(seed).shuffle(idx)
    df = df.loc[idx]
    if restrict_to is not None:
        other = labeled_keys(restrict_to)
        df = df[[(a, b) in other for a, b in zip(df.item_id, df.generator)]]
    if limit:
        df = df.head(limit)
    done = labeled_keys(out_path)
    return df[[(a, b) not in done for a, b in zip(df.item_id, df.generator)]]


def save_label(out_path: str | Path, labeler: str, item_id: str, generator: str,
               scores: dict[str, int], note: str = "") -> dict:
    for c in CRITERIA:
        if scores.get(c) not in (1, 2, 3, 4):
            raise ValueError(f"{c} must be 1-4, got {scores.get(c)!r}")
    rec = {"item_id": item_id, "generator": generator, "labeler": labeler,
           **{c: int(scores[c]) for c in CRITERIA}, "note": note, "ts": time.time()}
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
