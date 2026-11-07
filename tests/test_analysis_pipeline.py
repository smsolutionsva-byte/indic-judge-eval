"""Smoke test of analyze() on SYNTHETIC test fixtures written to a temp dir.
These are not human labels and are never written to labels/ or reported anywhere."""
import json

import numpy as np
import pandas as pd

from indic_judge import CRITERIA
from indic_judge.analysis import analyze
from indic_judge.config import load_config


def test_analyze_runs_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n = 30
    items = pd.DataFrame({"item_id": [f"x-{i:03d}" for i in range(n)],
                          "lang": ["hi", "kn", "en"] * (n // 3), "question": "q",
                          "reference": "r", "context": "c"})
    ans = pd.DataFrame({"item_id": items.item_id, "generator": "g",
                        "answer": ["a" * int(k) for k in rng.integers(5, 50, n)]})
    j = ans[["item_id", "generator"]].copy()
    for c in CRITERIA:
        j[c] = rng.integers(1, 5, n)
    j["rationale"], j["parse_ok"], j["judge_position"] = "r", True, range(n)
    j["judge_latency_s"], j["judge_in_tok"], j["judge_out_tok"] = 1.0, 1000, 50
    run = tmp_path / "run"
    run.mkdir()
    ans.to_csv(run / "answers.csv", index=False)
    j.to_csv(run / "judgments.csv", index=False)
    fake_a, fake_b = tmp_path / "synthetic_a.jsonl", tmp_path / "synthetic_b.jsonl"
    with open(fake_a, "w") as fa, open(fake_b, "w") as fb:
        for i, r in j.iterrows():
            rec = {"item_id": r.item_id, "generator": "g", "ts": i,
                   **{c: int(np.clip(r[c] + rng.integers(-1, 2), 1, 4)) for c in CRITERIA}}
            fa.write(json.dumps(rec) + "\n")
            if i < 20:
                fb.write(json.dumps(rec) + "\n")
    cfg = load_config("configs/sample20.yaml")
    cfg.labels.human_a, cfg.labels.human_b = str(fake_a), str(fake_b)
    cfg.analysis.bootstrap_iters = 20
    s = analyze(run, cfg, items)
    agree = pd.read_csv(run / "agreement.csv")
    assert set(agree.pair) == {"judge~human_a", "human_a~human_b", "judge~human_b"}
    hh = agree[(agree.pair == "human_a~human_b") & (agree.lang == "all") &
               (agree.criterion == "all")]
    assert hh.kappa.iloc[0] == 1.0  # identical files -> perfect agreement
    assert s["cost_latency"]["judge_in_tok_mean"] == 1000
    assert len(pd.read_csv(run / "worst_disagreements.csv")) == 10
