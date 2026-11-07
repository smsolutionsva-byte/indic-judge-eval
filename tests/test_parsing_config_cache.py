import pytest
from pydantic import ValidationError

from indic_judge.cache import JsonlCache, cache_key
from indic_judge.config import load_config
from indic_judge.judge import JudgeParseError, parse_judgment
from indic_judge.label import queue, save_label
from indic_judge.llm import OfflineError, complete

GOOD = '{"correctness": 4, "faithfulness": 3, "fluency": 2, "completeness": 1, "rationale": "ok"}'


@pytest.mark.parametrize("text", [GOOD, f"```json\n{GOOD}\n```", f"Here you go:\n{GOOD}\nThanks"])
def test_parse_judgment_variants(text):
    j = parse_judgment(text)
    assert (j.correctness, j.faithfulness, j.fluency, j.completeness) == (4, 3, 2, 1)


@pytest.mark.parametrize("text", [
    "", "no json here", '{"correctness": 5, "faithfulness": 3, "fluency": 2, "completeness": 1}',
    '{"correctness": 4}', '{"correctness": 4, broken',
])
def test_parse_judgment_rejects_bad(text):
    with pytest.raises(JudgeParseError):
        parse_judgment(text)


def test_configs_load():
    for p in ["configs/sample20.yaml", "configs/full.yaml"]:
        cfg = load_config(p)
        assert cfg.judge.temperature == 0.0
        assert cfg.generators
    assert load_config("configs/sample20.yaml").dataset.sample_n == 20


def test_config_rejects_unknown_provider(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run_name: x\ngenerators: [{name: a, provider: openai, model: m}]\n"
                 "judge: {model: m, prompt_file: p}\n")
    with pytest.raises(ValidationError):
        load_config(p)


def test_cache_roundtrip_and_reload(tmp_path):
    path = tmp_path / "c.jsonl"
    c = JsonlCache(path)
    k = cache_key(a=1, text="नमस्ते")
    c.put(k, {"text": "ಹಲೋ", "input_tokens": 1, "output_tokens": 2, "latency_s": 0.1})
    path.open("a", encoding="utf-8").write('{"key": "trunc')  # interrupted write
    c2 = JsonlCache(path)
    assert k in c2 and c2.get(k)["text"] == "ಹಲೋ" and len(c2) == 1


def test_cache_key_sensitive_to_every_part():
    base = {'provider': "anthropic", 'model': "m", 'prompt': "p", 'max_tokens': 1, 'temperature': 0.0}
    keys = {cache_key(**base)}
    for f, v in [("model", "m2"), ("prompt", "p2"), ("max_tokens", 2), ("temperature", 0.5)]:
        keys.add(cache_key(**{**base, f: v}))
    assert len(keys) == 5


def test_complete_uses_cache_and_never_calls_backend(tmp_path, monkeypatch):
    c = JsonlCache(tmp_path / "c.jsonl")
    k = cache_key(provider="anthropic", model="m", prompt="p", max_tokens=5, temperature=0.0)
    c.put(k, {"text": "hi", "input_tokens": 3, "output_tokens": 1, "latency_s": 0.5})

    def boom(*a, **kw):
        raise AssertionError("backend must not be called on a cache hit")

    monkeypatch.setitem(__import__("indic_judge.llm").llm.BACKENDS, "anthropic", boom)
    r = complete(c, "anthropic", "m", "p", 5)
    assert r.cached and r.text == "hi"
    with pytest.raises(OfflineError):
        complete(c, "anthropic", "m", "other", 5, offline=True)


def test_label_queue_and_overlap(tmp_path):
    import pandas as pd
    ans = pd.DataFrame({"item_id": ["a", "b", "c"], "generator": ["g"] * 3, "answer": ["x"] * 3})
    fa, fb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first = queue(ans, fa).iloc[0]
    save_label(fa, "A", first.item_id, "g", {'correctness': 4, 'faithfulness': 4, 'fluency': 4,
                                                  'completeness': 4})
    assert len(queue(ans, fa)) == 2
    assert list(queue(ans, fb, restrict_to=fa).item_id) == [first.item_id]
    with pytest.raises(ValueError):
        save_label(fa, "A", "b", "g", {'correctness': 5, 'faithfulness': 4, 'fluency': 4,
                                            'completeness': 4})
