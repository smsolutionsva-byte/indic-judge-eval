from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DatasetCfg(BaseModel):
    path: str = "data/items.jsonl"
    sample_n: int | None = None


class ModelCfg(BaseModel):
    name: str
    provider: Literal["anthropic", "ollama", "bharatcode"]
    model: str
    max_tokens: int = 300
    temperature: float = 0.0
    optional: bool = False  # skip silently if the backend is unreachable (e.g. no Ollama)


class JudgeCfg(BaseModel):
    provider: Literal["anthropic", "ollama", "bharatcode"] = "anthropic"
    model: str
    prompt_file: str
    max_tokens: int = 400
    temperature: float = 0.0
    shuffle_seed: int = 13


class LabelsCfg(BaseModel):
    human_a: str = "labels/human_a.jsonl"
    human_b: str = "labels/human_b.jsonl"


class AnalysisCfg(BaseModel):
    bootstrap_iters: int = Field(1000, ge=10)
    ci: float = Field(0.95, gt=0, lt=1)


class Price(BaseModel):
    input: float
    output: float


class RunConfig(BaseModel):
    run_name: str
    seed: int = 13
    dataset: DatasetCfg = DatasetCfg()
    generators: list[ModelCfg]
    judge: JudgeCfg
    labels: LabelsCfg = LabelsCfg()
    analysis: AnalysisCfg = AnalysisCfg()
    pricing_usd_per_mtok: dict[str, Price] = {}
    max_budget_usd: float = 3.0
    cache_path: str = "data/cache/llm_cache.jsonl"


def load_config(path: str | Path) -> RunConfig:
    with open(path, encoding="utf-8") as f:
        return RunConfig.model_validate(yaml.safe_load(f))
