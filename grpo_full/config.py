from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    num_steps: int = 100
    batch_size: int = 2
    num_generations: int = 4
    max_prompt_length: int = 256
    max_new_tokens: int = 96
    temperature: float = 0.9
    top_p: float = 0.95
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    clip_range: float = 0.2
    kl_beta: float = 0.04
    advantage_eps: float = 1e-6
    log_every: int = 1
    save_every: int = 50


@dataclass
class WandbConfig:
    project: str = "grpo-full"
    run_name: str = "debug"
    mode: str = "online"


@dataclass
class DatasetConfig:
    name: str = "openai/gsm8k"
    config_name: str = "main"
    split: str = "train"
    max_examples: int | None = None


@dataclass
class AppConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "outputs/grpo"
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _merge_dataclass(cls: type, values: dict[str, Any]):
    known = {field.name for field in cls.__dataclass_fields__.values()}
    kwargs = {key: value for key, value in values.items() if key in known}
    return cls(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    train = _merge_dataclass(TrainConfig, raw.get("train", {}))
    wandb = _merge_dataclass(WandbConfig, raw.get("wandb", {}))
    dataset = _merge_dataclass(DatasetConfig, raw.get("dataset", {}))
    return AppConfig(
        model_name=raw.get("model_name", AppConfig.model_name),
        output_dir=raw.get("output_dir", AppConfig.output_dir),
        seed=raw.get("seed", AppConfig.seed),
        dataset=dataset,
        train=train,
        wandb=wandb,
    )
