from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from grpo_full.data import format_chat_prompt, format_gsm8k_sft_answer
from grpo_full.grpo import get_device, select_dtype


@dataclass
class SFTDatasetConfig:
    name: str = "openai/gsm8k"
    config_name: str = "main"
    split: str = "train"
    max_examples: int | None = 2000


@dataclass
class SFTTrainConfig:
    num_steps: int = 300
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 768
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    log_every: int = 5
    save_every: int = 100


@dataclass
class SFTWandbConfig:
    project: str = "grpo-full"
    run_name: str = "qwen0.5b-gsm8k-sft"
    mode: str = "online"


@dataclass
class SFTLoraConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "outputs/sft-qwen-0.5b"
    seed: int = 42
    dataset: SFTDatasetConfig = field(default_factory=SFTDatasetConfig)
    train: SFTTrainConfig = field(default_factory=SFTTrainConfig)
    wandb: SFTWandbConfig = field(default_factory=SFTWandbConfig)
    lora: SFTLoraConfig = field(default_factory=SFTLoraConfig)


def merge_dataclass(cls: type, values: dict[str, Any]):
    known = {field.name for field in cls.__dataclass_fields__.values()}
    return cls(**{key: value for key, value in values.items() if key in known})


def load_sft_config(path: str | Path) -> SFTConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return SFTConfig(
        model_name=raw.get("model_name", SFTConfig.model_name),
        output_dir=raw.get("output_dir", SFTConfig.output_dir),
        seed=raw.get("seed", SFTConfig.seed),
        dataset=merge_dataclass(SFTDatasetConfig, raw.get("dataset", {})),
        train=merge_dataclass(SFTTrainConfig, raw.get("train", {})),
        wandb=merge_dataclass(SFTWandbConfig, raw.get("wandb", {})),
        lora=merge_dataclass(SFTLoraConfig, raw.get("lora", {})),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gsm8k_rows(cfg: SFTDatasetConfig) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "SFT requires the `datasets` package. Run `pip install -r requirements.txt` first."
        ) from exc

    dataset = load_dataset(cfg.name, cfg.config_name, split=cfg.split)
    if cfg.max_examples is not None:
        dataset = dataset.select(range(min(cfg.max_examples, len(dataset))))
    return [{"question": str(row["question"]), "answer": str(row["answer"])} for row in dataset]


class SFTBatcher:
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int, seed: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = random.Random(seed)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        examples = [self.encode(self.rng.choice(self.rows)) for _ in range(batch_size)]
        input_ids = pad_sequence(
            [example["input_ids"] for example in examples],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = pad_sequence(
            [example["labels"] for example in examples],
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def encode(self, row: dict[str, str]) -> dict[str, torch.Tensor]:
        prompt = format_chat_prompt(row["question"])
        target = format_gsm8k_sft_answer(row["answer"])
        full_text = f"{prompt}{target}{self.tokenizer.eos_token}"

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SFTTrainer:
    def __init__(self, cfg: SFTConfig):
        self.cfg = cfg
        self.train_cfg = cfg.train
        set_seed(cfg.seed)

        self.device = get_device()
        self.dtype = select_dtype(self.device)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.config.use_cache = False
        if cfg.lora.enabled:
            self.model = self.apply_lora(self.model)

        self.optimizer = AdamW(
            (param for param in self.model.parameters() if param.requires_grad),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        rows = load_gsm8k_rows(cfg.dataset)
        self.batcher = SFTBatcher(rows, self.tokenizer, self.train_cfg.max_length, cfg.seed)
        self.run = self.init_wandb()

    def apply_lora(self, model):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "LoRA SFT requires the `peft` package. Run `pip install -r requirements.txt` first."
            ) from exc

        lora_cfg = self.cfg.lora
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def init_wandb(self):
        os.environ.setdefault("WANDB_MODE", self.cfg.wandb.mode)
        return wandb.init(
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.run_name,
            config=asdict(self.cfg),
        )

    def save_checkpoint(self, step: int) -> None:
        ckpt_dir = self.output_dir / f"step-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

    def train(self) -> None:
        cfg = self.train_cfg
        self.optimizer.zero_grad(set_to_none=True)

        for step in trange(1, cfg.num_steps + 1, desc="SFT"):
            total_loss = 0.0
            for _ in range(cfg.gradient_accumulation_steps):
                batch = self.batcher.sample(cfg.batch_size)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss / cfg.gradient_accumulation_steps
                loss.backward()
                total_loss += float(loss.detach().cpu())

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                cfg.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                wandb.log(
                    {
                        "sft/loss": total_loss,
                        "sft/grad_norm": float(grad_norm.detach().cpu()),
                    },
                    step=step,
                )
            if step % cfg.save_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(cfg.num_steps)
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_sft_config(args.config)
    trainer = SFTTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
