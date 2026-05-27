from __future__ import annotations

import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.optim import AdamW
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_full.config import AppConfig
from grpo_full.data import ArithmeticPromptDataset, PromptExample
from grpo_full.rewards import total_reward


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def sequence_logprobs(model, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logprobs = F.log_softmax(logits, dim=-1)
    return logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def build_generation_masks(
    generated: Tensor,
    prompt_attention_mask: Tensor,
    prompt_width: int,
    eos_token_id: int | None,
) -> tuple[Tensor, Tensor]:
    attention_mask = torch.zeros_like(generated, dtype=torch.long)
    attention_mask[:, :prompt_width] = prompt_attention_mask
    completion_logprob_mask = torch.zeros(
        generated.shape[0],
        generated.shape[1] - 1,
        dtype=torch.bool,
        device=generated.device,
    )

    for row_idx in range(generated.shape[0]):
        completion_ids = generated[row_idx, prompt_width:]
        valid_len = completion_ids.numel()
        if eos_token_id is not None:
            eos_positions = (completion_ids == eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                valid_len = int(eos_positions[0].item()) + 1

        if valid_len > 0:
            attention_mask[row_idx, prompt_width : prompt_width + valid_len] = 1
            start = prompt_width - 1
            end = prompt_width + valid_len - 1
            completion_logprob_mask[row_idx, start:end] = True

    return attention_mask, completion_logprob_mask


class GRPOTrainer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.train_cfg = cfg.train
        set_seed(cfg.seed)

        self.device = get_device()
        self.dtype = select_dtype(self.device)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.reference = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.reference.eval()
        self.reference.requires_grad_(False)

        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        self.dataset = ArithmeticPromptDataset(seed=cfg.seed)
        self.run = self._init_wandb()

    def _init_wandb(self):
        os.environ.setdefault("WANDB_MODE", self.cfg.wandb.mode)
        return wandb.init(
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.run_name,
            config=asdict(self.cfg),
        )

    def _tokenize_prompts(self, examples: list[PromptExample]) -> dict[str, Tensor]:
        prompts = [example.prompt for example in examples]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.train_cfg.max_prompt_length,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    @torch.no_grad()
    def rollout(self, examples: list[PromptExample]) -> dict[str, object]:
        cfg = self.train_cfg
        repeated_examples = [
            example
            for example in examples
            for _ in range(cfg.num_generations)
        ]
        prompt_tensors = self._tokenize_prompts(repeated_examples)
        prompt_width = prompt_tensors["input_ids"].shape[1]

        self.policy.eval()
        generated = self.policy.generate(
            **prompt_tensors,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        attention_mask, completion_mask = build_generation_masks(
            generated=generated,
            prompt_attention_mask=prompt_tensors["attention_mask"],
            prompt_width=prompt_width,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        old_logprobs = sequence_logprobs(self.policy, generated, attention_mask)

        completions: list[str] = []
        rewards: list[float] = []
        for seq, example in zip(generated, repeated_examples, strict=True):
            completion_ids = seq[prompt_width:]
            completion = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            )
            completions.append(completion)
            rewards.append(total_reward(completion, example.answer))

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        reward_groups = reward_tensor.view(len(examples), cfg.num_generations)
        group_mean = reward_groups.mean(dim=1, keepdim=True)
        group_std = reward_groups.std(dim=1, keepdim=True, unbiased=False)
        advantages = (reward_groups - group_mean) / (group_std + cfg.advantage_eps)
        advantages = advantages.reshape(-1)

        return {
            "input_ids": generated,
            "attention_mask": attention_mask,
            "old_logprobs": old_logprobs.detach(),
            "advantages": advantages.detach(),
            "completion_mask": completion_mask,
            "rewards": reward_tensor.detach(),
            "completions": completions,
            "examples": repeated_examples,
        }

    def train_step(self, batch: dict[str, object]) -> dict[str, object]:
        cfg = self.train_cfg
        self.policy.train()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        old_logprobs = batch["old_logprobs"]
        advantages = batch["advantages"].to(self.dtype).unsqueeze(1)
        completion_mask = batch["completion_mask"]

        new_logprobs = sequence_logprobs(self.policy, input_ids, attention_mask)
        with torch.no_grad():
            ref_logprobs = sequence_logprobs(self.reference, input_ids, attention_mask)

        log_ratio = new_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * advantages
        policy_objective = torch.minimum(unclipped, clipped)
        policy_loss = -masked_mean(policy_objective, completion_mask)

        ref_policy_log_ratio = ref_logprobs - new_logprobs
        kl = torch.exp(ref_policy_log_ratio) - ref_policy_log_ratio - 1.0
        kl_loss = masked_mean(kl, completion_mask)

        loss = policy_loss + cfg.kl_beta * kl_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            cfg.max_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        rewards = batch["rewards"]
        masked_ratio = ratio.detach()[completion_mask]
        masked_kl = kl.detach()[completion_mask]
        clip_fraction = (
            (masked_ratio.lt(1.0 - cfg.clip_range) | masked_ratio.gt(1.0 + cfg.clip_range))
            .float()
            .mean()
            .item()
            if masked_ratio.numel() > 0
            else 0.0
        )
        completion_lens = completion_mask.sum(dim=1).float()

        return {
            "train/loss": float(loss.detach().cpu()),
            "train/policy_loss": float(policy_loss.detach().cpu()),
            "train/kl_loss": float(kl_loss.detach().cpu()),
            "train/mean_kl": float(masked_mean(kl.detach(), completion_mask).cpu()),
            "train/mean_reward": float(rewards.mean().cpu()),
            "train/reward_std": float(rewards.std(unbiased=False).cpu()),
            "train/mean_ratio": float(masked_ratio.mean().cpu()) if masked_ratio.numel() else math.nan,
            "train/clip_fraction": clip_fraction,
            "train/mean_completion_len": float(completion_lens.mean().cpu()),
            "train/grad_norm": float(grad_norm.detach().cpu()),
            "hist/rewards": wandb.Histogram(rewards.float().cpu().numpy()),
            "hist/advantages": wandb.Histogram(batch["advantages"].float().cpu().numpy()),
            "hist/ratios": wandb.Histogram(masked_ratio.float().cpu().numpy()),
            "hist/token_kl": wandb.Histogram(masked_kl.float().cpu().numpy()),
        }

    def log_samples(self, step: int, batch: dict[str, object]) -> None:
        rows = []
        examples = batch["examples"]
        completions = batch["completions"]
        rewards = batch["rewards"].detach().cpu().tolist()
        for example, completion, reward in zip(examples[:8], completions[:8], rewards[:8], strict=True):
            rows.append([step, example.prompt, completion, example.answer, reward])
        table = wandb.Table(
            columns=["step", "prompt", "completion", "expected", "reward"],
            data=rows,
        )
        wandb.log({"samples/table": table}, step=step)

    def save_checkpoint(self, step: int) -> None:
        ckpt_dir = self.output_dir / f"step-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

    def train(self) -> None:
        cfg = self.train_cfg
        for step in trange(1, cfg.num_steps + 1, desc="GRPO"):
            examples = self.dataset.sample(cfg.batch_size)
            batch = self.rollout(examples)
            metrics = self.train_step(batch)

            if step % cfg.log_every == 0:
                wandb.log(metrics, step=step)
                self.log_samples(step, batch)
            if step % cfg.save_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(cfg.num_steps)
        wandb.finish()
