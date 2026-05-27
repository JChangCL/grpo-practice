# Full GRPO Training Skeleton

This project is a small-model GRPO training skeleton with the full core mechanism intact:

- group-relative advantage normalization
- old-policy logprob snapshots
- PPO-style ratio clipping
- reference-model KL penalty
- token-level loss masking
- W&B metrics and histogram logging
- checkpoint saving

The default model is `Qwen/Qwen2.5-0.5B-Instruct`, and the default dataset is `openai/gsm8k`. You can replace the model with any Hugging Face causal LM or a local model path.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
wandb login
python -m grpo_full.train --config configs/grpo_tiny.yaml
```

To keep W&B logs offline:

```bash
WANDB_MODE=offline python -m grpo_full.train --config configs/grpo_tiny.yaml
```

## Project Structure

```text
configs/grpo_tiny.yaml       # Small-model full GRPO config
docs/outline.md              # Phased design and implementation outline
grpo_full/config.py          # Dataclass config and YAML loader
grpo_full/data.py            # GSM8K prompt dataset
grpo_full/rewards.py         # Replaceable reward functions
grpo_full/grpo.py            # GRPO sampling, loss, and trainer
grpo_full/eval.py            # GSM8K baseline/checkpoint evaluator
grpo_full/train.py           # CLI entrypoint
requirements.txt             # Python dependencies
```

## Dataset

The dataset is configured in `configs/grpo_tiny.yaml`:

```yaml
dataset:
  name: openai/gsm8k
  config_name: main
  split: train
  max_examples: 2000
```

GSM8K answers usually include reasoning text followed by `#### final_answer`. The trainer extracts the value after `####` and uses it as the reward target.

## Key Metrics

W&B logs:

- `train/loss`
- `train/policy_loss`
- `train/kl_loss`
- `train/mean_reward`
- `train/reward_std`
- `train/mean_kl`
- `train/clip_fraction`
- `train/mean_ratio`
- `train/mean_completion_len`
- `samples/table`

## Evaluation

Evaluate the baseline model on GSM8K:

```bash
python -m grpo_full.eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --split test \
  --max-examples 100 \
  --output-jsonl eval_outputs/baseline_gsm8k.jsonl
```

Evaluate a GRPO checkpoint:

```bash
python -m grpo_full.eval \
  --model outputs/grpo-qwen-0.5b/step-100 \
  --split test \
  --max-examples 100 \
  --output-jsonl eval_outputs/grpo_step100_gsm8k.jsonl
```

Compare `accuracy`, `correct`, and the JSONL records to inspect where the model improves or fails.

## Notes

This is a minimal research and learning implementation, not a high-throughput distributed trainer. For larger runs, the next steps are usually Accelerate/FSDP, vLLM rollouts, LoRA/QLoRA, a reward model server, and an evaluation harness.
