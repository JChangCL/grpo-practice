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
configs/sft_gsm8k.yaml       # GSM8K supervised fine-tuning config
configs/sft_lora_gsm8k.yaml  # GSM8K LoRA supervised fine-tuning config
configs/grpo_after_sft.yaml  # GRPO config that starts from the SFT checkpoint
docs/outline.md              # Phased design and implementation outline
grpo_full/config.py          # Dataclass config and YAML loader
grpo_full/data.py            # GSM8K prompt dataset
grpo_full/sft.py             # GSM8K supervised fine-tuning trainer
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

## SFT Then GRPO

On Colab, pull the latest code first:

```bash
%cd /content/grpo-practice
git pull
git log --oneline -3
```

Run a short GSM8K supervised fine-tuning stage:

```bash
python -m grpo_full.sft --config configs/sft_gsm8k.yaml
```

Evaluate all saved SFT checkpoints, not just the final one:

```bash
for step in 100 200 300; do
  python -m grpo_full.eval \
    --model outputs/sft-qwen-0.5b/step-${step} \
    --split test \
    --max-examples 100 \
    --output-jsonl eval_outputs/sft_step${step}_gsm8k.jsonl
done
```

Copy checkpoints and eval outputs to Google Drive before starting the next stage:

```bash
mkdir -p /content/drive/MyDrive/grpo-practice/checkpoints/sft-qwen-0.5b
mkdir -p /content/drive/MyDrive/grpo-practice/eval_outputs

cp -r outputs/sft-qwen-0.5b/step-* /content/drive/MyDrive/grpo-practice/checkpoints/sft-qwen-0.5b/
cp eval_outputs/sft_step*_gsm8k.jsonl /content/drive/MyDrive/grpo-practice/eval_outputs/
```

If the SFT checkpoint improves over the baseline `0.36` accuracy, run GRPO from the best SFT checkpoint. The default after-SFT config starts from `outputs/sft-qwen-0.5b/step-300`:

```bash
python -m grpo_full.train --config configs/grpo_after_sft.yaml
```

The `configs/grpo_after_sft.yaml` file expects the SFT checkpoint at `outputs/sft-qwen-0.5b/step-300`.

## LoRA SFT

Full-parameter SFT degraded GSM8K eval accuracy in early experiments. LoRA SFT is a more conservative alternative that trains adapters while keeping the base model mostly frozen:

```bash
pip install -r requirements.txt
python -m grpo_full.sft --config configs/sft_lora_gsm8k.yaml
```

If Colab reports an incompatible `torchao` version while PEFT is creating LoRA adapters, rerun `pip install -r requirements.txt` and restart the runtime before training.

Evaluate the LoRA adapter checkpoints with the same eval command. The evaluator automatically detects PEFT adapter checkpoints:

```bash
for step in 50 100 150 200; do
  python -m grpo_full.eval \
    --model outputs/sft-lora-qwen-0.5b-run1/step-${step} \
    --split test \
    --max-examples 100 \
    --output-jsonl eval_outputs/sft_lora_run1_step${step}_gsm8k.jsonl
done
```

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

## Prompt Sweep

Before running more training, compare prompt styles on the base model:

```bash
for style in default concise_xml careful_no_xml direct; do
  python -m grpo_full.eval \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --split test \
    --max-examples 100 \
    --prompt-style ${style} \
    --output-jsonl eval_outputs/baseline_prompt_${style}_gsm8k.jsonl
done
```

The default prompt style is the original project prompt. Other styles are eval-only unless a config or script explicitly uses them.

## Notes

This is a minimal research and learning implementation, not a high-throughput distributed trainer. For larger runs, the next steps are usually Accelerate/FSDP, vLLM rollouts, LoRA/QLoRA, a reward model server, and an evaluation harness.
