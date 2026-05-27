# Full GRPO Training Skeleton

這是一個「小模型、完整機制」的 GRPO 實作骨架，保留：

- group-relative advantage normalization
- old-policy logprob snapshot
- PPO-style ratio clipping
- reference-model KL penalty
- token-level loss masking
- W&B metrics / histogram logging
- checkpoint save / resume-friendly config

預設模型是 `Qwen/Qwen2.5-0.5B-Instruct`。你可以改成任何 Hugging Face causal LM 或本機模型路徑。

## 快速開始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
wandb login
python -m grpo_full.train --config configs/grpo_tiny.yaml
```

如果不想上傳到 W&B：

```bash
WANDB_MODE=offline python -m grpo_full.train --config configs/grpo_tiny.yaml
```

## 專案結構

```text
configs/grpo_tiny.yaml       # 小模型完整 GRPO 設定
docs/outline.md              # 分階段設計與實作 outline
grpo_full/config.py          # dataclass config + YAML loader
grpo_full/data.py            # prompt dataset
grpo_full/rewards.py         # 可替換 reward functions
grpo_full/grpo.py            # GRPO sampling / loss / trainer
grpo_full/train.py           # CLI entrypoint
requirements.txt             # dependencies
```

## 重要指標

W&B 會紀錄：

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

## 注意

這是研究/教學用最小完整實作，不是高吞吐量分散式訓練器。若要擴大到正式訓練，下一步通常是加入 Accelerate/FSDP、vLLM rollout、LoRA/QLoRA、reward model server、eval harness。
