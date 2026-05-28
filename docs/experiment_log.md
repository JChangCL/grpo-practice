# GRPO GSM8K Experiment Log

This document tracks the GRPO experiments run on `Qwen/Qwen2.5-0.5B-Instruct` with GSM8K.

## Setup

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `openai/gsm8k`
- Eval split: `test`
- Eval sample size: 100 examples
- Training method: GRPO with reference-model KL penalty, PPO-style ratio clipping, group-relative advantages, and W&B logging
- Current reward family: fallback numeric reward

## Baseline

| Model | Split | Examples | Correct | Accuracy |
|---|---:|---:|---:|---:|
| `Qwen/Qwen2.5-0.5B-Instruct` | test | 100 | 36 | 0.36 |

## Experiment Summary

| Experiment | Key Setting | Checkpoint | Correct | Accuracy | Result |
|---|---|---:|---:|---:|---|
| Baseline | Original model, no GRPO | none | 36 | 0.36 | Reference baseline |
| Run 1 | fallback reward, `t256`, `lr=5e-6`, `kl=0.04`, `steps=100` | step100 | 34 | 0.34 | Worse than baseline |
| Run 2 | fallback reward, `t256`, `lr=2e-6`, `kl=0.08`, `steps=200` | step50 | 37 | 0.37 | Slight improvement |
| Run 2 | same as above | step100 | 41 | 0.41 | Best checkpoint so far |
| Run 2 | same as above | step200 | 12 | 0.12 | Collapsed after too much training |
| Run 3 | fallback reward, `t256`, `lr=1e-6`, `kl=0.12`, `steps=125` | step75 | 36 | 0.36 | Same as baseline |
| Run 3 | same as above | step100 | 35 | 0.35 | Slightly worse than baseline |
| Run 4 | fallback reward, `t256`, `lr=1.5e-6`, `kl=0.10`, `steps=125` | step75 | 30 | 0.30 | Bad |
| Run 4 | same as above | step100 | 28 | 0.28 | Bad |
| Run 5 | format-weighted reward, `t256`, `lr=2e-6`, `kl=0.08`, `steps=125` | step100 | 35 | 0.35 | Format pressure hurt performance |

## Current Best

Current best checkpoint:

```text
Run 2 step100
accuracy = 0.41
baseline = 0.36
absolute improvement = +0.05
```

## Observations

- Run 1 had reward signal, but the update appeared too aggressive and did not improve eval accuracy.
- Run 2 found a useful early checkpoint at step100, but continued training to step200 caused severe collapse.
- Run 3 controlled KL very strongly, but it was too conservative and did not improve over baseline.
- Run 4 used an intermediate LR/KL setting, but eval performance degraded.
- Run 5 changed only the reward weighting toward formatted answers. It did not improve eval accuracy, suggesting that stronger format pressure was premature for this model.
- W&B reward curves alone were not enough to judge success. GSM8K eval accuracy was necessary because some runs showed reward signal without generalizing.

## Interpretation

The best result so far came from Run 2:

```yaml
learning_rate: 0.000002
kl_beta: 0.08
max_prompt_length: 512
max_new_tokens: 256
```

However, Run 2 also collapsed after step100. This suggests that the current reward provides useful signal early, but longer optimization can push the policy into behavior that does not generalize to the GSM8K test subset.

## Run 5 Outcome

Run 5 kept Run 2's training strength, but changed the reward weights to value formatted answers more strongly.

Config:

```yaml
train:
  num_steps: 125
  batch_size: 2
  num_generations: 4
  max_prompt_length: 512
  max_new_tokens: 256
  learning_rate: 0.000002
  kl_beta: 0.08
  save_every: 25

wandb:
  run_name: qwen0.5b-gsm8k-formatweighted-t256-lr2e6-kl008-run5
```

Reward direction:

```text
formatted correct: +1.2
formatted wrong:   -0.6
fallback correct:  +0.25
fallback wrong:    -0.2
format bonus:      +0.3
format missing:    -0.1
length reward:     +0.1 / -0.1
```

Result:

```text
Run 5 step100 accuracy = 0.35
Baseline accuracy = 0.36
Run 2 step100 accuracy = 0.41
```

Interpretation:

```text
Increasing format pressure too early hurt eval performance.
The model still benefited more from the original numeric fallback reward.
```

## Next Direction

The best pure-GRPO result remains Run 2 step100. Further LR/KL tuning did not beat it, and format-weighted reward underperformed. The next high-leverage direction is to add a short GSM8K supervised fine-tuning stage before GRPO:

```text
base model -> GSM8K SFT -> eval SFT -> GRPO from SFT checkpoint
```

This should give the model a better starting point for GSM8K reasoning and answer formatting before RL optimization.

Initial SFT plan:

```text
SFT config: configs/sft_gsm8k.yaml
SFT output: outputs/sft-qwen-0.5b/step-300
GRPO-after-SFT config: configs/grpo_after_sft.yaml
```

Colab run order:

```text
1. Pull commit 1595fe4 or newer.
2. Run SFT with configs/sft_gsm8k.yaml.
3. Evaluate SFT step100, step200, and step300 on the same 100-example GSM8K test subset.
4. Copy all SFT checkpoints and eval_outputs to Google Drive.
5. If any SFT checkpoint beats baseline accuracy 0.36, run GRPO-after-SFT from the best SFT checkpoint.
```

Decision rule:

```text
Do not assume step300 is best. The pure-GRPO runs already showed that earlier checkpoints can be better than later checkpoints.
```
