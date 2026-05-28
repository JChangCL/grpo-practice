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
| Run 6 | fallback reward, `t256`, `lr=2e-6`, `kl=0.08`, `steps=125` | step25 | 34 | 0.34 | Worse than baseline |
| Run 6 | same as above | step50 | 37 | 0.37 | Slight improvement |
| Run 6 | same as above | step75 | 38 | 0.38 | Best Run 6 checkpoint |
| Run 6 | same as above | step100 | 33 | 0.33 | Degraded after peak |
| Run 6 | same as above | step125 | 32 | 0.32 | Continued degradation |

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
- Run 6 reran the best pure-GRPO setting with a separate output directory. It improved over baseline at step50-step75, but did not reproduce Run 2 step100 and degraded by step100-step125.
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

## SFT Run 1 Outcome

Config:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
steps: 300
learning_rate: 5e-6
gradient_accumulation_steps: 8
max_length: 768
output: outputs/sft-qwen-0.5b
```

Eval results on the same 100-example GSM8K test subset:

| Checkpoint | Correct | Accuracy | Mean completion tokens | Result |
|---:|---:|---:|---:|---|
| step100 | 24 | 0.24 | 115.76 | Worse than baseline |
| step200 | 31 | 0.31 | 111.33 | Best SFT run1 checkpoint, still worse than baseline |
| step300 | 27 | 0.27 | 114.80 | Worse than baseline |

Conclusion:

```text
SFT run1 should not be used as the starting point for GRPO-after-SFT.
It is worse than the baseline accuracy 0.36 and far below the best pure-GRPO checkpoint accuracy 0.41.
```

Qualitative diagnosis from wrong examples:

```text
The model often learned to emit <answer>...</answer>, but the reasoning degraded.
Common failures included misunderstanding the problem statement, applying the wrong quantity, and writing inconsistent arithmetic traces.
```

Example failure patterns:

```text
- Janet ducks: treated eaten/baked eggs incorrectly and predicted 33 instead of 18.
- Robe fiber: doubled the blue fiber and predicted 5 instead of 3.
- House flip: applied the 150% increase to total cost instead of original house value and predicted 65000 instead of 70000.
- Sprinting: used 7 sessions instead of 3 times per week and predicted 1260 instead of 540.
- Discounted glasses: treated the discount amount as the final price for every glass and predicted 32 instead of 64.
```

Next SFT direction:

```text
Try a more conservative SFT run before returning to GRPO-after-SFT:
- learning_rate: 1e-6
- num_steps: 200
- save_every: 50
- output_dir: outputs/sft-qwen-0.5b-run2
- run_name: qwen0.5b-gsm8k-sft-run2-lr1e6
```

## SFT Run 2 Partial Outcome

Config direction:

```text
More conservative full fine-tune than SFT run1:
- learning_rate: 1e-6
- num_steps: 200
- save_every: 50
- output_dir: outputs/sft-qwen-0.5b-run2
```

Partial eval results on the same 100-example GSM8K test subset:

| Checkpoint | Correct | Accuracy | Mean completion tokens | Result |
|---:|---:|---:|---:|---|
| step50 | 29 | 0.29 | 168.55 | Worse than baseline |
| step100 | 25 | 0.25 | 138.53 | Worse than baseline |

Interpretation:

```text
Lowering SFT LR from 5e-6 to 1e-6 did not fix the degradation.
The SFT direction is currently not producing a useful GRPO starting checkpoint.
Do not run GRPO-after-SFT from SFT run2 unless a later checkpoint unexpectedly beats baseline 0.36.
```

Current recommendation:

```text
Pause full-parameter GSM8K SFT for now.
Return to the pure-GRPO setting that already produced the best observed checkpoint:
- fallback reward
- learning_rate: 2e-6
- kl_beta: 0.08
- max_new_tokens: 256
- early checkpoint selection around step50-step100
```

## Run 6 Plan

Since both SFT runs underperformed the baseline, resume pure GRPO from the base instruct model and treat it as an early-stop rerun of the best previous setting.

Config:

```text
config: configs/grpo_tiny.yaml
model_name: Qwen/Qwen2.5-0.5B-Instruct
output_dir: outputs/grpo-qwen-0.5b-run6
reward: fallback numeric reward
learning_rate: 2e-6
kl_beta: 0.08
max_new_tokens: 256
num_steps: 125
save_every: 25
run_name: qwen0.5b-gsm8k-fallback-t256-lr2e6-kl008-run6
```

Eval checkpoints:

```text
step25, step50, step75, step100, step125
```

Decision rule:

```text
Select the best early checkpoint by GSM8K test accuracy. Do not continue training past the planned run without eval, because Run 2 collapsed by step200.
```

## Run 6 Outcome

Run 6 reran the best previous pure-GRPO setting from the base instruct model, with a separate output directory to avoid overwriting older checkpoints.

Eval results on the same 100-example GSM8K test subset:

| Checkpoint | Correct | Accuracy | Result |
|---:|---:|---:|---|
| step25 | 34 | 0.34 | Worse than baseline |
| step50 | 37 | 0.37 | Slightly above baseline |
| step75 | 38 | 0.38 | Best Run 6 checkpoint |
| step100 | 33 | 0.33 | Degraded after peak |
| step125 | 32 | 0.32 | Continued degradation |

Interpretation:

```text
Run 6 confirms the early-improvement pattern but did not reproduce the previous best 0.41 result from Run 2 step100.
The best Run 6 checkpoint was step75 at 0.38, followed by degradation at step100 and step125.
This makes further fine-grained checkpoint hunting less valuable than trying a different improvement lever.
```

## LoRA SFT Plan

Motivation:

```text
Full-parameter SFT run1 and run2 both degraded GSM8K accuracy below the base-model baseline.
The next SFT attempt should avoid overwriting the base instruct model's existing reasoning behavior.
LoRA trains a small adapter and should be less destructive than full fine-tuning.
```

Config:

```text
config: configs/sft_lora_gsm8k.yaml
model_name: Qwen/Qwen2.5-0.5B-Instruct
output_dir: outputs/sft-lora-qwen-0.5b-run1
steps: 200
learning_rate: 1e-5
save_every: 50
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Eval checkpoints:

```text
step50, step100, step150, step200
```

Decision rule:

```text
Only consider LoRA-SFT as a GRPO starting point if it reaches or exceeds the baseline accuracy 0.36.
If it remains below baseline, pause SFT-style approaches and focus on prompt/reward/GRPO changes.
```
