# Full GRPO Outline

## Goal

Build a readable, editable, runnable GRPO training framework for small models while keeping the algorithm complete: KL, clipping, reference model, old logprobs, group-relative advantages, and W&B analysis are all included.

## Phase 1: Minimal Complete Training Loop

1. Load the policy model, reference model, and tokenizer.
2. Sample a batch from the prompt dataset.
3. Generate `num_generations` completions for each prompt to form a group.
4. Compute a reward for each completion.
5. Normalize advantages inside each group:
   `A_i = (r_i - mean(group_rewards)) / (std(group_rewards) + eps)`.
6. Record old policy logprobs from the sampling pass.
7. Recompute logprobs with the current policy.
8. Compute reference logprobs with the reference model.
9. Compute the token-level ratio:
   `ratio = exp(new_logprob - old_logprob)`.
10. Compute the clipped objective:
    `min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`.
11. Compute the per-token KL proxy:
    `kl = exp(ref_logprob - new_logprob) - (ref_logprob - new_logprob) - 1`.
12. Compute the final loss:
    `loss = -policy_objective + beta * kl`.
13. Mask prompt tokens and padding tokens so only completion tokens receive RL updates.
14. Run optimizer steps, gradient clipping, checkpoint saving, and W&B logging.

## Phase 2: Modular Rewards

The current trainer uses GSM8K training data and rule-based verifier rewards:

- `format_reward`: checks whether the completion contains `<answer>...</answer>`
- `math_reward`: checks whether the answer matches the GSM8K final answer after `####`
- `length_reward`: discourages completions that are too short or too long

Future reward sources can include:

- reward models
- LLM judges
- unit tests or verifiers
- tool-based evaluators

## Phase 3: W&B Analysis

Track three categories:

- learning signal: reward mean/std and advantage distribution
- policy stability: KL, ratio, and clip fraction
- qualitative samples: prompt/completion/reward tables

## Phase 4: Extensions

- LoRA/QLoRA: reduce memory usage.
- Accelerate/FSDP: scale to multiple GPUs.
- vLLM rollout: improve generation throughput.
- Eval harness: run fixed benchmark prompts for regression checks.
- Reward server: decouple verifiers from the trainer.
