# 完整 GRPO Outline

## 目標

做一個可讀、可改、可跑的小型 GRPO 訓練框架。模型可以小，但演算法不能閹割：KL、clip、reference model、old logprobs、group-relative advantages、W&B 分析全部保留。

## Phase 1: 最小完整訓練閉環

1. 載入 policy model、reference model、tokenizer。
2. 從 prompt dataset 取 batch。
3. 每個 prompt 產生 `num_generations` 個 completion，形成 group。
4. 對每個 completion 計算 reward。
5. 在同一 group 內做 normalized advantage：
   `A_i = (r_i - mean(group_rewards)) / (std(group_rewards) + eps)`。
6. 記錄 sampling 當下的 old policy logprob。
7. 用目前 policy 重新計算 logprob。
8. 用 reference model 計算 ref logprob。
9. 計算 token-level ratio：
   `ratio = exp(new_logprob - old_logprob)`。
10. 計算 clipped objective：
    `min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`。
11. 計算 per-token KL proxy：
    `kl = exp(ref_logprob - new_logprob) - (ref_logprob - new_logprob) - 1`。
12. 最終 loss：
    `loss = -policy_objective + beta * kl`。
13. mask 掉 prompt token 與 padding token，只對 completion token 更新。
14. optimizer step、gradient clipping、checkpoint、W&B logging。

## Phase 2: Reward 模組化

目前先內建 rule-based reward，方便立即測：

- `format_reward`: 是否包含 `<answer>...</answer>`
- `math_reward`: 能否答對簡單算術
- `length_reward`: 避免過短或過長

之後可替換成：

- reward model
- LLM judge
- unit test / verifier
- tool-based evaluator

## Phase 3: W&B 分析

需要看三種東西：

- learning signal: reward mean/std、advantage distribution
- policy stability: KL、ratio、clip fraction
- qualitative samples: prompt/completion/reward table

## Phase 4: 擴充方向

- LoRA/QLoRA: 降低顯存。
- Accelerate/FSDP: 多 GPU。
- vLLM rollout: 提高 generation throughput。
- Eval harness: 固定 benchmark prompt 做回歸。
- Reward server: 讓 verifier 與 trainer 解耦。
