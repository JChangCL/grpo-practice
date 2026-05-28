from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_full.data import GSM8KPromptDataset, PROMPT_STYLES
from grpo_full.grpo import get_device, select_dtype
from grpo_full.rewards import extract_numeric_prediction, normalize_number


def is_correct(prediction: str | None, expected: str) -> bool:
    if prediction is None:
        return False
    try:
        return normalize_number(prediction) == normalize_number(expected)
    except ValueError:
        return False


def load_eval_model(model_path: str, dtype: torch.dtype, device: torch.device):
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise ImportError(
                "Evaluating a LoRA checkpoint requires the `peft` package. Run `pip install -r requirements.txt` first."
            ) from exc

        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        return PeftModel.from_pretrained(base_model, model_path).to(device)

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)


def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    device = get_device()
    dtype = select_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_eval_model(args.model, dtype, device)
    model.eval()

    dataset = GSM8KPromptDataset(
        name=args.dataset,
        config_name=args.dataset_config,
        split=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
        prompt_style=args.prompt_style,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    total_completion_tokens = 0

    with output_path.open("w", encoding="utf-8") as f:
        for example in tqdm(dataset.examples, desc="Evaluating"):
            encoded = tokenizer(
                example.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_width = encoded["input_ids"].shape[1]

            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            completion_ids = generated[0, prompt_width:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            prediction = extract_numeric_prediction(completion)
            item_correct = is_correct(prediction, example.answer)

            correct += int(item_correct)
            total += 1
            total_completion_tokens += int(completion_ids.numel())

            record = {
                "question": example.question,
                "expected": example.answer,
                "prediction": prediction,
                "correct": item_correct,
                "completion": completion,
                "prompt_style": args.prompt_style,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    accuracy = correct / max(total, 1)
    mean_completion_tokens = total_completion_tokens / max(total, 1)
    return {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "mean_completion_tokens": mean_completion_tokens,
        "output_jsonl": str(output_path),
        "prompt_style": args.prompt_style,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-style", choices=sorted(PROMPT_STYLES), default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-jsonl", default="eval_outputs/gsm8k_eval.jsonl")
    args = parser.parse_args()

    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
