from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step, "
    "then put only the final answer inside <answer>...</answer>."
)

PROMPT_STYLES = {
    "default": SYSTEM_PROMPT,
    "concise_xml": "Solve the math problem. End with <answer>number</answer>.",
    "careful_no_xml": (
        "You are a careful math assistant. Solve the problem step by step, "
        "then give the final numeric answer."
    ),
    "direct": "Answer the math problem. Give the final numeric answer.",
}


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    answer: str
    question: str


def format_chat_prompt(question: str, prompt_style: str = "default") -> str:
    try:
        system_prompt = PROMPT_STYLES[prompt_style]
    except KeyError as exc:
        valid = ", ".join(sorted(PROMPT_STYLES))
        raise ValueError(f"Unknown prompt_style={prompt_style!r}. Valid styles: {valid}") from exc

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_gsm8k_answer(answer_text: str) -> str:
    marker = "####"
    if marker in answer_text:
        return answer_text.split(marker, maxsplit=1)[1].strip().replace(",", "")
    return answer_text.strip().replace(",", "")


def format_gsm8k_sft_answer(answer_text: str) -> str:
    marker = "####"
    if marker not in answer_text:
        return answer_text.strip()

    reasoning, final_answer = answer_text.split(marker, maxsplit=1)
    final_answer = final_answer.strip().replace(",", "")
    reasoning = reasoning.strip()
    if reasoning:
        return f"{reasoning}\n<answer>{final_answer}</answer>"
    return f"<answer>{final_answer}</answer>"


class GSM8KPromptDataset:
    def __init__(
        self,
        name: str = "openai/gsm8k",
        config_name: str = "main",
        split: str = "train",
        max_examples: int | None = None,
        seed: int = 0,
        prompt_style: str = "default",
    ):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "GSM8K loading requires the `datasets` package. "
                "Run `pip install -r requirements.txt` first."
            ) from exc

        raw_dataset = load_dataset(name, config_name, split=split)
        if max_examples is not None:
            raw_dataset = raw_dataset.select(range(min(max_examples, len(raw_dataset))))

        self.rng = random.Random(seed)
        self.prompt_style = prompt_style
        self.examples: list[PromptExample] = []
        for row in raw_dataset:
            example = self._row_to_example(row)
            self.examples.append(example)

        if not self.examples:
            raise ValueError(f"No examples loaded from {name}/{config_name} split={split}.")

    def _row_to_example(self, row: dict[str, Any]) -> PromptExample:
        question = str(row["question"])
        answer = extract_gsm8k_answer(str(row["answer"]))
        return PromptExample(
            prompt=format_chat_prompt(question, self.prompt_style),
            question=question,
            answer=answer,
        )

    def sample(self, batch_size: int) -> list[PromptExample]:
        return self.rng.sample(self.examples, batch_size)


class ArithmeticPromptDataset:
    def __init__(self, size: int = 10_000, seed: int = 0):
        rng = random.Random(seed)
        self.rng = rng
        self.examples: list[PromptExample] = []
        ops = ["+", "-", "*"]
        for _ in range(size):
            a = rng.randint(1, 30)
            b = rng.randint(1, 30)
            op = rng.choice(ops)
            if op == "+":
                answer = a + b
            elif op == "-":
                answer = a - b
            else:
                answer = a * b
            question = f"What is {a} {op} {b}?"
            self.examples.append(
                PromptExample(
                    prompt=format_chat_prompt(question),
                    question=question,
                    answer=str(answer),
                )
            )

    def sample(self, batch_size: int) -> list[PromptExample]:
        return self.rng.sample(self.examples, batch_size)
