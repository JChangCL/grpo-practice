from __future__ import annotations

import random
from dataclasses import dataclass


SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step, "
    "then put only the final answer inside <answer>...</answer>."
)


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    answer: str


class ArithmeticPromptDataset:
    def __init__(self, size: int = 10_000, seed: int = 0):
        rng = random.Random(seed)
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
            user = f"What is {a} {op} {b}?"
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            self.examples.append(PromptExample(prompt=prompt, answer=str(answer)))

    def sample(self, batch_size: int) -> list[PromptExample]:
        return random.sample(self.examples, batch_size)
