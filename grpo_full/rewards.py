from __future__ import annotations

import re


ANSWER_RE = re.compile(r"<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>")


def extract_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def format_reward(completion: str) -> float:
    return 0.2 if extract_answer(completion) is not None else -0.2


def math_reward(completion: str, expected: str) -> float:
    answer = extract_answer(completion)
    if answer is None:
        return 0.0
    try:
        return 1.0 if float(answer) == float(expected) else -0.5
    except ValueError:
        return -0.5


def length_reward(completion: str, min_chars: int = 12, max_chars: int = 600) -> float:
    n_chars = len(completion.strip())
    if min_chars <= n_chars <= max_chars:
        return 0.1
    return -0.1


def total_reward(completion: str, expected: str) -> float:
    return (
        format_reward(completion)
        + math_reward(completion, expected)
        + length_reward(completion)
    )
