from __future__ import annotations

import re


NUMERIC_RE = r"[-+]?\d[\d,]*(?:\.\d+)?"
ANSWER_RE = re.compile(rf"<answer>\s*({NUMERIC_RE})\s*</answer>")
ANY_NUMBER_RE = re.compile(NUMERIC_RE)


def extract_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if not match:
        return None
    return match.group(1).strip().replace(",", "")


def extract_last_number(text: str) -> str | None:
    matches = ANY_NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip().replace(",", "")


def extract_numeric_prediction(text: str) -> str | None:
    return extract_answer(text) or extract_last_number(text)


def normalize_number(text: str) -> float:
    return float(text.strip().replace(",", ""))


def format_reward(completion: str) -> float:
    return 0.2 if extract_answer(completion) is not None else -0.2


def math_reward(completion: str, expected: str) -> float:
    formatted_answer = extract_answer(completion)
    fallback_answer = extract_last_number(completion)

    try:
        expected_value = normalize_number(expected)
    except ValueError:
        return 0.0

    if formatted_answer is not None:
        try:
            return 1.0 if normalize_number(formatted_answer) == expected_value else -0.5
        except ValueError:
            return -0.5

    if fallback_answer is not None:
        try:
            return 0.5 if normalize_number(fallback_answer) == expected_value else -0.2
        except ValueError:
            return -0.2

    return 0.0


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
