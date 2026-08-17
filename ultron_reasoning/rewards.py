from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class RewardResult:
    total: float
    correctness: float
    format_score: float
    verifier_score: float


def format_reward(text: str) -> float:
    text = text.strip()
    if not text:
        return 0.0
    return 1.0 if "final" in text.lower() else 0.25


def compute_reward(response: str, verifier: Callable[[str], float]) -> RewardResult:
    correctness = float(verifier(response))
    formatting = format_reward(response)
    total = 0.8 * correctness + 0.2 * formatting
    return RewardResult(total, correctness, formatting, correctness)


def batch_rewards(responses: Sequence[str], verifier: Callable[[str], float]):
    return [compute_reward(x, verifier) for x in responses]
