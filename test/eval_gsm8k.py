"""
Evaluation for GSM8K‑style math word problems.

Uses the generic evaluation harness from eval_common with:
- GSM8KExample dataclass
- Validator: extract numeric answer from the answer premise's clause
- Success measure: exact match of integer answer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from llm_prolog.symbolic.types import Fact, PipelineResult

from test.eval_common import evaluate_examples, run_single_example


@dataclass(frozen=True)
class GSM8KExample:
    problem: str
    ground_truth: int


EXAMPLE_1 = GSM8KExample(
    problem=(
        "Alice has 3 apples. She buys 5 more apples at the store. "
        "How many apples does Alice have now?"
    ),
    ground_truth=8,
)

EXAMPLE_2 = GSM8KExample(
    problem=(
        "Kendra has 3 more than twice as many berries as Sam. Sam has half as many berries as Martha. "
        "If Martha has 40 berries, how many berries does Kendra have?"
    ),
    ground_truth=43,
)

EXAMPLE_3 = GSM8KExample(
    problem=(
        "Katy makes coffee using teaspoons of sugar and cups of water in the ratio of 7:13. "
        "If she used a total of 120 teaspoons of sugar and cups of water, calculate the number of teaspoonfuls of sugar she used."
    ),
    ground_truth=42,
)

EXAMPLE_4 = GSM8KExample(
    problem=(
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."
        "How many clips did Natalia sell altogether in April and May?"
    ),
    ground_truth=72,
)

EXAMPLE_5 = GSM8KExample(
    problem=(
        "Tim rides his bike back and forth to work for each of his 5 workdays."
        "His work is 20 miles away. He also goes for a weekend bike ride of 200 miles."
        "If he can bike at 25 mph how much time does he spend biking a week?"
    ),
    ground_truth=16,
)

EXAMPLE_6 = GSM8KExample(
    problem=(
        "Brennan was researching his school project and had to download files from the internet"
        "to his computer to use for reference. After downloading 800 files, he deleted 70% of them"
        "because they were not helpful. He downloaded 400 more files but again realized that"
        "3/5 of them were irrelevant. How many valuable files was he left with after deleting"
        "the unrelated files he downloaded in the second round?"
    ),
    ground_truth=400,
)


def gsm8k_validator(result: PipelineResult) -> Optional[int]:
    """Extract the derived numeric answer from the pipeline result (GSM8K format)."""
    answer = result.extract_answer_constant()
    if answer is None:
        return None
    try:
        return int(answer)
    except ValueError:
        return None


def gsm8k_success_measure(example: GSM8KExample, obtained: Optional[int]) -> bool:
    """Success = exact match of integer answer."""
    return obtained is not None and obtained == example.ground_truth


def run_single_example_gsm8k(example: GSM8KExample = EXAMPLE_1) -> None:
    """Run the pipeline on a single GSM8K example and print results."""
    run_single_example(
        example,
        gsm8k_validator,
        gsm8k_success_measure,
        show_derived_label="Derived numeric answer",
        show_expected_label="Ground truth",
    )


def evaluate_gsm8k(
    examples: Iterable[GSM8KExample],
    *,
    max_steps: int = 8,
) -> None:
    """Run the pipeline over GSM8K examples and print accuracy summary."""
    evaluate_examples(
        examples,
        gsm8k_validator,
        gsm8k_success_measure,
        max_steps=max_steps,
        show_derived_label="Derived numeric answer",
        show_expected_label="Ground truth",
    )


if __name__ == "__main__":
    evaluate_gsm8k(
        examples=[
            EXAMPLE_1,
            EXAMPLE_5,
            EXAMPLE_6,
        ],
        max_steps=20,
    )
