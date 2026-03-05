"""
Evaluation utilities for GSM8K‑style problems.

This module provides:
- A single hard‑coded GSM8K‑like example for debugging the pipeline.
- A tiny harness to run the pipeline and compare the derived answer with
  a ground‑truth numeric answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from llm_prolog.llm_client.llm_client import LLMClient, load_openrouter_config
from llm_prolog.pipeline import PipelineConfig, run_pipeline
from llm_prolog.symbolic.types import Fact, PipelineResult


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


def _extract_numeric_answer_from_fact(fact: Fact) -> Optional[int]:
    """
    Very small helper to extract a trailing numeric argument from an answer fact.

    Example expected shapes:
      - answer(8).
      - apples_total(8).
    """
    if not fact.predicate.args:
        return None
    last = fact.predicate.args[-1]
    if last.is_variable:
        return None
    try:
        return int(last.name)
    except ValueError:
        return None



def format_single_result(example : GSM8KExample, result : PipelineResult):
    print("*"*10)
    print("Problem: ", example.problem)
    print("*"*10)

    print(result)

    print("="*20)
    numeric_answer: Optional[int] = None
    if result.answer_premise and isinstance(result.answer_premise.clause, Fact):
        numeric_answer = _extract_numeric_answer_from_fact(result.answer_premise.clause)
    print("Derived numeric answer:", numeric_answer)
    print("Ground truth:", example.ground_truth)
    if numeric_answer is not None:
        print("Match:", numeric_answer == example.ground_truth)
    else:
        print("Match: False (no numeric answer extracted)")

def run_single_example(example: GSM8KExample = EXAMPLE_1) -> None:
    """Run the full pipeline on a single GSM8K‑like example and print results."""
    TEMPERATURE = 0.5

    llm_config = load_openrouter_config(temperature=TEMPERATURE)

    llm = LLMClient(llm_config)
    cfg = PipelineConfig(max_steps=5, explain=True)

    result = run_pipeline(
        problem=example.problem,
        llm=llm,
        config=cfg,
    )

    format_single_result(example, result)


def evaluate_examples(
    examples: Iterable[GSM8KExample],
    *,
    max_steps: int = 8,
) -> None:
    """
    Run the pipeline over a collection of examples and print a simple
    accuracy summary.
    """
    llm = LLMClient()
    cfg = PipelineConfig(max_steps=max_steps, explain=False)

    total = 0
    correct = 0
    for ex in examples:
        total += 1
        result = run_pipeline(
            problem=ex.problem,
            llm=llm,
            config=cfg,
        )

        numeric_answer: Optional[int] = None
        if result.answer_premise and isinstance(result.answer_premise.clause, Fact):
            numeric_answer = _extract_numeric_answer_from_fact(result.answer_premise.clause)

        is_correct = numeric_answer == ex.ground_truth
        if is_correct:
            correct += 1

        format_single_result(ex, result)
        print("-----")


    if total > 0:
        accuracy = correct / total
        print("=====")
        print(f"Total examples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.3f}")
    else:
        print("No examples to evaluate.")


if __name__ == "__main__":
    # For now, run the single example; users can call evaluate_examples
    # from elsewhere with a list of GSM8KExample instances.
    # run_single_example(example=EXAMPLE_2)

    evaluate_examples(examples=[
        # EXAMPLE_1, EXAMPLE_2, 
        EXAMPLE_3], max_steps=20)