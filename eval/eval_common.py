"""
Generic evaluation harness for LLM‑Prolog pipeline runs across datasets.

This module provides dataset-agnostic evaluation logic:
- Formatting of single-result and batch summaries
- Running the pipeline and comparing derived answers to ground truth

Each dataset supplies:
- An example type (e.g. a dataclass with problem + ground truth)
- A validator: extracts the "obtained answer" in a canonical form from PipelineResult
- A success measure: compares (example, obtained) and returns whether the answer is correct
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, TypeVar

from llm_prolog.llm_client.llm_client import LLMClient, load_openrouter_config
from llm_prolog.pipeline import PipelineConfig, run_symbolic_hybrid_pipeline


# Type for the value extracted by the validator (e.g. int, str, set of facts).
T = TypeVar("T")


def format_single_result(
    example: Any,
    result: Any,
    validator: Callable[[Any], Optional[T]],
    success_measure: Callable[[Any, Optional[Any]], bool],
    *,
    show_derived_label: str = "Derived answer",
    show_expected_label: str = "Expected",
) -> None:
    """
    Print a single evaluation result in a consistent format.

    - example: dataset-specific example (e.g. GSM8KExample)
    - result: pipeline result
    - validator: extracts obtained value from result
    - success_measure: (example, obtained) -> True if correct
    - show_derived_label / show_expected_label: optional labels for printed output
    """
    print("*" * 10)
    print("Problem:", getattr(example, "problem", example))
    print("*" * 10)

    print(result)

    print("=" * 20)
    obtained = validator(result)
    expected_repr = getattr(example, "ground_truth", getattr(example, "expected", "?"))
    print(f"{show_derived_label}:", obtained)
    print(f"{show_expected_label}:", expected_repr)
    is_correct = success_measure(example, obtained)
    print("Match:", is_correct)


def run_single_example(
    example: Any,
    validator: Callable[[Any], Optional[T]],
    success_measure: Callable[[Any, Optional[Any]], bool],
    *,
    temperature: float = 0.5,
    max_steps: int = 5,
    explain: bool = True,
    show_derived_label: str = "Derived answer",
    show_expected_label: str = "Expected",
    pipeline_runner: Optional[Callable[[str], Any]] = None,
) -> Any:
    """
    Run the full pipeline on a single example and print results.

    The example must have a .problem attribute (string) that is passed to the pipeline.
    Returns the PipelineResult for further use.
    """
    problem = getattr(example, "problem", None)
    if problem is None:
        raise ValueError("Example must have a 'problem' attribute.")

    if pipeline_runner is None:
        llm_config = load_openrouter_config(temperature=temperature)
        llm = LLMClient(llm_config)
        cfg = PipelineConfig(max_steps=max_steps, explain=explain)
        result = run_symbolic_hybrid_pipeline(
            problem=problem,
            llm=llm,
            config=cfg
        )
    else:
        result = pipeline_runner(problem)

    format_single_result(
        example,
        result,
        validator,
        success_measure,
        show_derived_label=show_derived_label,
        show_expected_label=show_expected_label,
    )
    return result


def evaluate_examples(
    examples: Iterable[Any],
    validator: Callable[[Any], Optional[T]],
    success_measure: Callable[[Any, Optional[Any]], bool],
    *,
    max_steps: int = 8,
    explain: bool = False,
    show_derived_label: str = "Derived answer",
    show_expected_label: str = "Expected",
    llm: Optional[LLMClient] = None,
    pipeline_runner: Optional[Callable[[str], Any]] = None,
) -> None:
    """
    Run the pipeline over a collection of examples and print a simple accuracy summary.

    - examples: iterable of dataset-specific examples (each must have .problem)
    - validator: extracts obtained answer from each PipelineResult
    - success_measure: (example, obtained) -> True if correct
    """
    client = llm or LLMClient()
    cfg = PipelineConfig(max_steps=max_steps, explain=explain)
    runner = pipeline_runner

    total = 0
    correct = 0
    for i, ex in enumerate(examples):
        total += 1
        problem = getattr(ex, "problem", None)
        if problem is None:
            raise ValueError(f"Example {ex} has no 'problem' attribute.")
        
        try:
            if runner is None:
                result = run_symbolic_hybrid_pipeline(
                    problem=problem,
                    llm=client,
                    config=cfg
                )
            else:
                result = runner(problem)

            obtained = validator(result)
            is_correct = success_measure(ex, obtained)
            if is_correct:
                correct += 1

            format_single_result(
                ex,
                result,
                validator,
                success_measure,
                show_derived_label=show_derived_label,
                show_expected_label=show_expected_label,
            )
            print("-----")
        except Exception as e:
            print(f"Example {i} failed due to error: {e}")

    if total > 0:
        accuracy = correct / total
        print("=====")
        print(f"Total examples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.3f}")
    else:
        print("No examples to evaluate.")
