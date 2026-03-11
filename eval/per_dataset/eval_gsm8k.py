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

import pandas as pd

from llm_prolog.symbolic.types import Fact, PipelineResult

from eval.eval_common import evaluate_examples, run_single_example
from eval.eval_suite import PipelineMode, SimpleEvalTask


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


def gsm8k_nlsymbol_validator(result: PipelineResult) -> Optional[int]:
    """Extract the derived numeric answer from the pipeline result (GSM8K format)."""
    answer = result.extract_answer_constant()
    if answer is None:
        return None
    try:
        return int(answer)
    except ValueError:
        return None


def _gsm8k_validator_symbolic_hybrid(result: object) -> Optional[int]:
    """
    Helper validator for PipelineMode.SYMBOLIC_HYBRID.
    Expects a PipelineResult and extracts the numeric answer via constants.
    """
    if not hasattr(result, "extract_answer_constant"):
        return None
    try:
        return gsm8k_nlsymbol_validator(result)  # type: ignore[arg-type]
    except Exception:
        return None


def _gsm8k_validator_text_answer(result: object) -> Optional[int]:
    """
    Helper for text-only pipelines (e.g. COT_BASELINE, COC_BASELINE, FULL_NL_PIPELINE).
    Attempts to parse the final integer from the free-form answer text.
    """
    answer_text = getattr(result, "answer_text", None)
    if not isinstance(answer_text, str):
        return None

    import re

    nums = re.findall(r"[-+]?\d+", answer_text)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


def gsm8k_main_validator(result: object, mode: PipelineMode) -> Optional[int]:
    """
    Main GSM8K validator that dispatches on PipelineMode.

    - SYMBOLIC_HYBRID: use the symbolic PipelineResult helper.
    - COT_BASELINE / COC_BASELINE / FULL_NL_PIPELINE: use text-based parsing.
    """
    if mode is PipelineMode.SYMBOLIC_HYBRID:
        return _gsm8k_validator_symbolic_hybrid(result)
    if mode in {
        PipelineMode.COT_BASELINE,
        PipelineMode.COC_BASELINE,
        PipelineMode.FULL_NL_PIPELINE,
    }:
        return _gsm8k_validator_text_answer(result)
    # Fallback: be conservative if a new mode is added.
    return None


def gsm8k_success_measure(example: GSM8KExample, obtained: Optional[int]) -> bool:
    """Success = exact match of integer answer."""
    return obtained is not None and obtained == example.ground_truth


def run_single_example_gsm8k(example: GSM8KExample = EXAMPLE_1) -> None:
    """Run the pipeline on a single GSM8K example and print results."""
    run_single_example(
        example,
        gsm8k_nlsymbol_validator,
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
        gsm8k_nlsymbol_validator,
        gsm8k_success_measure,
        max_steps=max_steps,
        show_derived_label="Derived numeric answer",
        show_expected_label="Ground truth",
    )

# ---------------------------------------------------------------------------
# Loading the gsm8k files from Hugging Face
# Also, suite integration: task registry
# ---------------------------------------------------------------------------

"""
The GSM8K dataset holds 7473 question answer pairs within 'train'! Each entry under the 'question' column 
is a specific question, which natural language answer can be found in the corresponding row under 'answer'.

The answer interleaves NL, equations and the final answer; e.g.

    Natalia sold 48/2 = <<48/2=24>>24 clips in May. \
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. \
    #### 72

>>> df.info()

<class 'pandas.DataFrame'>
RangeIndex: 7473 entries, 0 to 7472
Data columns (total 2 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   question  7473 non-null   str  
 1   answer    7473 non-null   str  
dtypes: str(2)
memory usage: 3.8 MB
"""

SIZE_OPTIONS = ("20", "all")


splits = {
    'train': 'main/train-00000-of-00001.parquet',
    'test': 'main/test-00000-of-00001.parquet'
}

def _parse_groundtruth_from_answer(
    full_answer : str,
) -> int:
    """
    Parse the ground-truth from an answer interleaving NL, equations and the final answer; e.g.

    Natalia sold 48/2 = <<48/2=24>>24 clips in May. \
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. \
    #### 72
    """
    answer_parts = full_answer.split("####")
    if len(answer_parts) != 2:
        raise Exception("Simple parse failed to get ground-truth answer from GSM8K example. "
                        f"Failing answer string: {full_answer}")
    try:
        return int(answer_parts[1])
    except:
        raise Exception("Failed parsing ground-truth answer for GSM8K example; "
                        f"'{answer_parts[1]}' can't be parsed into an integer.")


def load_gsm8k_examples(
    size: int | str,
    seed: int = 42,
    from_train_split: bool = True
) -> list[GSM8KExample]:
    """
    Load specified number of examples from the specified split at random using the given seed.
    'size' can be 'all', in which case all examples are returned.
    """
    import random

    split = "train" if from_train_split else "test"
    path = "hf://datasets/openai/gsm8k/" + splits[split]

    df = pd.read_parquet(path)

    if size == "all":
        indices = range(len(df))
    elif isinstance(size, str) and size.isdigit() or isinstance(size, int):
        size_int = int(size)
        if size_int > len(df):
            raise ValueError(f"Requested {size_int} examples, but only {len(df)} available in split '{split}'.")
        rng = random.Random(seed)
        indices = rng.sample(range(len(df)), size_int)
    else:
        raise ValueError(f"Unknown GSM8K size option: {size!r}")

    examples = []
    for i in indices:
        row = df.iloc[i]
        int_gt = _parse_groundtruth_from_answer(row["answer"])
        ex = GSM8KExample(
            question=row["question"],
            answer=int_gt,
        )
        examples.append(ex)
    return examples


def get_tasks(seed : int = 42) -> list[SimpleEvalTask]:
    tasks: list[SimpleEvalTask] = []
    for size in SIZE_OPTIONS:
        tasks.append(
            SimpleEvalTask(
                task_id=f"gsm8k:{size}",
                examples=load_gsm8k_examples(size, seed=seed),
                validator_fn=gsm8k_main_validator,
                success_measure_fn=gsm8k_success_measure,
            )
        )
    return tasks


TASKS = {t.task_id: t for t in get_tasks()}


if __name__ == "__main__":
    evaluate_gsm8k(
        examples=[
            EXAMPLE_1,
            EXAMPLE_5,
            EXAMPLE_6,
        ],
        max_steps=20,
    )