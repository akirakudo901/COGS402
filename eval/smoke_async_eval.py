"""
Smoke test for async evaluation pipeline.

Run with:
  python -m eval.smoke_async_eval

Requires: OPENROUTER_API_KEY set, httpx installed (pip install httpx).
Uses 2 GSM8K examples and max_steps=3 to keep the run short.
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv

from eval.eval_suite import (
    EvaluationSuite,
    ModelMapping,
    ModelSpec,
    PipelineMode,
    SimpleEvalTask,
    default_expected_repr,
    default_problem_str,
)
from eval.per_dataset.eval_gsm8k import (
    gsm8k_main_validator,
    gsm8k_success_measure,
    load_gsm8k_examples,
)
from llm_prolog.pipeline import PipelineConfig


def main() -> None:
    # load API key & optionally the model via dotenv
    load_dotenv()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("SKIP: OPENROUTER_API_KEY not set. Set it to run the smoke.")
        return

    def get_gsm8k_task(size: int, seed: int = 42) -> SimpleEvalTask:
        return SimpleEvalTask(
            task_id=f"gsm8k:{size}exs",
            examples=load_gsm8k_examples(size=size, seed=seed, from_train_split=True),
            validator_fn=gsm8k_main_validator,
            success_measure_fn=gsm8k_success_measure,
            problem_fn=default_problem_str,
            expected_fn=default_expected_repr,
        )

    spec = ModelSpec(model="openai/gpt-4.1-mini", temperature=0.5, max_tokens=None)
    suite = EvaluationSuite(
        name="Smoke async",
        tasks=[get_gsm8k_task(5)],
        pipeline_mode=PipelineMode.SYMBOLIC_HYBRID,
        model_by_role=ModelMapping.set_spec_to_all_roles(spec, PipelineMode.SYMBOLIC_HYBRID),
        prompt_overrides={},
        pipeline_cfg=PipelineConfig(max_steps=10, explain=True),
        keep_all_outcomes=True,
        keep_random_k=0,
        seed=0,
    )

    print("Run 1: max_in_flight=1")
    t0 = time.perf_counter()
    report1 = asyncio.run(suite.run_async(max_in_flight=1, print_progress=True))
    t1 = time.perf_counter()
    print(f"  Accuracy: {report1.overall_accuracy}, Time: {t1 - t0:.1f}s")
    print(f"Report 1: {report1}")

    print("Run 2: max_in_flight=5")
    t0 = time.perf_counter()
    report2 = asyncio.run(suite.run_async(max_in_flight=5, print_progress=True))
    t1 = time.perf_counter()
    print(f"  Accuracy: {report2.overall_accuracy}, Time: {t1 - t0:.1f}s")
    print(f"Report 2: {report2}")

    print("Smoke OK")


if __name__ == "__main__":
    main()
