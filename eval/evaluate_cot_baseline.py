"""
CoT baseline on GSM8K with artifact persistence (step 3d validation).

Run from repository root:
  export OPENROUTER_API_KEY="..."
  python -m eval.run_cot_gsm8k --size 5 --seed 42

Optional:
  python -m eval.artifact.validate_artifacts artifacts --latest
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from eval.artifact import new_run_id, persist_evaluation_run
from eval.artifact.validate_artifacts import validate_run_dir
from eval.evaluate_symbolic_hybrid import get_gsm8K_eval_task
from eval.eval_suite import (
    EvaluationSuite,
    ModelMapping,
    ModelSpec,
    PipelineMode,
)
from llm_prolog.pipeline import PipelineConfig


def run_cot_baseline_evaluation(
    suite: EvaluationSuite,
    artifacts_root: Path,
    variant_id: str,
    dataset_meta: dict,
    ablation_overrides: dict = None,
    write_failures: bool = True,
    max_in_flight: int = 4,
) -> Path:
    """
    Runs the evaluation suite, persists artifacts, validates artifacts, and
    returns the run directory path.
    """
    print(suite.get_description())
    report = asyncio.run(
        suite.run_async(
            max_in_flight=max_in_flight,
            print_progress=True,
            show_openrouter_balance=True,
        )
    )
    print(report)

    run_id = new_run_id()
    ablation = {
        "variant_id": variant_id,
        "component_overrides": ablation_overrides or {},
    }
    run_dir = persist_evaluation_run(
        artifacts_root=artifacts_root.resolve(),
        run_id=run_id,
        suite=suite,
        suite_report=report,
        dataset=dataset_meta,
        ablation=ablation,
        write_failures=write_failures,
    )
    print(f"Artifacts written to: {run_dir}")
    ok, errs = validate_run_dir(run_dir)
    if not ok:
        for err in errs:
            print(f"[artifact validation] {err}")
        raise RuntimeError(f"Artifact validation failed for run directory: {run_dir}")
    print("Artifact validation passed.")
    return run_dir

def evaluate_all_models_on_subset():
    import eval
    
    FROM_TRAIN_SPLIT = True
    SIZE = 30
    DATASET_SEED = 42
    KEEP_RESULT_SEED = 0
    
    TEMPERATURE = 0.5
    MAX_IN_FLIGHT = 15
    
    ARTIFACTS_DIR = Path("artifacts")
    VARIANT_ID = "cot_baseline_subset"
    NO_FAILURES = False


    split = "train" if FROM_TRAIN_SPLIT else "test"

    all_model_names = eval.get_all_model_names_filtered(
        include_llama_family=True,
        include_gpt_family=True,
        include_qwen_family=True,
        include_llama405b=False #expensive so not right now
        )
    all_model_names.remove(eval.GPT_4_1_MINI)
    # all_model_names = [eval.GPT_4_1_MINI,]
    
    specs = [ ModelSpec(model=name, temperature=TEMPERATURE, max_tokens=None)
              for name in all_model_names]
    
    pipeline_cfg = PipelineConfig(
        max_steps=10, 
        explain=True,
        ) #for run_meta only
    
    def get_suite(model_spec):
        return EvaluationSuite(
            name=f"CoT baseline GSM8K n={SIZE} ({split}, seed={DATASET_SEED})",
            tasks=[get_gsm8K_eval_task(size=SIZE, seed=DATASET_SEED, from_train_split=FROM_TRAIN_SPLIT)],
            pipeline_mode=PipelineMode.COT_BASELINE,
            model_by_role=ModelMapping.set_spec_to_all_roles(model_spec, PipelineMode.COT_BASELINE),
            prompt_overrides={},
            pipeline_cfg=pipeline_cfg,
            keep_all_outcomes=True,
            keep_random_k=0,
            seed=KEEP_RESULT_SEED,
        )
    
    suites = [get_suite(spec) for spec in specs]

    dataset_meta = {
        "name": "gsm8k",
        "split": split,
        "subset_spec": {"size": SIZE, "seed": DATASET_SEED, "random_sample": True},
    }

    for s in suites:
        run_cot_baseline_evaluation(
            suite=s,
            artifacts_root=ARTIFACTS_DIR,
            variant_id=VARIANT_ID,
            dataset_meta=dataset_meta,
            ablation_overrides={},
            write_failures=not NO_FAILURES,
            max_in_flight=MAX_IN_FLIGHT,
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run CoT baseline on a GSM8K subset and write artifacts.")
    parser.add_argument("--size", type=int, default=5, help="Number of examples (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for subset sampling")
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="GSM8K split to sample from (default: train)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root directory for run_<id> folders (default: ./artifacts)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4.1-mini",
        help="OpenRouter model id for cot_solver",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--suite-seed",
        type=int,
        default=None,
        help="EvaluationSuite seed (e.g. reservoir sampling). Default: same as --seed.",
    )
    parser.add_argument("--max-in-flight", type=int, default=4)
    parser.add_argument(
        "--variant-id",
        type=str,
        default="cot_baseline_subset",
        help="ablation.variant_id stored in run_meta.json",
    )
    parser.add_argument("--no-failures", action="store_true", help="Skip writing failures.jsonl")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set. Set it before running (real API, not stub).")
        raise SystemExit(2)

    from_train = args.split == "train"
    suite_seed = args.suite_seed if args.suite_seed is not None else args.seed
    spec = ModelSpec(model=args.model, temperature=args.temperature, max_tokens=None)
    pipeline_cfg = PipelineConfig(max_steps=10, explain=True)  # Used for run_meta only
    suite = EvaluationSuite(
        name=f"CoT baseline GSM8K n={args.size} ({args.split}, seed={args.seed})",
        tasks=[get_gsm8K_eval_task(size=args.size, seed=args.seed, from_train_split=from_train)],
        pipeline_mode=PipelineMode.COT_BASELINE,
        model_by_role=ModelMapping.set_spec_to_all_roles(spec, PipelineMode.COT_BASELINE),
        prompt_overrides={},
        pipeline_cfg=pipeline_cfg,
        keep_all_outcomes=True,
        keep_random_k=0,
        seed=suite_seed,
    )

    dataset_meta = {
        "name": "gsm8k",
        "split": args.split,
        "subset_spec": {"size": args.size, "seed": args.seed, "random_sample": True},
    }

    run_cot_baseline_evaluation(
        suite=suite,
        artifacts_root=args.artifacts_dir,
        variant_id=args.variant_id,
        dataset_meta=dataset_meta,
        ablation_overrides={},
        write_failures=not args.no_failures,
        max_in_flight=args.max_in_flight,
    )


if __name__ == "__main__":
    # main()
    evaluate_all_models_on_subset()
