"""
The script where the actual evaluation occurs.
"""

from typing import Optional, Sequence

import asyncio

from llm_prolog.pipeline import PipelineConfig

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

def spec_to_name(spec : ModelSpec) -> str:
    return spec.model.split("/")[-1]

def get_gsm8K_eval_task(
    size : Optional[int] = None, 
    seed : int = 42, 
    from_train_split : bool = True, 
    ids : Optional[Sequence[int]] = None
) -> SimpleEvalTask:
    split_tag = "train" if from_train_split else "test"
    if not size and not ids:
        raise Exception("Either 'size' or 'ids' must be given to 'get_gsm8K_eval_task'.")
    elif size:
        if size and ids:
            print("size and ids given to 'get_gsm8K_eval_task', size takes priority.")
        return SimpleEvalTask(
            task_id=f"gsm8k:{size}exs:{split_tag}:seed={seed}", 
            examples=load_gsm8k_examples(size=size, seed=seed, from_train_split=from_train_split), 
            validator_fn=gsm8k_main_validator, 
            success_measure_fn=gsm8k_success_measure,
            problem_fn=default_problem_str,
            expected_fn=default_expected_repr
        )
    elif ids:
        return SimpleEvalTask(
            task_id=f"gsm8k:ids={ids}:{split_tag}", 
            examples=load_gsm8k_examples(size=0, from_train_split=from_train_split, ids=ids), 
            validator_fn=gsm8k_main_validator, 
            success_measure_fn=gsm8k_success_measure,
            problem_fn=default_problem_str,
            expected_fn=default_expected_repr
        )


# eval suite 1: trial
trial_spec = ModelSpec(model="openai/gpt-4.1-mini", temperature=0.5, max_tokens=None)

mode = PipelineMode.SYMBOLIC_HYBRID
pipeline_cfg = PipelineConfig(max_steps=10, explain=True)

TASK_SIZE = 1
suite1 = EvaluationSuite(
    name=f"{spec_to_name(trial_spec)} Symbolic Hybrid On GSM8K ({TASK_SIZE}EX)",
    tasks=[get_gsm8K_eval_task(size=TASK_SIZE),],
    pipeline_mode=mode,
    model_by_role=ModelMapping.set_spec_to_all_roles(trial_spec, mode),
    prompt_overrides=None,
    pipeline_cfg=pipeline_cfg,
    keep_all_outcomes=True,
    keep_random_k=0,
    seed=0
)
multi_suite1 = [suite1, ]

# eval suite 2: qwen different sizes on GSM8K of size 20
specs = [
    # ModelSpec(model="meta-llama/llama-3.2-3b-instruct", temperature=0.5, max_tokens=None),
    # ModelSpec(model="meta-llama/llama-3.1-8b-instruct", temperature=0.5, max_tokens=None),
    # ModelSpec(model="meta-llama/llama-3.3-70b-instruct", temperature=0.5, max_tokens=None),
    # ModelSpec(model="meta-llama/llama-3.1-405b-instruct", temperature=0.5, max_tokens=None), #EXPENSIVE! $5 WON'T LAST FOR LONG
    
    # GPTs
    # ModelSpec(model="openai/gpt-5-mini", temperature=0.5, max_tokens=None),
    ModelSpec(model="openai/gpt-4.1-mini", temperature=0.5, max_tokens=None),

    # Qwens
    # ModelSpec(model="qwen/qwen3-235b-a22b-2507", temperature=0.5, max_tokens=None),
    # ModelSpec(model="qwen/qwen3-30b-a3b-instruct-2507", temperature=0.5, max_tokens=None),
    # ModelSpec(model="qwen/qwen3-coder-30b-a3b-instruct", temperature=0.5, max_tokens=None),
]

mode = PipelineMode.SYMBOLIC_HYBRID
pipeline_cfg = PipelineConfig(max_steps=10, explain=True)

TASK_SIZE = 50
multi_suite2 = [
    EvaluationSuite(
        name=f"{spec_to_name(specific_spec)} Symbolic Hybrid On GSM8K ({TASK_SIZE}EX)",
        tasks=[get_gsm8K_eval_task(size=TASK_SIZE),],
        pipeline_mode=mode,
        model_by_role=ModelMapping.set_spec_to_all_roles(specific_spec, mode),
        prompt_overrides=None,
        pipeline_cfg=pipeline_cfg,
        keep_all_outcomes=True,
        keep_random_k=0,
        seed=0
    )
    for specific_spec in specs
]

# eval suite 3: GPT4.1mini on different failure modes to see if they're fixed
spec = ModelSpec(model="openai/gpt-4.1-mini", temperature=0.5, max_tokens=None)

mode = PipelineMode.SYMBOLIC_HYBRID
pipeline_cfg = PipelineConfig(
    max_steps=20, 
    explain=True, 
    use_termination_checks=False, 
    use_final_termination_check=True
    )

fail_derive_ids = [3, 9, 10, 14, 17, 23, 28, 33, 39, 40, 48, 49]
combined_already_ids = [3, 9, 10, 14, 17, 23, 24, 28, 33, 35, 39, 40, 44, 48, 49]
only_one_premise_ids = [5, 13, 23, 24, 32, 35, 39, 44]

combined_already_ids = [id for id in combined_already_ids if id not in fail_derive_ids]
only_one_premise_ids = [id for id in only_one_premise_ids  
                        if (id not in fail_derive_ids) and (id not in combined_already_ids)]

fail_derive_name = f"{spec_to_name(spec)} Symbolic Hybrid On GSM8K For 'Failing to derive new premise'"
combined_already_name = f"{spec_to_name(spec)} Symbolic Hybrid On GSM8K For 'Combining previously combined premises'"
only_one_premise_name = f"{spec_to_name(spec)} Symbolic Hybrid On GSM8K For 'Selecting only one premise'"

def return_suite(name, ids):
    return EvaluationSuite(
        name=name,
        tasks=[get_gsm8K_eval_task(ids=ids),],
        pipeline_mode=mode,
        model_by_role=ModelMapping.set_spec_to_all_roles(spec, mode),
        prompt_overrides=None,
        pipeline_cfg=pipeline_cfg,
        keep_all_outcomes=True,
        keep_random_k=0,
        seed=0
    )

multi_suite3 = [
    return_suite(fail_derive_name, fail_derive_ids),
    return_suite(combined_already_name, combined_already_ids),
    return_suite(only_one_premise_name, only_one_premise_ids)
]

#################
# RUN OF CHOICE #
#################
suites_of_choice = multi_suite3

for s in suites_of_choice:
    print("~"*50)
    print(s.get_description())
    report = asyncio.run(s.run_async(max_in_flight=15, print_progress=True, show_openrouter_balance=True))
    print(report)