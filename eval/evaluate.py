"""
The script where the actual evaluation occurs.
"""

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

def get_gsm8K_eval_task(size : int, seed : int = 42, from_train_split : bool = True) -> SimpleEvalTask:
    return SimpleEvalTask(
        task_id=f"gsm8k:{size}exs", 
        examples=load_gsm8k_examples(size=size, seed=seed, from_train_split=from_train_split), 
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

#################
# RUN OF CHOICE #
#################
suites_of_choice = multi_suite2

for s in suites_of_choice:
    print("~"*50)
    print(s.get_description())
    report = s.run()
    print(report)