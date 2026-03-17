"""
Evaluation suite runner.

This module provides a dataset-agnostic way to run multiple evaluation tasks
under different pipeline modes and ablation configurations (per-component models,
prompt variants, and dataset task variants).

See 'brainstorming', 'eval_suite_plan.md' for more details.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from llm_prolog.llm_client.async_llm_client import AsyncLLMClient
from llm_prolog.llm_client.llm_client import LLMClient
from llm_prolog.llm_executor import LLMExecutor
from llm_prolog.pipeline import PipelineConfig, run_pipeline_mode_async


class PipelineMode(str, Enum):
    SYMBOLIC_HYBRID = "symbolic_hybrid"
    COT_BASELINE = "cot_baseline"
    # Reserved for later:
    COC_BASELINE = "coc_baseline"
    FULL_NL_PIPELINE = "full_nl_pipeline"

    def get_required_roles(self) -> List[LLMRole]:
        if self == PipelineMode.SYMBOLIC_HYBRID:
            return [LLMRole.NL_TO_SYMBOL, LLMRole.SELECTOR, LLMRole.SYMBOL_TO_NL]
        elif self == PipelineMode.COT_BASELINE:
            return [LLMRole.COT_SOLVER]
        elif self == PipelineMode.COC_BASELINE:
            print("get_required_roles for COC_BASELINE mode is yet to be implemented.")
            return []
        elif self == PipelineMode.FULL_NL_PIPELINE:
            print("get_required_roles for FULL_NL_PIPELINE mode is yet to be implemented.")
            return []


class LLMRole(str, Enum):
    NL_TO_SYMBOL = "nl_to_symbol"
    SELECTOR = "selector"
    SYMBOL_TO_NL = "symbol_to_nl"
    COT_SOLVER = "cot_solver"


@dataclass(frozen=True)
class ModelSpec:
    model: str
    temperature: float | None = None
    max_tokens: int | None = None


PromptOverrides = Mapping[LLMRole, str]

@dataclass
class ModelMapping(Mapping[LLMRole, ModelSpec]):
    mapping: dict[LLMRole, ModelSpec] = field(default_factory=dict)

    def __getitem__(self, key: LLMRole) -> ModelSpec:
        return self.mapping[key]

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __contains__(self, key):
        return key in self.mapping
    
    def items(self):
        return self.mapping.items()

    @classmethod
    def set_spec_to_all_roles(
        cls, 
        spec : ModelSpec, 
        mode : PipelineMode = None
    ) -> "ModelMapping":
        if mode is None:
            roles_to_fill = [role for role in LLMRole]
        else:
            roles_to_fill = mode.get_required_roles()
        return cls(mapping=dict((role, spec) for role in roles_to_fill))

TObtained = TypeVar("TObtained")
TExample = TypeVar("TExample")


class EvalTask(Protocol[TExample, TObtained]):
    task_id: str

    def load_examples(self) -> Iterable[TExample]:
        """Return examples (already in the concrete form needed by the runner)."""

    def validator(self, result: Any, pipeline_mode: PipelineMode) -> Optional[TObtained]:
        """
        Extract a canonical 'obtained' value from a pipeline result.

        The validator can specialize its behavior based on the active PipelineMode.
        """

    def success_measure(self, example: TExample, obtained: Optional[TObtained]) -> bool:
        """Return True iff the obtained value is correct for the example."""

    def expected_repr(self, example: TExample) -> str:
        """String repr of the ground truth, used in reports."""

    def problem_str(self, example: TExample) -> str:
        """The problem string to pass to the pipeline/baseline."""


@dataclass(frozen=True)
class ExampleOutcome:
    idx: int
    example_id: str
    problem: str
    expected: str
    obtained: str
    result: Any
    correct: bool
    error: Optional[str] = None

    def __str__(self) -> str:
        lines = []
        lines.append(f"Index {self.idx}, Example ID {self.example_id}.")
        lines.append(f"  Problem: {self.problem}")
        lines.append(f"  Expected '{self.expected}', got '{self.obtained}': " + 
                      ('' if self.correct else 'in') + "correct.")
        lines.append(f"  Pipeline Result (of Any type): {self.result}")
        if not self.correct:
            lines.append(f"  Reason for error: {self.error}")
        return "\n".join(lines)


@dataclass(frozen=True)
class TaskReport:
    task_id: str
    pipeline_mode: PipelineMode
    total: int
    correct: int
    accuracy: float
    outcomes: Tuple[ExampleOutcome, ...] = ()
    extra_stats: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = []
        lines.append(f"Task {self.task_id} (mode={self.pipeline_mode.name}):")
        lines.append(f"  Got {self.correct} / {self.total} correct; {self.accuracy} accuracy.")
        lines.append(f"  Outcomes:")
        for eo in self.outcomes:
            lines.append("-"*10)
            lines.append(str(eo))
            lines.append("-"*10)
        if self.extra_stats:
            lines.append("="*10)
            lines.append(f" Extra Stats: {self.extra_stats}")
            lines.append("="*10)
        return "\n".join(lines)


@dataclass(frozen=True)
class SuiteReport:
    pipeline_mode: PipelineMode
    task_reports: Tuple[TaskReport, ...]

    @property
    def overall_accuracy(self) -> float:
        totals = sum(r.total for r in self.task_reports)
        correct = sum(r.correct for r in self.task_reports)
        return (correct / totals) if totals else 0.0
    
    def __str__(self) -> str:
        lines = []
        for tr in self.task_reports:
            lines.append("#"*10)
            lines.append(str(tr))
            lines.append("#"*10)
        lines.append(f"Overall Accuracy: {self.overall_accuracy}")
        return "\n".join(lines)



PipelineRunner = Callable[[str], Any]


def _get_example_fields(task: EvalTask[Any, Any], ex: Any, idx: int) -> Tuple[str, str, str]:
    problem = task.problem_str(ex)
    expected = task.expected_repr(ex)
    example_id = str(getattr(ex, "id", idx))
    return problem, expected, example_id


def _make_outcome_from_result(
    *,
    task: EvalTask[Any, Any],
    pipeline_mode: PipelineMode,
    ex: Any,
    idx: int,
    example_id: str,
    problem: str,
    expected: str,
    result: Any,
) -> Tuple[ExampleOutcome, bool]:
    obtained_val = task.validator(result, pipeline_mode)
    ok = task.success_measure(ex, obtained_val)
    outcome = ExampleOutcome(
        idx=idx,
        example_id=example_id,
        problem=problem,
        expected=expected,
        obtained=str(obtained_val),
        result=result,
        correct=ok,
        error=None,
    )
    return outcome, ok


def _make_outcome_from_exception(
    *,
    idx: int,
    example_id: str,
    problem: str,
    expected: str,
    exc: Exception,
) -> Tuple[ExampleOutcome, bool]:
    outcome = ExampleOutcome(
        idx=idx,
        example_id=example_id,
        problem=problem,
        expected=expected,
        obtained="",
        result=None,
        correct=False,
        error=str(exc),
    )
    return outcome, False


def _placeholder_outcome(idx: int) -> ExampleOutcome:
    return ExampleOutcome(
        idx=idx,
        example_id=str(idx),
        problem="",
        expected="",
        obtained="",
        result=None,
        correct=False,
        error="Missing async outcome",
    )


def _collect_outcomes_in_order(
    *,
    outcomes_in_order: Sequence[ExampleOutcome],
    rng,
    keep_all_outcomes: bool,
    keep_random_k: int,
) -> List[ExampleOutcome]:
    outcomes_collect: List[ExampleOutcome] = []
    seen = 0
    for outcome in outcomes_in_order:
        seen += 1
        if keep_all_outcomes:
            outcomes_collect.append(outcome)
        elif keep_random_k and len(outcomes_collect) < keep_random_k:
            outcomes_collect.append(outcome)
        elif keep_random_k and keep_random_k > 0:
            # Reservoir sampling.
            j = rng.randint(0, seen - 1)
            if j < keep_random_k:
                outcomes_collect[j] = outcome
    return outcomes_collect


def default_problem_str(example: Any) -> str:
    problem = getattr(example, "problem", None)
    if not isinstance(problem, str):
        raise ValueError("Example must have a string 'problem' attribute.")
    return problem


def default_expected_repr(example: Any) -> str:
    gt = getattr(example, "ground_truth", getattr(example, "expected", "?"))
    return str(gt)


@dataclass(frozen=True)
class SimpleEvalTask:
    task_id: str
    examples: Sequence[Any]
    validator_fn: Callable[[Any, PipelineMode], Optional[Any]]
    success_measure_fn: Callable[[Any, Optional[Any]], bool]
    problem_fn: Callable[[Any], str] = default_problem_str
    expected_fn: Callable[[Any], str] = default_expected_repr

    def load_examples(self) -> Iterable[Any]:
        return self.examples

    def validator(self, result: Any, mode: PipelineMode) -> Optional[Any]:
        return self.validator_fn(result, mode)

    def success_measure(self, example: Any, obtained: Optional[Any]) -> bool:
        return self.success_measure_fn(example, obtained)

    def expected_repr(self, example: Any) -> str:
        return self.expected_fn(example)

    def problem_str(self, example: Any) -> str:
        return self.problem_fn(example)


@dataclass
class EvaluationSuite:
    name: str
    tasks: Sequence[EvalTask[Any, Any]]
    pipeline_mode: PipelineMode
    model_by_role: ModelMapping
    prompt_overrides: PromptOverrides = field(default_factory=dict)
    pipeline_cfg: PipelineConfig = field(default_factory=PipelineConfig)
    keep_all_outcomes: bool = False
    keep_random_k: int = 0
    seed: int = 0

    def __post_init__(self):
        # Ensure model_by_role contains a mapping for all roles (in LLMRole)
        missing_roles = [role for role in self.pipeline_mode.get_required_roles() 
                         if role not in self.model_by_role.mapping]
        if missing_roles:
            raise ValueError(f"model_by_role is missing mappings for roles: {missing_roles}")
        # also indicate that keep_all_outcomes takes precedence over keep_random_k
        if self.keep_all_outcomes and self.keep_random_k:
            print("With both keep_all_outcomes and keep_random_k set, the former takes precedence.")
    
    def get_description(self) -> str:
        def _spec_to_model_name(spec : ModelSpec) -> str:
            return spec.model.split("/")[-1].strip()
        
        out = ""
        # pipeline mode
        out += f"{self.pipeline_mode.name} mode"
        # roles
        out += " with "
        required_roles = self.pipeline_mode.get_required_roles()
        role_str, name_to_roles_map = [], {}
        
        for role, spec in self.model_by_role.items():
            if role in required_roles:
                model_name = _spec_to_model_name(spec)
                name_to_roles_map[model_name] = name_to_roles_map.get(model_name, []) + [role]
        
        for name, roles in name_to_roles_map.items():
            agg_roles_str = ' & '.join([r.name for r in roles])
            role_str.append(f"{agg_roles_str} : {name}")
        
        out += ";".join(role_str)
        # tasks
        out += " on (" + ', '.join([t.task_id for t in self.tasks]) + ")"
        return out

    def _make_llm(self) -> LLMClient:
        # The per-role ModelSpec is passed per request via LLMClient overrides.
        return LLMClient()

    def _runner(self, *, llm: LLMClient) -> PipelineRunner:
        from llm_prolog.pipeline import run_pipeline_mode

        def _run(problem: str) -> Any:
            return run_pipeline_mode(
                problem=problem,
                mode=self.pipeline_mode,
                pipeline_cfg=self.pipeline_cfg,
                llm=llm,
                model_by_role=self.model_by_role,
                prompt_overrides=self.prompt_overrides,
            )

        return _run

    def run(self) -> SuiteReport:
        llm = self._make_llm()
        runner = self._runner(llm=llm)
        reports: List[TaskReport] = []
        for task in self.tasks:
            reports.append(self.run_task(task, runner=runner))
        return SuiteReport(pipeline_mode=self.pipeline_mode, task_reports=tuple(reports))

    async def run_async(self, max_in_flight: int = 8, print_progress : bool = False) -> SuiteReport:
        """
        Run all tasks with concurrent pipelines; LLM calls are bounded by max_in_flight.
        Uses one shared AsyncLLMClient and LLMExecutor per suite run.
        """
        async with AsyncLLMClient() as client:
            executor = LLMExecutor(client, max_in_flight=max_in_flight)
            reports: List[TaskReport] = []
            for task in self.tasks:
                report = await self.run_task_async(
                    task, llm_exec=executor, max_in_flight=max_in_flight, print_progress=print_progress
                )
                reports.append(report)
        return SuiteReport(pipeline_mode=self.pipeline_mode, task_reports=tuple(reports))

    async def run_task_async(
        self,
        task: EvalTask[Any, Any],
        *,
        llm_exec: LLMExecutor,
        max_in_flight: int = 8,
        print_progress: bool = False
    ) -> TaskReport:
        """
        Run one task with each example in its own async pipeline; results aggregated by index.
        """
        import random

        rng = random.Random(self.seed)
        examples = list(task.load_examples())
        total = len(examples)
        ordered_outcomes: List[Optional[ExampleOutcome]] = [None] * total
        ordered_correct: List[bool] = [False] * total

        async def run_one(i: int, ex: Any) -> Tuple[int, ExampleOutcome, bool]:
            problem, expected, example_id = _get_example_fields(task, ex, i)
            try:
                if print_progress:
                    print(f"Starting task {task.task_id}, example {example_id} at: {datetime.now().strftime('%H:%M:%S')}.")
                result = await run_pipeline_mode_async(
                    problem=problem,
                    mode=self.pipeline_mode,
                    pipeline_cfg=self.pipeline_cfg,
                    llm_exec=llm_exec,
                    model_by_role=self.model_by_role,
                    prompt_overrides=self.prompt_overrides
                )
                outcome, ok = _make_outcome_from_result(
                    task=task,
                    pipeline_mode=self.pipeline_mode,
                    ex=ex,
                    idx=i,
                    example_id=example_id,
                    problem=problem,
                    expected=expected,
                    result=result,
                )
                if print_progress:
                    print(f"Completed task {task.task_id}, example {example_id} without exception at: {datetime.now().strftime('%H:%M:%S')}.")
            except Exception as e:
                outcome, ok = _make_outcome_from_exception(
                    idx=i,
                    example_id=example_id,
                    problem=problem,
                    expected=expected,
                    exc=e,
                )
                if print_progress:
                    print(f"Failed task {task.task_id}, example {example_id} with exception at: {datetime.now().strftime('%H:%M:%S')}.")
            return i, outcome, ok

        tasks = [run_one(i, ex) for i, ex in enumerate(examples)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for i, outcome, ok in results:
            ordered_outcomes[i] = outcome
            ordered_correct[i] = ok

        correct = sum(1 for ok in ordered_correct if ok)
        outcomes_in_order: List[ExampleOutcome] = []
        for i in range(total):
            outcomes_in_order.append(ordered_outcomes[i] or _placeholder_outcome(i))

        outcomes_collect = _collect_outcomes_in_order(
            outcomes_in_order=outcomes_in_order,
            rng=rng,
            keep_all_outcomes=self.keep_all_outcomes,
            keep_random_k=self.keep_random_k,
        )

        accuracy = (correct / total) if total else 0.0
        return TaskReport(
            task_id=task.task_id,
            pipeline_mode=self.pipeline_mode,
            total=total,
            correct=correct,
            accuracy=accuracy,
            outcomes=tuple(outcomes_collect),
            extra_stats={"max_in_flight": max_in_flight},
        )

    def run_task(
        self, task: EvalTask[Any, Any], *, runner: PipelineRunner
    ) -> TaskReport:
        import random

        rng = random.Random(self.seed)
        total = 0
        correct = 0
        outcomes_in_order: List[ExampleOutcome] = []

        for i, ex in enumerate(task.load_examples()):
            total += 1
            problem, expected, example_id = _get_example_fields(task, ex, i)
            try:
                result = runner(problem)
                outcome, ok = _make_outcome_from_result(
                    task=task,
                    pipeline_mode=self.pipeline_mode,
                    idx=i,
                    example_id=example_id,
                    problem=problem,
                    expected=expected,
                    ex=ex,
                    result=result,
                )
                if ok:
                    correct += 1
            except Exception as e:
                outcome, _ = _make_outcome_from_exception(
                    idx=i,
                    example_id=example_id,
                    problem=problem,
                    expected=expected,
                    exc=e,
                )

            outcomes_in_order.append(outcome)

        accuracy = (correct / total) if total else 0.0
        outcomes_collect = _collect_outcomes_in_order(
            outcomes_in_order=outcomes_in_order,
            rng=rng,
            keep_all_outcomes=self.keep_all_outcomes,
            keep_random_k=self.keep_random_k,
        )
        return TaskReport(
            task_id=task.task_id,
            pipeline_mode=self.pipeline_mode,
            total=total,
            correct=correct,
            accuracy=accuracy,
            outcomes=tuple(outcomes_collect),
            extra_stats={},
        )

