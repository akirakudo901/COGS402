"""
Evaluation suite runner.

This module provides a dataset-agnostic way to run multiple evaluation tasks
under different pipeline modes and ablation configurations (per-component models,
prompt variants, and dataset task variants).

See 'brainstorming', 'eval_suite_plan.md' for more details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

from llm_prolog.llm_client.llm_client import LLMClient
from llm_prolog.pipeline import PipelineConfig


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
class ModelMapping:
    mapping: dict[LLMRole, ModelSpec] = field(default_factory=dict)

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
    correct: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class TaskReport:
    task_id: str
    pipeline_mode: PipelineMode
    total: int
    correct: int
    accuracy: float
    outcomes: Tuple[ExampleOutcome, ...] = ()
    extra_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuiteReport:
    pipeline_mode: PipelineMode
    task_reports: Tuple[TaskReport, ...]

    @property
    def overall_accuracy(self) -> float:
        totals = sum(r.total for r in self.task_reports)
        correct = sum(r.correct for r in self.task_reports)
        return (correct / totals) if totals else 0.0


PipelineRunner = Callable[[str], Any]


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

    def validator(self, result: Any) -> Optional[Any]:
        return self.validator_fn(result)

    def success_measure(self, example: Any, obtained: Optional[Any]) -> bool:
        return self.success_measure_fn(example, obtained)

    def expected_repr(self, example: Any) -> str:
        return self.expected_fn(example)

    def problem_str(self, example: Any) -> str:
        return self.problem_fn(example)


@dataclass
class EvaluationSuite:
    tasks: Sequence[EvalTask[Any, Any]]
    pipeline_mode: PipelineMode
    model_by_role: ModelMapping
    prompt_overrides: PromptOverrides = field(default_factory=dict)
    pipeline_cfg: PipelineConfig = field(default_factory=PipelineConfig)
    keep_all_outcomes: bool = False
    keep_random_k: int = 0
    seed: int = 0

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

    def run_task(self, task: EvalTask[Any, Any], *, runner: PipelineRunner) -> TaskReport:
        import random

        rng = random.Random(self.seed)
        outcomes: List[ExampleOutcome] = []
        total = 0
        correct = 0

        for i, ex in enumerate(task.load_examples()):
            total += 1
            problem = task.problem_str(ex)
            expected = task.expected_repr(ex)
            try:
                result = runner(problem)
                obtained_val = task.validator(result, self.pipeline_mode)
                ok = task.success_measure(ex, obtained_val)
                if ok:
                    correct += 1
                obtained_str = str(obtained_val)
                outcome = ExampleOutcome(
                    idx=i,
                    example_id=str(getattr(ex, "id", i)),
                    problem=problem,
                    expected=expected,
                    obtained=obtained_str,
                    correct=ok,
                    error=None,
                )
            except Exception as e:
                outcome = ExampleOutcome(
                    idx=i,
                    example_id=str(getattr(ex, "id", i)),
                    problem=problem,
                    expected=expected,
                    obtained="",
                    correct=False,
                    error=str(e),
                )

            if self.keep_all_outcomes:
                outcomes.append(outcome)
            elif self.keep_random_k and len(outcomes) < self.keep_random_k:
                outcomes.append(outcome)
            elif self.keep_random_k and self.keep_random_k > 0:
                # Reservoir sampling.
                j = rng.randint(0, total - 1)
                if j < self.keep_random_k:
                    outcomes[j] = outcome

        accuracy = (correct / total) if total else 0.0
        return TaskReport(
            task_id=task.task_id,
            pipeline_mode=self.pipeline_mode,
            total=total,
            correct=correct,
            accuracy=accuracy,
            outcomes=tuple(outcomes),
            extra_stats={},
        )

