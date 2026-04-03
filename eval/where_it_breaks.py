"""
Where-it-breaks experiment scaffolding (step 3e).

This module encodes variant generation for the staircase protocol and documents
what downstream analysis should read from persisted artifacts.

Phase 1 has three modes:
- ``run_where_it_breaks_phase_1`` — full symbolic-hybrid pipeline per ladder model (legacy).
- ``run_where_it_breaks_phase_1_well_defined_symbol`` — NL→symbol only per model, SWI
  well-defined check (``nl_symbol_conversion_assess``), optional LLM termination repair
  (``check_wib_phase_1_termination_async`` in ``llm_prolog.selector``), then persist.
  Use a dedicated ``artifacts_root`` subdirectory (e.g. ``phase-1-wd/``) or filter
  ``run_meta.ablation.variant_id`` when ranking so results do not mix with legacy Phase 1.
- ``run_where_it_breaks_phase_1_wd_reanalyze_from_stored_run`` (offline) — load
  ``examples.jsonl`` + ``run_meta.json`` from an existing run directory, re-run the
  well-defined SWI check using ``initial_premises_for_hybrid_reuse_from_stored_result``
  (same seeding as symbolic-hybrid reuse), and write metrics under a new directory.
  ``run_where_it_breaks_phase_1_wd_reanalyze_under_parent`` walks subdirectories that
  contain ``examples.jsonl`` and writes one artifact tree each (paths mirror the source layout).

Logged / persisted per run (see eval.artifact):
- run_meta.json: run_id, pipeline_mode, dataset (subset_spec incl. example_ids), pipeline_config,
  seed, model_specs_by_role, ablation, code_version, suite_name, overall_accuracy,
  run_timing, llm_usage, failure_counts_by_category, harness (max_in_flight, suite_seed,
  dataset_subset_seed)
- examples.jsonl: example_id, problem, ground_truth, validator, obtained,
  success, reason, output_summary (mode-specific)
- failures.jsonl: example_id, failure_id, failure_category, failure_note,
  component_context (roles, step_index, used_premise_ids), debug_snapshot

Planned derived outputs (to be produced by analysis scripts):
- ablation_variants.json — list of variant specs before runs
- model_rankings.json — accuracy ladder from Phase 1
- failure_mode_profiles.json — aggregates from failures.jsonl
- representative_examples.json — ids for qualitative follow-up

Phase 2 has two entry points:
- ``run_where_it_breaks_phase_2`` — full pipeline per example; sweeps ``nl_to_symbol`` and ``selector``.
- ``run_where_it_breaks_phase_2_reuse_initial_symbolization`` — initial clauses from a prior run's
  ``examples.jsonl``; selector loop only (``run_symbolic_hybrid_after_nl_async``;
  ``EvaluationSuite.symbolic_hybrid_initial_by_example_id``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval import GPT_4_1_MINI
from eval.artifact.analyze_artifacts import _get_model_family_and_name
from eval.artifact.artifact_persist import new_run_id, persist_evaluation_run
from eval.artifact.validate_artifacts import validate_run_dir
from eval.eval_suite import (
    EvaluationSuite,
    ExampleOutcome,
    LLMRole,
    ModelMapping,
    ModelSpec,
    PipelineMode,
    SimpleEvalTask,
    SuiteReport,
    TaskReport,
    _collect_outcomes_in_order,
    _get_example_fields,
    _make_outcome_from_exception,
    _make_outcome_from_result,
    _placeholder_outcome,
)
from eval.metrics.well_defined_nl_symbol import (
    NlSymbolWellDefinedOutcome,
    nl_symbol_conversion_assess,
    summarize_well_defined_nl_symbol_metrics,
    well_defined_nl_symbol_summary_to_jsonable,
)
from llm_prolog.llm_client.async_llm_client import AsyncLLMClient
from llm_prolog.llm_executor import LLMExecutor
from llm_prolog.nl_symbol_converter import convert_problem_to_symbols_async
from llm_prolog.pipeline import PipelineConfig
from llm_prolog.selector import check_wib_phase_1_termination_async
from llm_prolog.symbolic.types import (
    AnswerSpec,
    Fact,
    PipelineResult,
    Predicate,
    PredicateArg,
    Premise,
    Term,
    initial_premises_for_hybrid_reuse_from_stored_result,
)


def _default_where_it_breaks_root(project_root: Path) -> Path:
    return project_root / "artifacts" / "where-it-breaks"


def _dataset_meta_from_suite(suite: EvaluationSuite) -> Dict[str, Any]:
    """
    Produce the `dataset` dict consumed by `persist_evaluation_run`.

    This follows the pattern in `eval/evaluate_symbolic_hybrid.py`.
    """
    # Best-effort: pull from the first task.
    task = suite.tasks[0] if suite.tasks else None
    task_id = getattr(task, "task_id", "unknown_task")
    task_id_str = str(task_id)
    split = "train" if ":train" in task_id_str else "test" if ":test" in task_id_str else "unknown"
    return {
        "name": "gsm8k" if task_id_str.startswith("gsm8k") else "unknown",
        "split": split,
        "subset_spec": {"task_id": task_id_str},
    }


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run_and_persist(
    *,
    suite: EvaluationSuite,
    artifacts_root: Path,
    variant_id: str,
    dataset_meta: Mapping[str, Any],
    ablation_overrides: Mapping[str, Any] | None = None,
    write_failures: bool = True,
    max_in_flight: int = 15,
) -> Tuple[Any, Path]:
    report = asyncio.run(
        suite.run_async(
            max_in_flight=max_in_flight,
            print_progress=True,
            show_openrouter_balance=True,
        )
    )

    _roles = suite.pipeline_mode.get_required_roles()
    _model = suite.model_by_role[_roles[0]].model if _roles else ""
    run_id = new_run_id(_model)
    run_dir = persist_evaluation_run(
        artifacts_root=artifacts_root.resolve(),
        run_id=run_id,
        suite=suite,
        suite_report=report,
        dataset=dataset_meta,
        ablation={
            "variant_id": variant_id,
            "component_overrides": dict(ablation_overrides or {}),
        },
        write_failures=write_failures,
    )
    print(f"Artifacts written to: {run_dir}")

    ok, errs = validate_run_dir(run_dir)
    if not ok:
        for err in errs:
            print(f"[artifact validation] {err}")
        raise RuntimeError(f"Artifact validation failed for run directory: {run_dir}")
    print("Artifact validation passed.")
    return report, run_dir


def _spec_to_name(spec: ModelSpec) -> str:
    return spec.model.split("/")[-1]


def _suite_with_all_roles_set_to(
    *,
    base_suite: EvaluationSuite,
    spec: ModelSpec,
) -> EvaluationSuite:
    return EvaluationSuite(
        name=base_suite.name,
        tasks=base_suite.tasks,
        pipeline_mode=base_suite.pipeline_mode,
        model_by_role=ModelMapping.set_spec_to_all_roles(spec, base_suite.pipeline_mode),
        prompt_overrides=base_suite.prompt_overrides,
        pipeline_cfg=base_suite.pipeline_cfg,
        keep_all_outcomes=base_suite.keep_all_outcomes,
        keep_random_k=base_suite.keep_random_k,
        seed=base_suite.seed,
    )


def _suite_with_role_overrides(
    *,
    base_suite: EvaluationSuite,
    default_spec: ModelSpec,
    overrides: Mapping[str, ModelSpec],
) -> EvaluationSuite:
    """
    Clone suite, setting all required roles to default_spec, then overriding named roles.
    """
    mapping = ModelMapping.set_spec_to_all_roles(default_spec, base_suite.pipeline_mode)
    for role_name, spec in overrides.items():
        role = LLMRole(role_name)
        mapping.mapping[role] = spec
    return EvaluationSuite(
        name=base_suite.name,
        tasks=base_suite.tasks,
        pipeline_mode=base_suite.pipeline_mode,
        model_by_role=mapping,
        prompt_overrides=base_suite.prompt_overrides,
        pipeline_cfg=base_suite.pipeline_cfg,
        keep_all_outcomes=base_suite.keep_all_outcomes,
        keep_random_k=base_suite.keep_random_k,
        seed=base_suite.seed,
    )


def model_spec_nl_to_symbol_from_run_meta(run_meta: Mapping[str, Any]) -> ModelSpec:
    d = ((run_meta.get("model_specs_by_role") or {}).get("nl_to_symbol")) or {}
    model = d.get("model")
    if not model:
        raise ValueError("run_meta.json missing model_specs_by_role.nl_to_symbol.model")
    return ModelSpec(
        model=str(model),
        temperature=d.get("temperature"),
        max_tokens=d.get("max_tokens"),
    )


def load_symbolic_hybrid_initial_state_from_run_dir(
    run_dir: str | Path,
) -> Tuple[Dict[str, Tuple[List[Premise], AnswerSpec]], Dict[str, Any]]:
    """
    Read a persisted evaluation run directory and build per-example (initial premises,
    answer_spec) for symbolic hybrid reuse.

    Requires full PipelineResult objects in examples.jsonl ``output`` fields.
    Premises are NL→symbol clauses plus, when that row ``success`` is true and the stored
    pipeline ``success`` is true, any ``final_termination_check`` linking rules from
    ``final_premises`` (see ``initial_premises_for_hybrid_reuse_from_stored_result``).
    """
    p = Path(run_dir).resolve()
    meta_path = p / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"run_meta.json not found under {p}")
    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if run_meta.get("pipeline_mode") != "symbolic_hybrid":
        raise ValueError(
            f"Expected pipeline_mode 'symbolic_hybrid', got {run_meta.get('pipeline_mode')!r}"
        )
    ex_path = p / "examples.jsonl"
    if not ex_path.is_file():
        raise FileNotFoundError(f"examples.jsonl not found under {p}")

    out: Dict[str, Tuple[List[Premise], AnswerSpec]] = {}
    for line in ex_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        eid = str(row.get("example_id", ""))
        payload = row.get("output")
        if not isinstance(payload, dict) or payload.get("result_type") != "PipelineResult":
            raise ValueError(
                f"example_id={eid!r}: need output.result_type=='PipelineResult' to reuse premises"
            )
        pr = PipelineResult.from_json_dict(payload)
        example_ok = bool(row.get("success"))
        initials = initial_premises_for_hybrid_reuse_from_stored_result(
            pr,
            example_task_success=example_ok,
        )
        if not initials:
            raise ValueError(f"example_id={eid!r}: no initial NL premises in stored final_premises")
        out[eid] = (initials, pr.answer_spec)
    if not out:
        raise ValueError(f"No examples loaded from {ex_path}")
    return out, run_meta


def nl_to_symbol_model_id_from_run_meta(run_meta: Mapping[str, Any]) -> str | None:
    """Best-effort NL→symbol model id from persisted ``run_meta.json`` (matches Phase 1 / reuse helpers)."""
    d = ((run_meta.get("model_specs_by_role") or {}).get("nl_to_symbol")) or {}
    m = d.get("model")
    if m:
        return str(m)
    co = (run_meta.get("ablation") or {}).get("component_overrides") or {}
    ar = co.get("all_roles")
    return str(ar) if ar else None


def run_where_it_breaks_phase_1_wd_reanalyze_from_stored_run(
    *,
    source_run_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Offline Phase‑1‑style well-defined check: read ``examples.jsonl`` from ``source_run_dir``,
    score each row with ``initial_premises_for_hybrid_reuse_from_stored_result`` semantics
    (``initial_premises_for_hybrid_reuse_from_stored_result``), and write JSON artifacts
    to ``output_dir``. Copies ``run_meta.json`` for provenance and records the NL→symbol model
    from that file.
    """
    src = Path(source_run_dir).resolve()
    out = _ensure_dir(Path(output_dir).resolve())
    meta_path = src / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"run_meta.json not found under {src}")
    ex_path = src / "examples.jsonl"
    if not ex_path.is_file():
        raise FileNotFoundError(f"examples.jsonl not found under {src}")

    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = summarize_well_defined_nl_symbol_metrics(src)
    nl_model = nl_to_symbol_model_id_from_run_meta(run_meta)
    payload = {
        "analysis": "where_it_breaks_phase_1_wd_reanalyze_hybrid_initial",
        "source_run_dir": str(src),
        "nl_to_symbol_model_from_run_meta": nl_model,
        "run_id": run_meta.get("run_id"),
        "summary": well_defined_nl_symbol_summary_to_jsonable(summary),
    }
    (out / "well_defined_hybrid_initial_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "run_meta_source.json").write_text(
        json.dumps(run_meta, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote hybrid-initial well-defined reanalysis to {out}")
    return out


def discover_run_dirs_with_examples_jsonl(
    parent_dir: str | Path,
    *,
    recursive: bool = True,
) -> List[Path]:
    """
    Return directories under ``parent_dir`` that contain ``examples.jsonl``.

    If ``recursive`` is False, only immediate subdirectories are considered.
    If True, every directory that contains ``examples.jsonl`` anywhere under ``parent_dir``
    is returned (including ``parent_dir`` itself when applicable), sorted by path.
    """
    parent = Path(parent_dir).resolve()
    if not parent.is_dir():
        raise NotADirectoryError(str(parent))
    found: List[Path] = []
    if recursive:
        for ex in sorted(parent.rglob("examples.jsonl")):
            found.append(ex.parent)
    else:
        for child in sorted(parent.iterdir()):
            if child.is_dir() and (child / "examples.jsonl").is_file():
                found.append(child)
    # De-dupe while preserving order
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in found:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def run_where_it_breaks_phase_1_wd_reanalyze_under_parent(
    *,
    source_runs_parent: str | Path,
    output_parent: str | Path,
    recursive: bool = True,
) -> List[Path]:
    """
    For each directory under ``source_runs_parent`` that contains ``examples.jsonl``, run
    :func:`run_where_it_breaks_phase_1_wd_reanalyze_from_stored_run` and write under
    ``output_parent`` preserving relative paths (e.g. ``phase-1-wd/run_.../`` → same shape).
    """
    parent = Path(source_runs_parent).resolve()
    out_root = _ensure_dir(Path(output_parent).resolve())
    run_dirs = discover_run_dirs_with_examples_jsonl(parent, recursive=recursive)
    if not run_dirs:
        raise ValueError(f"No examples.jsonl found under {parent} (recursive={recursive})")
    written: List[Path] = []
    for run_dir in run_dirs:
        rel = run_dir.relative_to(parent)
        out = out_root / rel
        written.append(
            run_where_it_breaks_phase_1_wd_reanalyze_from_stored_run(
                source_run_dir=run_dir,
                output_dir=out,
            )
        )
    return written


def _validate_seeds_cover_suite_tasks(
    suite: EvaluationSuite,
    seeds: Mapping[str, Any],
) -> None:
    for task in suite.tasks:
        for i, ex in enumerate(task.load_examples()):
            eid = str(getattr(ex, "id", i))
            if eid not in seeds:
                raise KeyError(
                    f"Initial-state map missing example_id={eid!r} (task {task.task_id!r})"
                )


def _suite_phase2_reuse_nl_from_artifact(
    *,
    base_suite: EvaluationSuite,
    best_spec: ModelSpec,
    nl_spec_from_artifact: ModelSpec,
    selector_spec: ModelSpec,
    seeds: Dict[str, Tuple[List[Premise], AnswerSpec]],
) -> EvaluationSuite:
    mapping = ModelMapping.set_spec_to_all_roles(best_spec, base_suite.pipeline_mode)
    mapping.mapping[LLMRole.NL_TO_SYMBOL] = nl_spec_from_artifact
    mapping.mapping[LLMRole.SELECTOR] = selector_spec
    return EvaluationSuite(
        name=base_suite.name,
        tasks=base_suite.tasks,
        pipeline_mode=base_suite.pipeline_mode,
        model_by_role=mapping,
        prompt_overrides=base_suite.prompt_overrides,
        pipeline_cfg=base_suite.pipeline_cfg,
        keep_all_outcomes=base_suite.keep_all_outcomes,
        keep_random_k=base_suite.keep_random_k,
        seed=base_suite.seed,
        symbolic_hybrid_initial_by_example_id=seeds,
    )


def run_where_it_breaks_phase_1(
    *,
    base_suite: EvaluationSuite,
    ladder_specs: Sequence[ModelSpec],
    artifacts_root: Path,
    write_failures: bool = True,
    max_in_flight: int = 15,
) -> List[Path]:
    """
    Phase 1: run the same EvaluationSuite across a list of ModelSpecs (applied to all required roles).
    """
    run_dirs: List[Path] = []
    dataset_meta = _dataset_meta_from_suite(base_suite)
    for spec in ladder_specs:
        suite = _suite_with_all_roles_set_to(base_suite=base_suite, spec=spec)
        suite.name = f"[where-it-breaks phase-1] {_spec_to_name(spec)} :: {base_suite.name}"
        _, run_dir = _run_and_persist(
            suite=suite,
            artifacts_root=artifacts_root,
            variant_id="where_it_breaks_phase_1",
            dataset_meta=dataset_meta,
            ablation_overrides={"all_roles": spec.model},
            write_failures=write_failures,
            max_in_flight=max_in_flight,
        )
        run_dirs.append(run_dir)
    return run_dirs


def _wib_ground_truth_value(ex: Any, expected_fallback: str) -> Any:
    g = getattr(ex, "ground_truth", None)
    if g is not None:
        return g
    try:
        return float(expected_fallback)
    except (TypeError, ValueError):
        return expected_fallback


def _wib_constant_name_for_ground_truth(gt: Any) -> str:
    if isinstance(gt, float) and math.isfinite(gt) and gt == round(gt):
        return str(int(gt))
    if isinstance(gt, int):
        return str(gt)
    if isinstance(gt, float) and math.isfinite(gt):
        return str(gt)
    s = str(gt).strip()
    return s if s else "0"


def _wib_substitute_var_in_predicate(pred: Predicate, var_name: str, const_name: str) -> Predicate:
    new_args: List[PredicateArg] = []
    for a in pred.args:
        if isinstance(a, Term):
            if a.is_variable and a.name == var_name:
                new_args.append(Term.constant(const_name))
            else:
                new_args.append(a)
        else:
            new_args.append(_wib_substitute_var_in_predicate(a, var_name, const_name))
    return Predicate(name=pred.name, args=tuple(new_args))


def _wib_synthetic_answer_fact_premise(answer_spec: AnswerSpec, ground_truth: Any) -> Premise:
    cname = _wib_constant_name_for_ground_truth(ground_truth)
    pred = _wib_substitute_var_in_predicate(answer_spec.target, answer_spec.variable_name, cname)
    return Premise(
        id=-1,
        clause=Fact(predicate=pred),
        nl=None,
        source="wib_phase1_synthetic_answer",
    )


def _pipeline_result_wib_phase1_well_defined(
    *,
    initial_premises: List[Premise],
    answer_spec: AnswerSpec,
    ground_truth: Any,
    task_success: bool,
    reason: str,
    llm_interactions: List[Dict[str, Any]],
    repair_premise: Optional[Premise],
) -> PipelineResult:
    final = list(initial_premises)
    if repair_premise is not None:
        final.append(repair_premise)
    final_sorted = sorted(final, key=lambda p: p.id)
    ap: Optional[Premise] = None
    if task_success:
        ap = _wib_synthetic_answer_fact_premise(answer_spec, ground_truth)
    return PipelineResult(
        success=task_success,
        answer_premise=ap,
        steps=[],
        answer_spec=answer_spec,
        final_premises=final_sorted,
        reason=reason,
        llm_interactions=llm_interactions,
    )


def _wib_reason_from_nl_outcome(o: NlSymbolWellDefinedOutcome, *, prefix: str) -> str:
    if o.ok:
        return f"{prefix}_success"
    return f"{prefix}_{o.category.value}"


async def _run_wib_phase1_task_async(
    task: SimpleEvalTask,
    suite: EvaluationSuite,
    llm_exec: LLMExecutor,
    *,
    tc_spec: ModelSpec,
    max_tc_candidates: int,
    max_in_flight: int,
    print_progress: bool,
) -> TaskReport:
    examples = list(task.load_examples())
    total = len(examples)
    ordered_outcomes: List[Optional[ExampleOutcome]] = [None] * total
    ordered_correct: List[bool] = [False] * total

    async def run_one(i: int, ex: Any) -> Tuple[int, ExampleOutcome, bool]:
        problem, expected, example_id = _get_example_fields(task, ex, i)
        gt = _wib_ground_truth_value(ex, expected)
        nl_spec = suite.model_by_role[LLMRole.NL_TO_SYMBOL]
        try:
            if print_progress:
                print(
                    f"Starting [wib phase1 wd] task {task.task_id}, example {example_id} "
                    f"at: {datetime.now().strftime('%H:%M:%S')}."
                )
            po = suite.prompt_overrides
            nl_prompt_ov = po.get(LLMRole.NL_TO_SYMBOL) if po else None
            initial_premises, answer_spec, nl_trace = await convert_problem_to_symbols_async(
                problem,
                llm_exec,
                model=getattr(nl_spec, "model", None),
                temperature=getattr(nl_spec, "temperature", None),
                max_tokens=getattr(nl_spec, "max_tokens", None),
                system_prompt_override=nl_prompt_ov if isinstance(nl_prompt_ov, str) else None,
            )
            initial = list(initial_premises)
            interactions: List[Dict[str, Any]] = [nl_trace]
            wd0 = nl_symbol_conversion_assess(initial, answer_spec, gt)

            repair_premise: Optional[Premise] = None
            if wd0.ok:
                pr = _pipeline_result_wib_phase1_well_defined(
                    initial_premises=initial,
                    answer_spec=answer_spec,
                    ground_truth=gt,
                    task_success=True,
                    reason="wib_phase1_nl_symbol_well_defined",
                    llm_interactions=interactions,
                    repair_premise=None,
                )
            else:
                aux_list, tc_trace = await check_wib_phase_1_termination_async(
                    problem,
                    initial,
                    answer_spec,
                    llm_exec=llm_exec,
                    max_candidates=max_tc_candidates,
                    model=getattr(tc_spec, "model", None),
                    temperature=getattr(tc_spec, "temperature", None),
                    max_tokens=getattr(tc_spec, "max_tokens", None),
                )
                interactions.append(tc_trace)
                last_wd = wd0
                ok_after = False
                for aux in aux_list:
                    combined = initial + [aux]
                    wd_try = nl_symbol_conversion_assess(combined, answer_spec, gt)
                    last_wd = wd_try
                    if wd_try.ok:
                        ok_after = True
                        repair_premise = aux
                        break
                if ok_after and repair_premise is not None:
                    pr = _pipeline_result_wib_phase1_well_defined(
                        initial_premises=initial,
                        answer_spec=answer_spec,
                        ground_truth=gt,
                        task_success=True,
                        reason="wib_phase1_tc_repair",
                        llm_interactions=interactions,
                        repair_premise=repair_premise,
                    )
                else:
                    pr = _pipeline_result_wib_phase1_well_defined(
                        initial_premises=initial,
                        answer_spec=answer_spec,
                        ground_truth=gt,
                        task_success=False,
                        reason=_wib_reason_from_nl_outcome(last_wd, prefix="wib_phase1"),
                        llm_interactions=interactions,
                        repair_premise=None,
                    )

            outcome, ok = _make_outcome_from_result(
                task=task,
                pipeline_mode=suite.pipeline_mode,
                ex=ex,
                idx=i,
                example_id=example_id,
                problem=problem,
                expected=expected,
                result=pr,
            )
            if print_progress:
                print(
                    f"Completed [wib phase1 wd] task {task.task_id}, example {example_id} "
                    f"at: {datetime.now().strftime('%H:%M:%S')}."
                )
        except Exception as e:
            import traceback

            tb_str = traceback.format_exc()
            outcome, ok = _make_outcome_from_exception(
                idx=i,
                example_id=example_id,
                problem=problem,
                expected=expected,
                exc=Exception(f"{e}\nStack trace:\n{tb_str}"),
            )
            if print_progress:
                print(
                    f"Failed [wib phase1 wd] task {task.task_id}, example {example_id} "
                    f"at: {datetime.now().strftime('%H:%M:%S')}."
                )
        return i, outcome, ok

    tasks = [run_one(i, ex) for i, ex in enumerate(examples)]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    for i, outcome, ok in results:
        ordered_outcomes[i] = outcome
        ordered_correct[i] = ok

    correct = sum(1 for x in ordered_correct if x)
    rng = random.Random(suite.seed)
    outcomes_collect = _collect_outcomes_in_order(
        outcomes_in_order=[ordered_outcomes[i] or _placeholder_outcome(i) for i in range(total)],
        rng=rng,
        keep_all_outcomes=suite.keep_all_outcomes,
        keep_random_k=suite.keep_random_k,
    )
    accuracy = (correct / total) if total else 0.0
    return TaskReport(
        task_id=task.task_id,
        pipeline_mode=suite.pipeline_mode,
        total=total,
        correct=correct,
        accuracy=accuracy,
        outcomes=tuple(outcomes_collect),
        extra_stats={"max_in_flight": max_in_flight},
    )


async def _run_wib_phase1_suite_async(
    suite: EvaluationSuite,
    *,
    tc_spec: ModelSpec,
    max_tc_candidates: int,
    max_in_flight: int,
    print_progress: bool,
    show_openrouter_balance: bool,
) -> SuiteReport:
    async with AsyncLLMClient() as client:
        client.reset_usage_stats()
        usage_before: Dict[str, Any] = {}
        if show_openrouter_balance:
            usage_before = dict(client.get_usage_stats())
            print(f"[LLM usage] Before suite run: {usage_before}")
        wall0 = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        executor = LLMExecutor(client, max_in_flight=max(1, max_in_flight))
        treps: List[TaskReport] = []
        for task in suite.tasks:
            treps.append(
                await _run_wib_phase1_task_async(
                    task,
                    suite,
                    executor,
                    tc_spec=tc_spec,
                    max_tc_candidates=max_tc_candidates,
                    max_in_flight=max_in_flight,
                    print_progress=print_progress,
                )
            )
        finished_at = datetime.now(timezone.utc).isoformat()
        duration_s = time.perf_counter() - wall0
        llm_usage = client.get_usage_stats()
        if show_openrouter_balance:
            print(f"[LLM usage] After suite run: {llm_usage}")
            delta = {
                k: float(llm_usage.get(k, 0)) - float(usage_before.get(k, 0))
                for k in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "n_requests",
                    "cost_usd",
                )
            }
            print(
                f"[LLM usage] This suite — cost_usd={delta.get('cost_usd', 0):.6f}, "
                f"total_tokens={delta.get('total_tokens', 0)}, n_requests={delta.get('n_requests', 0)}"
            )
        run_metadata: Dict[str, Any] = {
            "run_timing": {
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": round(duration_s, 6),
            },
            "llm_usage": llm_usage,
            "max_in_flight": max_in_flight,
        }
        return SuiteReport(
            pipeline_mode=suite.pipeline_mode,
            task_reports=tuple(treps),
            run_metadata=run_metadata,
        )


def run_where_it_breaks_phase_1_well_defined_symbol(
    *,
    base_suite: EvaluationSuite,
    ladder_specs: Sequence[ModelSpec],
    artifacts_root: Path,
    termination_checker_spec: ModelSpec | None = None,
    max_tc_candidates: int = 5,
    write_failures: bool = True,
    max_in_flight: int = 15,
) -> List[Path]:
    """
    Phase 1 (well-defined variant): NL→symbol per ladder model, SWI well-defined metric, optional
    ``check_wib_phase_1_termination_async`` repair, then persist. Does not run the selector loop.

    Persist ``ablation.variant_id`` is ``where_it_breaks_phase_1_well_defined`` (distinct from legacy
    ``where_it_breaks_phase_1``). Prefer writing under e.g. ``.../phase-1-wd/`` so
    ``get_phase1_models_ordered_by_accuracy`` on a mixed tree does not blend modes unless filtered.
    """
    tc = termination_checker_spec or ModelSpec(model=GPT_4_1_MINI, temperature=0.5, max_tokens=None)
    dataset_meta = _dataset_meta_from_suite(base_suite)

    async def _run_all() -> List[Path]:
        run_dirs: List[Path] = []
        for spec in ladder_specs:
            suite = _suite_with_all_roles_set_to(base_suite=base_suite, spec=spec)
            suite.name = (
                f"[where-it-breaks phase-1 well-defined] {_spec_to_name(spec)} :: {base_suite.name}"
            )
            report = await _run_wib_phase1_suite_async(
                suite,
                tc_spec=tc,
                max_tc_candidates=max_tc_candidates,
                max_in_flight=max_in_flight,
                print_progress=True,
                show_openrouter_balance=True,
            )
            _roles = suite.pipeline_mode.get_required_roles()
            _model = suite.model_by_role[_roles[0]].model if _roles else ""
            run_id = new_run_id(_model)
            run_dir = persist_evaluation_run(
                artifacts_root=artifacts_root.resolve(),
                run_id=run_id,
                suite=suite,
                suite_report=report,
                dataset=dataset_meta,
                ablation={
                    "variant_id": "where_it_breaks_phase_1_well_defined",
                    "component_overrides": {
                        "all_roles": spec.model,
                        "termination_checker_model": tc.model,
                        "max_tc_candidates": max_tc_candidates,
                    },
                },
                write_failures=write_failures,
            )
            print(f"Artifacts written to: {run_dir}")
            ok, errs = validate_run_dir(run_dir)
            if not ok:
                for err in errs:
                    print(f"[artifact validation] {err}")
                raise RuntimeError(f"Artifact validation failed for run directory: {run_dir}")
            print("Artifact validation passed.")
            run_dirs.append(run_dir)
        return run_dirs

    return asyncio.run(_run_all())


def run_where_it_breaks_phase_2(
    *,
    base_suite: EvaluationSuite,
    best_spec: ModelSpec,
    ladder_specs: Sequence[ModelSpec],
    artifacts_root: Path,
    write_failures: bool = True,
    max_in_flight: int = 15,
) -> List[Path]:
    """
    Phase 2 (full pipeline): staircase analysis.

    For each role in ``nl_to_symbol`` and ``selector``:
    - fix all roles to ``best_spec``
    - sweep the target role over ``ladder_specs`` (typically ordered worst retained → best fixed)
    """
    run_dirs: List[Path] = []
    dataset_meta = _dataset_meta_from_suite(base_suite)
    for role_name in ("nl_to_symbol", "selector"):
        for k, spec in enumerate(ladder_specs):
            suite = _suite_with_role_overrides(
                base_suite=base_suite,
                default_spec=best_spec,
                overrides={role_name: spec},
            )
            suite.name = (
                f"[where-it-breaks phase-2] sweep={role_name} k={k} "
                f"{_spec_to_name(spec)} (best={_spec_to_name(best_spec)}) :: {base_suite.name}"
            )
            _, run_dir = _run_and_persist(
                suite=suite,
                artifacts_root=artifacts_root,
                variant_id="where_it_breaks_phase_2",
                dataset_meta=dataset_meta,
                ablation_overrides={
                    "fixed_best_model": best_spec.model,
                    "swept_role": role_name,
                    "swept_model": spec.model,
                    "swept_index": k,
                },
                write_failures=write_failures,
                max_in_flight=max_in_flight,
            )
            run_dirs.append(run_dir)
    return run_dirs


def run_where_it_breaks_phase_2_reuse_initial_symbolization(
    *,
    base_suite: EvaluationSuite,
    best_spec: ModelSpec,
    ladder_specs: Sequence[ModelSpec],
    artifacts_root: Path,
    source_run_dir: str | Path,
    write_failures: bool = True,
    max_in_flight: int = 15,
) -> List[Path]:
    """
    Phase 2 with reused initial premises: load ``run_meta.json`` + ``examples.jsonl`` from
    ``source_run_dir``, take NL→symbol clauses from each stored ``PipelineResult``, and sweep
    only the **selector** role over ``ladder_specs``. The nl_to_symbol model on the suite matches
    ``run_meta.json`` (no NL calls in this mode).
    """
    run_dirs: List[Path] = []
    dataset_meta = _dataset_meta_from_suite(base_suite)
    reuse_path = str(Path(source_run_dir).resolve())
    seeds, run_meta = load_symbolic_hybrid_initial_state_from_run_dir(reuse_path)
    nl_spec_from_artifact = model_spec_nl_to_symbol_from_run_meta(run_meta)
    source_run_id = str(run_meta.get("run_id", ""))
    _validate_seeds_cover_suite_tasks(base_suite, seeds)

    role_name = "selector"
    for k, spec in enumerate(ladder_specs):
        suite = _suite_phase2_reuse_nl_from_artifact(
            base_suite=base_suite,
            best_spec=best_spec,
            nl_spec_from_artifact=nl_spec_from_artifact,
            selector_spec=spec,
            seeds=seeds,
        )
        suite.name = (
            f"[where-it-breaks phase-2] reuse_nl sweep={role_name} k={k} "
            f"{_spec_to_name(spec)} (best={_spec_to_name(best_spec)}) :: {base_suite.name}"
        )
        _, run_dir = _run_and_persist(
            suite=suite,
            artifacts_root=artifacts_root,
            variant_id="where_it_breaks_phase_2",
            dataset_meta=dataset_meta,
            ablation_overrides={
                "fixed_best_model": best_spec.model,
                "swept_role": role_name,
                "swept_model": spec.model,
                "swept_index": k,
                "reuse_initial_symbolization_run_dir": reuse_path,
                "source_run_id": source_run_id,
                "nl_to_symbol_model_from_artifact": nl_spec_from_artifact.model,
            },
            write_failures=write_failures,
            max_in_flight=max_in_flight,
        )
        run_dirs.append(run_dir)
    return run_dirs


def _parse_model_specs_csv(text: str, *, temperature: float | None, max_tokens: int | None) -> List[ModelSpec]:
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    return [ModelSpec(model=p, temperature=temperature, max_tokens=max_tokens) for p in parts]


def _load_run_metas_from_dir(root: Path) -> List[Dict[str, Any]]:
    run_metas: List[Dict[str, Any]] = []
    if not root.is_dir():
        return run_metas
    for meta_path in sorted(root.rglob("run_meta.json")):
        try:
            run_metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[WARN] Failed to read {meta_path}: {e}")
    return run_metas


def get_phase1_models_ordered_by_accuracy(phase1_dir: str | Path) -> List[str]:
    """
    Given a directory containing where-it-breaks Phase 1 artifacts (i.e., containing run dirs with
    `run_meta.json`), return model ids ordered by accuracy (highest to lowest).

    This is intended for automatically constructing Phase 2's ladder ordering from Phase 1 results.

    If the folder mixes legacy Phase 1 runs (``ablation.variant_id`` ``where_it_breaks_phase_1``)
    with the well-defined variant (``where_it_breaks_phase_1_well_defined``), each ``run_meta.json``
    is still read; the latest ``overall_accuracy`` per model wins. Prefer separate roots (e.g.
    ``phase-1/`` vs ``phase-1-wd/``) when comparing modes.
    """
    phase1_root = Path(phase1_dir).resolve()
    phase1_metas = _load_run_metas_from_dir(phase1_root)
    if not phase1_metas:
        raise ValueError(f"No run_meta.json files found under {phase1_root}")

    # Collect per-model accuracy (latest/last-seen value wins if duplicates exist).
    model_to_acc: Dict[str, float] = {}
    seen_order: List[str] = []
    for rm in phase1_metas:
        model = (
            ((rm.get("model_specs_by_role") or {}).get("nl_to_symbol") or {}).get("model")
            or ((rm.get("ablation") or {}).get("component_overrides") or {}).get("all_roles")
            or "unknown"
        )
        model_str = str(model)
        if model_str not in seen_order:
            seen_order.append(model_str)

        acc = rm.get("overall_accuracy")
        try:
            model_to_acc[model_str] = float(acc) if acc is not None else 0.0
        except (TypeError, ValueError):
            model_to_acc[model_str] = 0.0

    # Sort by accuracy desc; tie-break by first-seen order for stability.
    order_index = {m: i for i, m in enumerate(seen_order)}
    ranked_models = sorted(
        model_to_acc.keys(),
        key=lambda m: (-model_to_acc[m], order_index.get(m, 10**9), m),
    )
    return ranked_models


def analyze_where_it_breaks_folder(
    parent_folder: str | Path,
    *,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Analyze a where-it-breaks artifact parent folder and generate:
    - Phase 1 bar chart: one bar per model, y=accuracy.
    - Phase 2 scatter chart: x=role-to-sweep, dots=phase-2 run accuracies colored by swept model,
      plus reference accuracies from phase 1.

    `parent_folder` should contain `phase-1/` and `phase-2/`.
    """
    import matplotlib.pyplot as plt

    parent = Path(parent_folder).resolve()
    phase1_root = parent / "phase-1"
    phase2_root = parent / "phase-2"
    phase1_metas = _load_run_metas_from_dir(phase1_root)
    phase2_metas = _load_run_metas_from_dir(phase2_root)

    if not phase1_metas:
        raise ValueError(f"No run_meta.json files found under {phase1_root}")
    if not phase2_metas:
        raise ValueError(f"No run_meta.json files found under {phase2_root}")

    # ----- Phase 1 data (model -> accuracy) -----
    phase1_model_acc: Dict[str, float] = {}
    for rm in phase1_metas:
        model = (
            ((rm.get("model_specs_by_role") or {}).get("nl_to_symbol") or {}).get("model")
            or ((rm.get("ablation") or {}).get("component_overrides") or {}).get("all_roles")
            or "unknown"
        )
        acc = rm.get("overall_accuracy")
        try:
            phase1_model_acc[str(model)] = float(acc) if acc is not None else 0.0
        except (TypeError, ValueError):
            phase1_model_acc[str(model)] = 0.0

    # Preserve first-seen order from phase 1 metas where possible.
    seen_order: List[str] = []
    for rm in phase1_metas:
        model = (
            ((rm.get("model_specs_by_role") or {}).get("nl_to_symbol") or {}).get("model")
            or ((rm.get("ablation") or {}).get("component_overrides") or {}).get("all_roles")
            or "unknown"
        )
        model_str = str(model)
        if model_str not in seen_order:
            seen_order.append(model_str)
    phase1_models = [m for m in seen_order if m in phase1_model_acc]
    phase1_accs = [phase1_model_acc[m] for m in phase1_models]
    model_families      = [_get_model_family_and_name(m)[0] for m in phase1_models]
    short_phase1_models = [_get_model_family_and_name(m)[1] for m in phase1_models]

    # ----- Plot 1: phase 1 bar graph -----
    # Assign each family a unique color using a qualitative colormap (e.g. tab20)
    unique_families = list(dict.fromkeys(model_families))  # preserves order
    cmap = plt.get_cmap("tab20")
    family_to_color = {fam: cmap(i % 20) for i, fam in enumerate(unique_families)}
    bar_colors = [family_to_color[fam] for fam in model_families]

    fig1 = plt.figure(figsize=(max(7, 0.9 * len(phase1_models)), 4.5))
    ax1 = fig1.add_subplot(111)
    bars = ax1.bar(range(len(short_phase1_models)), phase1_accs, color=bar_colors)
    ax1.set_xticks(range(len(short_phase1_models)))
    ax1.set_xticklabels(short_phase1_models, rotation=35, ha="right")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Where-it-breaks Phase 1: model performance")
    ax1.set_ylim(0.0, 1.0)

    # Make a legend mapping family to color
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=family_to_color[fam], label=fam) for fam in unique_families]
    ax1.legend(handles=handles, title="Model family", loc="best")

    fig1.tight_layout()

    # ----- Phase 2 data (role, model, accuracy) -----
    phase2_points: List[Tuple[str, str, float, int | None]] = []
    fixed_best_models: List[str] = []
    for rm in phase2_metas:
        overrides = ((rm.get("ablation") or {}).get("component_overrides") or {})
        role = overrides.get("swept_role")
        model = overrides.get("swept_model")
        swept_index = overrides.get("swept_index")
        fbm = overrides.get("fixed_best_model")
        if isinstance(fbm, str) and fbm:
            fixed_best_models.append(fbm)
        if role is None:
            role = "unknown"
        if model is None:
            # Fallback: infer from role's assigned model.
            role_models = rm.get("model_specs_by_role") or {}
            model = (role_models.get(str(role)) or {}).get("model", "unknown")
        acc = rm.get("overall_accuracy")
        try:
            acc_val = float(acc) if acc is not None else 0.0
        except (TypeError, ValueError):
            acc_val = 0.0
        idx_val: int | None = None
        if isinstance(swept_index, int):
            idx_val = swept_index
        else:
            try:
                idx_val = int(swept_index)
            except (TypeError, ValueError):
                idx_val = None
        phase2_points.append((str(role), str(model), acc_val, idx_val))

    # Best model for titles/leftmost reference.
    best_model: str
    if fixed_best_models:
        # Choose most common fixed-best model, to guard against mixed folders.
        from collections import Counter

        best_model = Counter(fixed_best_models).most_common(1)[0][0]
    else:
        # Fallback: best model is the top Phase 1 performer.
        best_model = max(phase1_model_acc.items(), key=lambda kv: kv[1])[0]
    best_model_short = _get_model_family_and_name(best_model)[1]

    role_order = ["nl_to_symbol", "selector"]
    all_roles = sorted({r for r, _m, _a, _i in phase2_points if r not in role_order})
    role_order.extend(all_roles)
    role_to_x = {r: i for i, r in enumerate(role_order)}

    models_for_colors = sorted({m for _r, m, _a, _i in phase2_points} | set(phase1_models))
    cmap = plt.get_cmap("tab20")
    model_to_color = {m: cmap(i % 20) for i, m in enumerate(models_for_colors)}

    # ----- Plot 2: phase 2 role sweep scatter + phase 1 reference -----
    fig2 = plt.figure(figsize=(max(7, 1.8 * len(role_order)), 4.8))
    ax2 = fig2.add_subplot(111)

    # Plot phase 2 dots.
    for role, model, acc, _idx in phase2_points:
        x = role_to_x.get(role, 0)
        ax2.scatter(
            x,
            acc,
            color=model_to_color.get(model, "gray"),
            marker="o",
            s=65,
            alpha=0.9,
        )

    # Overlay phase 1 reference points (same model's all-roles baseline) at each role.
    for model, ref_acc in phase1_model_acc.items():
        for role in role_order:
            x = role_to_x[role]
            ax2.scatter(
                x,
                ref_acc,
                color=model_to_color.get(model, "gray"),
                marker="x",
                s=60,
                alpha=0.8,
            )

    ax2.set_xticks(range(len(role_order)))
    ax2.set_xticklabels(role_order)
    ax2.set_xlabel("Role-to-sweep")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(
        f"Where-it-breaks Phase 2: staircase performance (+ phase 1 references) | best={best_model_short}"
    )
    ax2.set_xlim(-1.0, len(role_order))
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis="y", alpha=0.2)

    # Legend: color per model + marker meaning.
    from matplotlib.lines import Line2D

    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=model_to_color[m], markersize=7, label=m.split("/")[-1])
        for m in models_for_colors
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="phase 2 run"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="phase 1 reference"),
    ]
    ax2.legend(handles=model_handles + marker_handles, loc="best", fontsize=8, ncol=2)
    fig2.tight_layout()

    # ----- Additional Phase 2 figures: one Phase-1-style bar chart per swept component -----
    # Group phase 2 points by role, keeping ladder order by swept_index when present.
    points_by_role: Dict[str, List[Tuple[str, float, int | None]]] = {}
    for role, model, acc, idx in phase2_points:
        points_by_role.setdefault(role, []).append((model, acc, idx))

    component_figs: Dict[str, Any] = {}
    for role in role_order:
        if role not in points_by_role:
            continue
        pts = points_by_role[role]
        # Sort by swept_index when available; otherwise stable by model name.
        pts_sorted = sorted(
            pts,
            key=lambda t: (t[2] is None, t[2] if t[2] is not None else 10**9, t[0]),
        )
        swept_models = [m for (m, _a, _i) in pts_sorted]
        swept_accs = [a for (_m, a, _i) in pts_sorted]

        # Build x-axis order with best model at leftmost.
        model_order = [best_model] + [m for m in swept_models if m != best_model]
        phase2_acc_by_model = {m: a for m, a, _i in pts_sorted}
        phase2_accs_aligned: List[float | None] = []
        phase1_accs_aligned: List[float] = []
        for m in model_order:
            phase2_accs_aligned.append(phase2_acc_by_model.get(m))
            phase1_accs_aligned.append(phase1_model_acc.get(m, 0.0))

        short_models = [_get_model_family_and_name(m)[1] for m in model_order]
        families = [_get_model_family_and_name(m)[0] for m in model_order]
        unique_fams = list(dict.fromkeys(families))
        fam_to_color = {fam: plt.get_cmap("tab20")(i % 20) for i, fam in enumerate(unique_fams)}
        colors = [fam_to_color[fam] for fam in families]

        figc = plt.figure(figsize=(max(7, 0.9 * len(model_order)), 4.8))
        axc = figc.add_subplot(111)
        xs = list(range(len(model_order)))
        width = 0.42

        # Phase 2 bars (solid). If missing (e.g. best model isn't run in that sweep), draw zero-height.
        phase2_heights = [float(v) if v is not None else 0.0 for v in phase2_accs_aligned]
        axc.bar(
            [x - width / 2 for x in xs],
            phase2_heights,
            width=width,
            color=colors,
            label="phase 2 (sweep run)",
        )

        # Phase 1 reference bars (hatched contour).
        axc.bar(
            [x + width / 2 for x in xs],
            phase1_accs_aligned,
            width=width,
            facecolor="none",
            edgecolor=colors,
            hatch="..",
            linewidth=1.5,
            label="phase 1 (reference)",
        )

        # Best model Phase 1 marker (star) at the leftmost position.
        best_ref_acc = phase1_model_acc.get(best_model, 0.0)
        axc.scatter(
            [0],
            [best_ref_acc],
            marker="*",
            s=220,
            color="black",
            zorder=5,
            label="best model (phase 1)",
        )

        axc.set_xticks(xs)
        axc.set_xticklabels(short_models, rotation=35, ha="right")
        axc.set_xlabel("Model (best model shown at leftmost)")
        axc.set_ylabel("Accuracy")
        axc.set_ylim(0.0, 1.0)
        axc.set_title(
            f"Where-it-breaks Phase 2 sweep: {role} | best={best_model_short} (phase 1 star)"
        )
        axc.grid(axis="y", alpha=0.2)

        # Legend: families + bar meaning + star.
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        fam_handles = [Patch(facecolor=fam_to_color[f], label=f) for f in unique_fams]
        style_handles = [
            Patch(facecolor="gray", edgecolor="gray", label="phase 2 (sweep run)"),
            Patch(facecolor="none", edgecolor="gray", hatch="..", label="phase 1 (reference)"),
            Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=12, label="best model (phase 1)"),
        ]
        axc.legend(handles=fam_handles + style_handles, title="Legend", loc="best", fontsize=8, ncol=2)
        figc.tight_layout()
        component_figs[role] = figc

    if save_dir is not None:
        out = Path(save_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out / "where_it_breaks_phase1_performance.png", dpi=200, bbox_inches="tight")
        fig2.savefig(out / "where_it_breaks_phase2_staircase.png", dpi=200, bbox_inches="tight")
        for role, figc in component_figs.items():
            figc.savefig(out / f"where_it_breaks_phase2_sweep_{role}.png", dpi=200, bbox_inches="tight")
        print(f"[where-it-breaks] Saved plots to {out}")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        for figc in component_figs.values():
            plt.close(figc)

    return {
        "parent_folder": str(parent),
        "phase1_model_acc": phase1_model_acc,
        "phase2_points": [
            {"role": r, "model": m, "accuracy": a, "swept_index": i} for (r, m, a, i) in phase2_points
        ],
    }

def where_it_breaks_50exs():
    # Internal constants for a fixed reproducible experiment.
    gsm8k_size = 50
    gsm8k_seed = 99
    from_train_split = True  # matches __main__ default behavior
    temperature = 0.5
    max_tokens = None
    max_in_flight = 15
    write_failures = True

    # Use all models provided by eval/__init__.py.
    from eval import get_all_model_names_filtered

    all_model_names = get_all_model_names_filtered(
        include_llama405b=False,
        include_llama_family=True,
        include_gpt_family=True,
        include_qwen_family=True
    )

    project_root = Path(__file__).resolve().parents[1]
    where_root = project_root / "artifacts" / "where-it-breaks-50EXS"

    # Import here to avoid forcing dataset deps on module import.
    from eval.per_dataset.eval_gsm8k import (
        gsm8k_main_validator,
        gsm8k_success_measure,
        load_gsm8k_examples,
    )
    from eval.eval_suite import default_expected_repr, default_problem_str

    task = SimpleEvalTask(
        task_id=(
            f"gsm8k:{gsm8k_size}exs:{'train' if from_train_split else 'test'}:seed={gsm8k_seed}"
        ),
        examples=load_gsm8k_examples(
            size=gsm8k_size, seed=gsm8k_seed, from_train_split=from_train_split
        ),
        validator_fn=gsm8k_main_validator,
        success_measure_fn=gsm8k_success_measure,
        problem_fn=default_problem_str,
        expected_fn=default_expected_repr,
    )

    ladder_models_csv = ",".join(all_model_names)
    ladder_specs = _parse_model_specs_csv(
        ladder_models_csv,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not ladder_specs:
        raise SystemExit("No models parsed from ALL_MODEL_NAMES")

    mode = PipelineMode.SYMBOLIC_HYBRID
    pipeline_cfg = PipelineConfig(
        max_steps=20, 
        explain=False,
        use_termination_checks=False,
        use_final_termination_check=True,
        selector_num_candidates=3
        )
    base_suite = EvaluationSuite(
        name=f"Symbolic Hybrid On GSM8K ({gsm8k_size}EX)",
        tasks=[task],
        pipeline_mode=mode,
        model_by_role=ModelMapping.set_spec_to_all_roles(ladder_specs[0], mode),
        prompt_overrides=None,
        pipeline_cfg=pipeline_cfg,
        keep_all_outcomes=True,
        keep_random_k=0,
        seed=0,
    )

    phase1_root = _ensure_dir(where_root / "phase-1")
    phase2_root = _ensure_dir(where_root / "phase-2")
    (where_root / "README.json").write_text(
        json.dumps(
            {
                "analysis": "where-it-breaks",
                "layout": {
                    "phase-1": str(phase1_root),
                    "phase-2": str(phase2_root),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Phase 1: run across all models to measure accuracy ladder.
    run_where_it_breaks_phase_1(
        base_suite=base_suite,
        ladder_specs=ladder_specs,
        artifacts_root=phase1_root,
        write_failures=write_failures,
        max_in_flight=max_in_flight,
    )

    # Phase 2: run staircase sweep using phase-1 accuracy ordering (best -> worst).
    ranked_models = get_phase1_models_ordered_by_accuracy(phase1_root)
    spec_by_model = {s.model: s for s in ladder_specs}
    ranked_specs: List[ModelSpec] = [
        spec_by_model.get(
            m, ModelSpec(model=m, temperature=temperature, max_tokens=max_tokens)
        )
        for m in ranked_models
    ]
    if ranked_specs:
        phase2_best_spec = ranked_specs[0]
        phase2_ladder_specs = ranked_specs[1:]
    else:
        phase2_best_spec = ladder_specs[0]
        phase2_ladder_specs = ladder_specs[1:]

    run_where_it_breaks_phase_2(
        base_suite=base_suite,
        best_spec=phase2_best_spec,
        ladder_specs=phase2_ladder_specs,
        artifacts_root=phase2_root,
        write_failures=write_failures,
        max_in_flight=max_in_flight,
    )

    # Produce the two required figures (phase 1 + phase 2).
    return analyze_where_it_breaks_folder(where_root, show=False, save_dir=where_root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the where-it-breaks Phase 1 + Phase 2 analyses.")
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "Root directory to write artifacts under. "
            "If omitted, defaults to <project_root>/artifacts/where-it-breaks."
        ),
    )
    p.add_argument(
        "--phase",
        choices=["1", "1-wd", "2", "all", "1-wd-reanalyze", "1-wd-reanalyze-batch"],
        default="all",
        help=(
            "Which phase(s): 1=legacy full pipeline, 1-wd=NL→symbol well-defined + TC repair, "
            "2=staircase, all=legacy phase 1 then phase 2 (not 1-wd). "
            "1-wd-reanalyze=offline SWI well-defined on hybrid-initial premises from one run dir; "
            "1-wd-reanalyze-batch=same for every subdirectory (see --reanalyze-non-recursive) with examples.jsonl."
        ),
    )
    p.add_argument(
        "--tc-model",
        type=str,
        default=GPT_4_1_MINI,
        help="OpenRouter model id for WIB Phase 1 termination repair (--phase 1-wd only).",
    )
    p.add_argument(
        "--tc-n-candidates",
        type=int,
        default=5,
        help="Max distinct linking-rule candidates from the termination repair LLM (1-wd only).",
    )
    p.add_argument(
        "--phase2-reuse-run-dir",
        type=Path,
        default=None,
        help=(
            "For phase 2 only: evaluation run directory (run_meta.json + examples.jsonl) "
            "to reuse stored initial NL symbolizations; sweeps selector only."
        ),
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated ordered model ids (best first). Required for phases 1, 1-wd, 2, all.",
    )
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--no-failures", action="store_true", help="Do not write failures.jsonl.")
    p.add_argument("--max-in-flight", type=int, default=15)
    p.add_argument(
        "--gsm8k-size",
        type=int,
        default=None,
        help="How many GSM8K examples to evaluate (required for phases 1, 1-wd, 2, all).",
    )
    p.add_argument("--gsm8k-seed", type=int, default=42)
    p.add_argument("--gsm8k-train", action="store_true", help="Use GSM8K train split (default).")
    p.add_argument("--gsm8k-test", action="store_true", help="Use GSM8K test split.")
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing where-it-breaks artifacts under --artifacts-dir.",
    )
    p.add_argument(
        "--analysis-save-dir",
        type=Path,
        default=None,
        help="If provided, save generated phase-1/phase-2 plots to this directory.",
    )
    p.add_argument(
        "--no-show-plots",
        action="store_true",
        help="Generate plots without opening an interactive window.",
    )
    p.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Existing evaluation run directory with run_meta.json + examples.jsonl (--phase 1-wd-reanalyze).",
    )
    p.add_argument(
        "--reanalyze-output-dir",
        type=Path,
        default=None,
        help="Directory to write well_defined_hybrid_initial_summary.json and run_meta_source.json.",
    )
    p.add_argument(
        "--source-runs-parent",
        type=Path,
        default=None,
        help="Root to scan for run directories containing examples.jsonl (--phase 1-wd-reanalyze-batch).",
    )
    p.add_argument(
        "--reanalyze-output-parent",
        type=Path,
        default=None,
        help="Root under which to mirror each run's reanalysis output (--phase 1-wd-reanalyze-batch).",
    )
    p.add_argument(
        "--reanalyze-non-recursive",
        action="store_true",
        help="Batch mode: only immediate subdirectories of --source-runs-parent (default is recursive).",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if args.phase in ("1-wd-reanalyze", "1-wd-reanalyze-batch"):
        if args.phase == "1-wd-reanalyze":
            if args.source_run_dir is None or args.reanalyze_output_dir is None:
                raise SystemExit(
                    "--phase 1-wd-reanalyze requires --source-run-dir and --reanalyze-output-dir"
                )
            run_where_it_breaks_phase_1_wd_reanalyze_from_stored_run(
                source_run_dir=args.source_run_dir,
                output_dir=args.reanalyze_output_dir,
            )
        else:
            if args.source_runs_parent is None or args.reanalyze_output_parent is None:
                raise SystemExit(
                    "--phase 1-wd-reanalyze-batch requires --source-runs-parent and --reanalyze-output-parent"
                )
            run_where_it_breaks_phase_1_wd_reanalyze_under_parent(
                source_runs_parent=args.source_runs_parent,
                output_parent=args.reanalyze_output_parent,
                recursive=not args.reanalyze_non_recursive,
            )
        raise SystemExit(0)

    where_root = args.artifacts_dir.resolve() if args.artifacts_dir else _default_where_it_breaks_root(project_root)

    if args.analyze_only:
        analyze_where_it_breaks_folder(
            where_root,
            show=not args.no_show_plots,
            save_dir=args.analysis_save_dir,
        )
        raise SystemExit(0)

    if args.gsm8k_size is None:
        raise SystemExit(
            "--gsm8k-size is required for this phase (or use --analyze-only / 1-wd-reanalyze / 1-wd-reanalyze-batch)."
        )
    if not args.models:
        raise SystemExit(
            "--models is required for this phase (or use --analyze-only / 1-wd-reanalyze / 1-wd-reanalyze-batch)."
        )

    # Import here to avoid forcing dataset deps on module import.
    from eval.per_dataset.eval_gsm8k import (
        gsm8k_main_validator,
        gsm8k_success_measure,
        load_gsm8k_examples,
    )
    from eval.eval_suite import default_expected_repr, default_problem_str

    from_train_split = True
    if args.gsm8k_test:
        from_train_split = False
    if args.gsm8k_train:
        from_train_split = True

    # Build the "exact same" base EvaluationSuite (tasks/pipeline_cfg fixed; only models vary per phase).
    task = SimpleEvalTask(
        task_id=f"gsm8k:{args.gsm8k_size}exs:{'train' if from_train_split else 'test'}:seed={args.gsm8k_seed}",
        examples=load_gsm8k_examples(size=args.gsm8k_size, seed=args.gsm8k_seed, from_train_split=from_train_split),
        validator_fn=gsm8k_main_validator,
        success_measure_fn=gsm8k_success_measure,
        problem_fn=default_problem_str,
        expected_fn=default_expected_repr,
    )

    ladder_specs = _parse_model_specs_csv(
        args.models,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if not ladder_specs:
        raise SystemExit("No models parsed from --models")

    mode = PipelineMode.SYMBOLIC_HYBRID
    pipeline_cfg = PipelineConfig(
        max_steps=20, 
        explain=False,
        use_termination_checks=False,
        use_final_termination_check=True,
        selector_num_candidates=3
        )
    base_suite = EvaluationSuite(
        name=f"Symbolic Hybrid On GSM8K ({args.gsm8k_size}EX)",
        tasks=[task],
        pipeline_mode=mode,
        model_by_role=ModelMapping.set_spec_to_all_roles(ladder_specs[0], mode),
        prompt_overrides=None,
        pipeline_cfg=pipeline_cfg,
        keep_all_outcomes=True,
        keep_random_k=0,
        seed=0,
    )

    phase1_root = _ensure_dir(where_root / "phase-1")
    phase1_wd_root = _ensure_dir(where_root / "phase-1-wd")
    phase2_root = _ensure_dir(where_root / "phase-2")
    (where_root / "README.json").write_text(
        json.dumps(
            {
                "analysis": "where-it-breaks",
                "layout": {
                    "phase-1": str(phase1_root),
                    "phase-1-wd": str(phase1_wd_root),
                    "phase-2": str(phase2_root),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_failures = not args.no_failures
    if args.phase in ("1", "all"):
        run_where_it_breaks_phase_1(
            base_suite=base_suite,
            ladder_specs=ladder_specs,
            artifacts_root=phase1_root,
            write_failures=write_failures,
            max_in_flight=args.max_in_flight,
        )
    if args.phase == "1-wd":
        run_where_it_breaks_phase_1_well_defined_symbol(
            base_suite=base_suite,
            ladder_specs=ladder_specs,
            artifacts_root=phase1_wd_root,
            termination_checker_spec=ModelSpec(
                model=args.tc_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ),
            max_tc_candidates=args.tc_n_candidates,
            write_failures=write_failures,
            max_in_flight=args.max_in_flight,
        )
    if args.phase in ("2", "all"):
        # If phase 2 runs after phase 1 in the same invocation, prefer the empirically-derived
        # phase-1 accuracy ordering (best -> worst) instead of trusting the user-provided order.
        phase2_best_spec = ladder_specs[0]
        phase2_ladder_specs = list(ladder_specs[1:])
        if args.phase == "all":
            ranked_models = get_phase1_models_ordered_by_accuracy(phase1_root)
            spec_by_model = {s.model: s for s in ladder_specs}
            ranked_specs: List[ModelSpec] = [
                spec_by_model.get(
                    m,
                    ModelSpec(model=m, temperature=args.temperature, max_tokens=args.max_tokens),
                )
                for m in ranked_models
            ]
            if ranked_specs:
                phase2_best_spec = ranked_specs[0]
                phase2_ladder_specs = ranked_specs[1:]
        if args.phase2_reuse_run_dir is not None:
            run_where_it_breaks_phase_2_reuse_initial_symbolization(
                base_suite=base_suite,
                best_spec=phase2_best_spec,
                ladder_specs=phase2_ladder_specs,
                artifacts_root=phase2_root,
                source_run_dir=args.phase2_reuse_run_dir,
                write_failures=write_failures,
                max_in_flight=args.max_in_flight,
            )
        else:
            run_where_it_breaks_phase_2(
                base_suite=base_suite,
                best_spec=phase2_best_spec,
                ladder_specs=phase2_ladder_specs,
                artifacts_root=phase2_root,
                write_failures=write_failures,
                max_in_flight=args.max_in_flight,
            )
