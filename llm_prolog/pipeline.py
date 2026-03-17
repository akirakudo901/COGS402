"""
Pipeline orchestrator for the LLM‑Prolog system.

This module wires together:
- NL‑Symbol converter
- Selector
- Symbolic inference engine
- Optional Symbol‑NL converter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple

from tqdm import tqdm

from .llm_client.llm_client import LLMClient
from .llm_executor import LLMExecutor
from .cot_baseline import run_cot_baseline, run_cot_baseline_async
from .nl_symbol_converter import convert_problem_to_symbols, convert_problem_to_symbols_async
from .selector import select_next_step, select_next_step_async
from .symbolic.inference import infer_new_premise, unify_predicates
from .symbolic.types import (
    AnswerSpec,
    PipelineResult,
    PipelineStep,
    Premise,
    SelectorDecision,
    parse_fact_or_rule
)
from .symbol_nl_converter import symbols_to_nl, symbols_to_nl_async


@dataclass
class PipelineConfig:
    max_steps: int = 10
    explain: bool = True


def _role_key(role: Any) -> str:
    """
    Normalize a role key (Enum or string) to a stable string.
    Expected inputs:
    - string keys like 'nl_to_symbol'
    - Enum keys with a `.value` string
    """
    if isinstance(role, str):
        return role
    value = getattr(role, "value", None)
    if isinstance(value, str):
        return value
    return str(role)


def _get_model_spec(model_by_role: Optional[Mapping[Any, Any]], role: str) -> Any | None:
    if not model_by_role:
        return None
    for k, v in model_by_role.items():
        if _role_key(k) == role:
            return v
    return None


def _get_prompt_override(prompt_overrides: Optional[Mapping[Any, str]], role: str) -> str | None:
    if not prompt_overrides:
        return None
    for k, v in prompt_overrides.items():
        if _role_key(k) == role:
            return v
    return None


def _append_background_premises(
    premises: List[Premise],
    background_clauses: List[str],
) -> List[Premise]:
    next_id = max((p.id for p in premises), default=0) + 1
    for text in background_clauses:
        clause = parse_fact_or_rule(str(text))
        premises.append(
            Premise(
                id=next_id,
                clause=clause,
                nl=None,
                source="selector_background",
            )
        )
        next_id += 1
    return premises


def _find_premise_by_id(premises: List[Premise], pid: int) -> Optional[Premise]:
    for p in premises:
        if p.id == pid:
            return p
    return None


def _answer_matches(premise: Premise, answer_spec: AnswerSpec) -> bool:
    clause = premise.clause
    # Only facts can directly be answers for now.
    from .symbolic.types import Fact  # local import to avoid circulars

    if not isinstance(clause, Fact):
        return False

    # Require that the derived fact unify with the full answer head pattern,
    # including any constant arguments, and that the distinguished answer
    # variable be bound to a concrete value.
    subst = unify_predicates(answer_spec.target, clause.predicate)
    if subst is None:
        return False

    bound = subst.get(answer_spec.variable_name)
    return bound is not None and not bound.is_variable


def _apply_explanations_to_premises(premises: List[Premise], explanations: Mapping[int, str]) -> None:
    for p in premises:
        if p.id in explanations:
            p.nl = explanations[p.id]


def _process_symbolic_decision_step(
    *,
    step_idx: int,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    decision: SelectorDecision,
    used_premise_sets: Set[FrozenSet[int]],
) -> Tuple[List[Premise], Optional[Premise], PipelineStep, bool, Optional[str]]:
    """
    Apply one selector decision to the symbolic state.

    Returns:
    - updated premises (same list object; returned for convenience)
    - final answer premise if found
    - a PipelineStep to append
    - should_stop (break loop)
    - reason (if should_stop)
    """
    if decision.background_premises:
        premises = _append_background_premises(premises, decision.background_premises)

    if len(decision.selected_premise_ids) < 2:
        return (
            premises,
            None,
            PipelineStep(
                step_index=step_idx,
                used_premise_ids=decision.selected_premise_ids,
                new_premise=None,
                decision=decision,
                success=False,
                note="Selector did not choose two premises; skipping inference.",
            ),
            False,
            None,
        )

    # Detect reuse of an already‑combined set of premises (order‑insensitive).
    selected_set = frozenset(decision.selected_premise_ids)
    if selected_set in used_premise_sets:
        return (
            premises,
            None,
            PipelineStep(
                step_index=step_idx,
                used_premise_ids=decision.selected_premise_ids,
                new_premise=None,
                decision=decision,
                success=False,
                note="Inference step failed due to selecting premises already combined previously.",
            ),
            False,
            None,
        )

    selected_premises: List[Premise] = []
    missing_ids: List[int] = []
    for pid in decision.selected_premise_ids:
        premise = _find_premise_by_id(premises, pid)
        if premise is None:
            missing_ids.append(pid)
        else:
            selected_premises.append(premise)

    if missing_ids:
        return (
            premises,
            None,
            PipelineStep(
                step_index=step_idx,
                used_premise_ids=decision.selected_premise_ids,
                new_premise=None,
                decision=decision,
                success=False,
                note=f"Selector referenced unknown premise IDs: {missing_ids}",
            ),
            False,
            None,
        )

    used_premise_sets.add(selected_set)

    new_clause = infer_new_premise(selected_premises)
    if new_clause is None:
        return (
            premises,
            None,
            PipelineStep(
                step_index=step_idx,
                used_premise_ids=decision.selected_premise_ids,
                new_premise=None,
                decision=decision,
                success=False,
                note="Inference failed to derive a new clause from selected premises.",
            ),
            False,
            None,
        )

    new_id = max((p.id for p in premises), default=0) + 1
    new_premise = Premise(
        id=new_id,
        clause=new_clause,
        nl=None,
        source="inference",
        parent_ids=list(decision.selected_premise_ids),
    )
    premises.append(new_premise)

    step = PipelineStep(
        step_index=step_idx,
        used_premise_ids=decision.selected_premise_ids,
        new_premise=new_premise,
        decision=decision,
        success=True,
        note=None,
    )

    if _answer_matches(new_premise, answer_spec):
        return premises, new_premise, step, True, "answer_head_matched"

    return premises, None, step, False, None


def _run_symbolic_steps(
    problem: str,
    pipeline_cfg: PipelineConfig,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    *,
    llm: LLMClient,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
    step_iter: Optional[Iterable[int]] = None,
) -> tuple[bool, Optional[Premise], List[PipelineStep], Optional[str], Set[FrozenSet[int]]]:
    steps: List[PipelineStep] = []
    success = False
    final_answer: Optional[Premise] = None
    reason: Optional[str] = None
    used_premise_sets: Set[FrozenSet[int]] = set()

    iterator = step_iter if step_iter is not None else tqdm(range(pipeline_cfg.max_steps))
    for step_idx in iterator:
        decision: SelectorDecision = select_next_step(
            problem=problem,
            premises=premises,
            answer_spec=answer_spec,
            llm=llm,
            previous_premise_sets=[sorted(list(s)) for s in used_premise_sets],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt_override=system_prompt_override,
        )

        premises, found_answer, step, should_stop, stop_reason = _process_symbolic_decision_step(
            step_idx=step_idx,
            premises=premises,
            answer_spec=answer_spec,
            decision=decision,
            used_premise_sets=used_premise_sets,
        )
        steps.append(step)
        if should_stop:
            success = True
            final_answer = found_answer
            reason = stop_reason
            break

    return success, final_answer, steps, reason, used_premise_sets


async def _run_symbolic_steps_async(
    problem: str,
    pipeline_cfg: PipelineConfig,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    *,
    llm_exec: LLMExecutor,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
    step_iter: Optional[Iterable[int]] = None,
) -> tuple[bool, Optional[Premise], List[PipelineStep], Optional[str], Set[FrozenSet[int]]]:
    steps: List[PipelineStep] = []
    success = False
    final_answer: Optional[Premise] = None
    reason: Optional[str] = None
    used_premise_sets: Set[FrozenSet[int]] = set()

    iterator = step_iter if step_iter is not None else range(pipeline_cfg.max_steps)
    for step_idx in iterator:
        decision: SelectorDecision = await select_next_step_async(
            problem=problem,
            premises=premises,
            answer_spec=answer_spec,
            llm_exec=llm_exec,
            previous_premise_sets=[sorted(list(s)) for s in used_premise_sets],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt_override=system_prompt_override,
        )

        premises, found_answer, step, should_stop, stop_reason = _process_symbolic_decision_step(
            step_idx=step_idx,
            premises=premises,
            answer_spec=answer_spec,
            decision=decision,
            used_premise_sets=used_premise_sets,
        )
        steps.append(step)
        if should_stop:
            success = True
            final_answer = found_answer
            reason = stop_reason
            break

    return success, final_answer, steps, reason, used_premise_sets


def run_symbolic_hybrid_pipeline(
    problem: str,
    *,
    llm: LLMClient,
    pipeline_cfg : PipelineConfig,
    model_by_role: Optional[Mapping[Any, Any]] = None,
    prompt_overrides: Optional[Mapping[Any, str]] = None,
) -> PipelineResult:
    """
    Symbolic hybrid pipeline with optional per-component model/prompt overrides.

    Roles:
    - nl_to_symbol: NL->symbol conversion
    - selector: premise selection + background premise proposal
    - symbol_to_nl: NL explanations of symbolic premises

    If roles aren't specified, we fall back to the LLMClient's model & config.
    """

    nl2sym = _get_model_spec(model_by_role, "nl_to_symbol")
    premises, answer_spec = convert_problem_to_symbols(
        problem,
        llm,
        model=getattr(nl2sym, "model", None) if nl2sym else None,
        temperature=getattr(nl2sym, "temperature", None) if nl2sym else None,
        max_tokens=getattr(nl2sym, "max_tokens", None) if nl2sym else None,
        system_prompt_override=_get_prompt_override(prompt_overrides, "nl_to_symbol"),
    )

    sel_spec = _get_model_spec(model_by_role, "selector")
    success, final_answer, steps, reason, used_premise_sets = _run_symbolic_steps(
        problem=problem,
        pipeline_cfg=pipeline_cfg,
        premises=premises,
        answer_spec=answer_spec,
        llm=llm,
        model=getattr(sel_spec, "model", None) if sel_spec else None,
        temperature=getattr(sel_spec, "temperature", None) if sel_spec else None,
        max_tokens=getattr(sel_spec, "max_tokens", None) if sel_spec else None,
        system_prompt_override=_get_prompt_override(prompt_overrides, "selector"),
    )

    if not success and reason is None:
        reason = "max_steps_exhausted"

    if pipeline_cfg.explain:
        sym2nl = _get_model_spec(model_by_role, "symbol_to_nl")
        try:
            explanations = symbols_to_nl(
                problem,
                premises,
                llm,
                model=getattr(sym2nl, "model", None) if sym2nl else None,
                temperature=getattr(sym2nl, "temperature", None) if sym2nl else None,
                max_tokens=getattr(sym2nl, "max_tokens", None) if sym2nl else None,
                system_prompt_override=_get_prompt_override(prompt_overrides, "symbol_to_nl"),
            )
            _apply_explanations_to_premises(premises, explanations)
        except Exception:
            pass

    return PipelineResult(
        success=success,
        answer_premise=final_answer,
        steps=steps,
        answer_spec=answer_spec,
        final_premises=premises,
        reason=reason,
    )


async def run_symbolic_hybrid_pipeline_async(
    problem: str,
    *,
    llm_exec: LLMExecutor,
    pipeline_cfg: PipelineConfig,
    model_by_role: Optional[Mapping[Any, Any]] = None,
    prompt_overrides: Optional[Mapping[Any, str]] = None,
) -> PipelineResult:
    """
    Async symbolic hybrid pipeline; all LLM calls go through LLMExecutor.
    """
    nl2sym = _get_model_spec(model_by_role, "nl_to_symbol")
    premises, answer_spec = await convert_problem_to_symbols_async(
        problem,
        llm_exec,
        model=getattr(nl2sym, "model", None) if nl2sym else None,
        temperature=getattr(nl2sym, "temperature", None) if nl2sym else None,
        max_tokens=getattr(nl2sym, "max_tokens", None) if nl2sym else None,
        system_prompt_override=_get_prompt_override(prompt_overrides, "nl_to_symbol"),
    )

    sel_spec = _get_model_spec(model_by_role, "selector")
    success, final_answer, steps, reason, used_premise_sets = await _run_symbolic_steps_async(
        problem=problem,
        pipeline_cfg=pipeline_cfg,
        premises=premises,
        answer_spec=answer_spec,
        llm_exec=llm_exec,
        model=getattr(sel_spec, "model", None) if sel_spec else None,
        temperature=getattr(sel_spec, "temperature", None) if sel_spec else None,
        max_tokens=getattr(sel_spec, "max_tokens", None) if sel_spec else None,
        system_prompt_override=_get_prompt_override(prompt_overrides, "selector"),
    )

    if not success and reason is None:
        reason = "max_steps_exhausted"

    if pipeline_cfg.explain:
        sym2nl = _get_model_spec(model_by_role, "symbol_to_nl")
        try:
            explanations = await symbols_to_nl_async(
                problem,
                premises,
                llm_exec,
                model=getattr(sym2nl, "model", None) if sym2nl else None,
                temperature=getattr(sym2nl, "temperature", None) if sym2nl else None,
                max_tokens=getattr(sym2nl, "max_tokens", None) if sym2nl else None,
                system_prompt_override=_get_prompt_override(prompt_overrides, "symbol_to_nl"),
            )
            _apply_explanations_to_premises(premises, explanations)
        except Exception:
            pass

    return PipelineResult(
        success=success,
        answer_premise=final_answer,
        steps=steps,
        answer_spec=answer_spec,
        final_premises=premises,
        reason=reason,
    )


def run_pipeline_mode(
    *,
    problem: str,
    mode: Any,
    pipeline_cfg: PipelineConfig,
    llm: Optional[LLMClient] = None,
    model_by_role: Optional[Mapping[Any, Any]] = None,
    prompt_overrides: Optional[Mapping[Any, str]] = None
) -> Any:
    """
    Unified entrypoint for the evaluation suite.
    If roles aren't specified, we fall back to the LLMClient's model & config.

    - mode may be a string (e.g. 'symbolic_hybrid') or an Enum with `.value`.
    - Returns:
      - PipelineResult for symbolic hybrid
      - CoTResult for CoT baseline
    """
    client = llm or LLMClient()
    mode_key = _role_key(mode)

    if mode_key in ("symbolic_hybrid", "PipelineMode.SYMBOLIC_HYBRID"):
        return run_symbolic_hybrid_pipeline(
            problem,
            llm=client,
            pipeline_cfg=pipeline_cfg,
            model_by_role=model_by_role,
            prompt_overrides=prompt_overrides,
        )

    if mode_key in ("cot_baseline", "PipelineMode.COT_BASELINE"):
        cot_spec = _get_model_spec(model_by_role, "cot_solver")
        return run_cot_baseline(
            problem,
            llm=client,
            model_spec=cot_spec,
            system_prompt_override=_get_prompt_override(prompt_overrides, "cot_solver"),
        )

    raise ValueError(f"Unsupported pipeline mode: {mode_key!r}")


async def run_pipeline_mode_async(
    *,
    problem: str,
    mode: Any,
    pipeline_cfg: PipelineConfig,
    llm_exec: LLMExecutor,
    model_by_role: Optional[Mapping[Any, Any]] = None,
    prompt_overrides: Optional[Mapping[Any, str]] = None,
) -> Any:
    """
    Async unified entrypoint: runs one problem through the pipeline via LLMExecutor.
    Returns PipelineResult or CoTResult depending on mode.
    """
    mode_key = _role_key(mode)

    if mode_key in ("symbolic_hybrid", "PipelineMode.SYMBOLIC_HYBRID"):
        return await run_symbolic_hybrid_pipeline_async(
            problem,
            llm_exec=llm_exec,
            pipeline_cfg=pipeline_cfg,
            model_by_role=model_by_role,
            prompt_overrides=prompt_overrides,
        )

    if mode_key in ("cot_baseline", "PipelineMode.COT_BASELINE"):
        cot_spec = _get_model_spec(model_by_role, "cot_solver")
        return await run_cot_baseline_async(
            problem,
            llm_exec=llm_exec,
            model_spec=cot_spec,
            system_prompt_override=_get_prompt_override(prompt_overrides, "cot_solver"),
        )

    raise ValueError(f"Unsupported pipeline mode: {mode_key!r}")