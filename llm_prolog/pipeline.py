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
from typing import Any, FrozenSet, List, Mapping, Optional, Set

from tqdm import tqdm

from .llm_client.llm_client import LLMClient
from .cot_baseline import CoTResult, run_cot_baseline
from .nl_symbol_converter import convert_problem_to_symbols
from .selector import select_next_step
from .symbolic.inference import infer_new_premise, unify_predicates
from .symbolic.types import (
    AnswerSpec,
    PipelineResult,
    PipelineStep,
    Premise,
    SelectorDecision,
    parse_fact_or_rule
)
from .symbol_nl_converter import symbols_to_nl


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

    steps: List[PipelineStep] = []
    success = False
    final_answer: Optional[Premise] = None
    reason: Optional[str] = None
    used_premise_sets: Set[FrozenSet[int]] = set()

    sel_spec = _get_model_spec(model_by_role, "selector")
    for step_idx in tqdm(range(pipeline_cfg.max_steps)):
        decision: SelectorDecision = select_next_step(
            problem=problem,
            premises=premises,
            answer_spec=answer_spec,
            llm=llm,
            previous_premise_sets=[sorted(list(s)) for s in used_premise_sets],
            model=getattr(sel_spec, "model", None) if sel_spec else None,
            temperature=getattr(sel_spec, "temperature", None) if sel_spec else None,
            max_tokens=getattr(sel_spec, "max_tokens", None) if sel_spec else None,
            system_prompt_override=_get_prompt_override(prompt_overrides, "selector"),
        )

        # Integrate any new background premises first.
        if decision.background_premises:
            premises = _append_background_premises(premises, decision.background_premises)

        if len(decision.selected_premise_ids) < 2:
            steps.append(
                PipelineStep(
                    step_index=step_idx,
                    used_premise_ids=decision.selected_premise_ids,
                    new_premise=None,
                    decision=decision,
                    success=False,
                    note="Selector did not choose two premises; skipping inference.",
                )
            )
            continue

        # Detect reuse of an already‑combined set of premises (order‑insensitive).
        selected_set = frozenset(decision.selected_premise_ids)
        if selected_set in used_premise_sets:
            steps.append(
                PipelineStep(
                    step_index=step_idx,
                    used_premise_ids=decision.selected_premise_ids,
                    new_premise=None,
                    decision=decision,
                    success=False,
                    note="Inference step failed due to selecting premises already combined previously.",
                )
            )
            continue

        # Support variable number of selected_premise_ids
        selected_premises: List[Premise] = []
        missing_ids: List[int] = []
        for pid in decision.selected_premise_ids:
            premise = _find_premise_by_id(premises, pid)
            if premise is None:
                missing_ids.append(pid)
            else:
                selected_premises.append(premise)

        if missing_ids:
            steps.append(
                PipelineStep(
                    step_index=step_idx,
                    used_premise_ids=decision.selected_premise_ids,
                    new_premise=None,
                    decision=decision,
                    success=False,
                    note=f"Selector referenced unknown premise IDs: {missing_ids}",
                )
            )
            continue

        used_premise_sets.add(selected_set)

        new_clause = infer_new_premise(selected_premises)
        if new_clause is None:
            steps.append(
                PipelineStep(
                    step_index=step_idx,
                    used_premise_ids=decision.selected_premise_ids,
                    new_premise=None,
                    decision=decision,
                    success=False,
                    note="Inference failed to derive a new clause from selected premises.",
                )
            )
            continue

        new_id = max((p.id for p in premises), default=0) + 1
        new_premise = Premise(
            id=new_id,
            clause=new_clause,
            nl=None,
            source="inference",
            parent_ids=list(decision.selected_premise_ids),
        )
        premises.append(new_premise)

        steps.append(
            PipelineStep(
                step_index=step_idx,
                used_premise_ids=decision.selected_premise_ids,
                new_premise=new_premise,
                decision=decision,
                success=True,
                note=None,
            )
        )

        if _answer_matches(new_premise, answer_spec):
            success = True
            final_answer = new_premise
            reason = "answer_head_matched"
            break

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
            for p in premises:
                if p.id in explanations:
                    p.nl = explanations[p.id]
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