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
from typing import FrozenSet, List, Optional, Set, Tuple

from .llm_client.llm_client import LLMClient
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
    return_premises : bool = True # return all premises at end of pipeline?


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


def run_pipeline(
    problem: str,
    query: str,
    llm: Optional[LLMClient] = None,
    config: Optional[PipelineConfig] = None,
) -> Tuple[PipelineResult, Optional[List[Premise]]]:
    """
    Run the full LLM‑Prolog pipeline on a single problem‑query pair.
    """
    cfg = config or PipelineConfig()
    client = llm or LLMClient()

    premises, answer_spec = convert_problem_to_symbols(problem, query, client)

    steps: List[PipelineStep] = []
    success = False
    final_answer: Optional[Premise] = None
    reason: Optional[str] = None
    used_premise_sets: Set[FrozenSet[int]] = set()

    for step_idx in range(cfg.max_steps):
        decision: SelectorDecision = select_next_step(
            problem=problem,
            query=query,
            premises=premises,
            answer_spec=answer_spec,
            llm=client,
            previous_premise_sets=[sorted(list(s)) for s in used_premise_sets],
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
        selected_premises = []
        missing_ids = []
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

        # Record that we've now attempted to combine this particular set of premises.
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

    # Optionally annotate all premises with NL explanations.
    if cfg.explain:
        try:
            explanations = symbols_to_nl(problem, query, premises, client)
            for p in premises:
                if p.id in explanations:
                    p.nl = explanations[p.id]
        except Exception:
            # Explanations are best‑effort; do not fail the pipeline if they break.
            pass

    result = PipelineResult(
        success=success,
        answer_premise=final_answer,
        steps=steps,
        reason=reason,
    )

    if cfg.return_premises:
        return result, premises
    else:
        return result
