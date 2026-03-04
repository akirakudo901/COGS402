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
from typing import List, Optional, Tuple

from .llm_client import LLMClient
from .nl_symbol_converter import convert_problem_to_symbols
from .selector import select_next_step
from .inference import infer_new_premise
from .symbol_nl_converter import symbols_to_nl
from .types import (
    AnswerSpec,
    PipelineResult,
    PipelineStep,
    Premise,
    SelectorDecision,
    parse_fact_or_rule,
    same_predicate_shape,
)


@dataclass
class PipelineConfig:
    max_steps: int = 10
    explain: bool = True


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
    from .types import Fact  # local import to avoid circulars

    if not isinstance(clause, Fact):
        return False
    return same_predicate_shape(clause.predicate, answer_spec.target)


def run_pipeline(
    problem: str,
    query: str,
    llm: Optional[LLMClient] = None,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
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

    for step_idx in range(cfg.max_steps):
        decision: SelectorDecision = select_next_step(
            problem=problem,
            query=query,
            premises=premises,
            answer_spec=answer_spec,
            llm=client,
        )

        if decision.should_stop:
            reason = decision.stop_reason or "selector_requested_stop"
            break

        # Integrate any new background premises first.
        if decision.background_premises:
            premises = _append_background_premises(premises, decision.background_premises)

        if len(decision.selected_premise_ids) < 2:
            steps.append(
                PipelineStep(
                    step_index=step_idx,
                    used_premise_ids=decision.selected_premise_ids,
                    new_premise_id=None,
                    decision=decision,
                    success=False,
                    note="Selector did not choose two premises; skipping inference.",
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
                    new_premise_id=None,
                    decision=decision,
                    success=False,
                    note=f"Selector referenced unknown premise IDs: {missing_ids}",
                )
            )
            continue

        new_clause = infer_new_premise(*selected_premises)
        if new_clause is None:
            steps.append(
                PipelineStep(
                    step_index=step_idx,
                    used_premise_ids=decision.selected_premise_ids,
                    new_premise_id=None,
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

        step = PipelineStep(
            step_index=step_idx,
            used_premise_ids=decision.selected_premise_ids,
            new_premise_id=new_id,
            decision=decision,
            success=True,
            note=None,
        )
        steps.append(step)

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

    return PipelineResult(
        success=success,
        answer_premise=final_answer,
        steps=steps,
        reason=reason,
    )

