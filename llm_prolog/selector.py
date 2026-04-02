"""
Selector module with append-friendly prompt sessions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_client.llm_client import LLMClient
from .llm_executor import LLMExecutor
from .selector_legacy import (
    TerminationCheckerDecision,
    _build_system_prompt as _legacy_build_system_prompt,
    _decision_from_data,
    _termination_checker_decision_from_data,
)
from .symbolic.types import AnswerSpec, Premise, SelectorDecision, render_premises
from .system_prompts import TERMINATION_CHECK_SYSTEM_PROMPT


PREMISE_RENDER_VERBOSITY = 2


@dataclass
class SelectorPromptSession:
    """
    Initial snapshot plus incremental updates only (see `append_*` methods).
    The pipeline records the outcome of each symbolic step before the next selector call.
    Each API call is one user message via `generate_json`.
    """

    accumulated_user_prompt: str
    _outcome_count: int = field(default=0, repr=False)

    def append_success_new_premise(self, premise: Premise) -> None:
        """Append a delta after a successful inference step (new derived premise)."""
        self._outcome_count += 1
        self.accumulated_user_prompt += (
            f"## Outcome {self._outcome_count}, newly derived:\n\n"
            f"{render_premises([premise], verbosity_level=PREMISE_RENDER_VERBOSITY)}\n"
        )

    def append_latest_failure(
        self,
        *,
        note: str,
        proposed_premise: Optional[str],
        combined_premise_ids: List[int],
    ) -> None:
        """Append a single latest failed attempt (not grouped with prior failures)."""
        self._outcome_count += 1
        prop = proposed_premise if proposed_premise is not None else "None"
        ids_repr = ", ".join(str(i) for i in combined_premise_ids)
        self.accumulated_user_prompt += (
            f"## Outcome {self._outcome_count}, latest inference failed:\n\n"
            f"- note: {note}\n"
            f"- proposed_premise: {prop}\n"
            f"- combined_premise_ids: [{ids_repr}]\n"
        )


def select_next_step(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm: LLMClient,
    failed_steps_context: str = "",
    *,
    use_termination_checks: bool = True,
    allow_background_premises: bool = True,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
    prompt_session: SelectorPromptSession | None = None,
) -> tuple[SelectorDecision, Dict[str, Any]]:
    """
    Ask the LLM which premises to combine next and what goal to pursue.

    Always returns a second element: trace dict for the LLM call (prompts + raw/parsed answer).
    """
    user_content = _build_user_content(
        problem,
        premises,
        answer_spec,
        failed_steps_context,
        prompt_session=prompt_session,
    )
    system_prompt = _build_system_prompt(
        use_termination_checks=use_termination_checks,
        allow_background_premises=allow_background_premises,
        select_multiple_candidates=False,
        system_prompt_override=system_prompt_override,
    )
    parsed, raw = llm.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        return_raw=True,
    )
    decision = _decision_from_data(parsed)
    trace = {
        "component": "selector_select_next_step",
        "system_prompt": system_prompt,
        "user_prompt": user_content,
        "raw_answer": raw,
        "parsed_answer": parsed,
    }
    return decision, trace


def _build_system_prompt(
    *,
    use_termination_checks: bool,
    allow_background_premises: bool,
    select_multiple_candidates: bool,
    system_prompt_override: str | None,
) -> str:
    return _legacy_build_system_prompt(
        use_termination_checks=use_termination_checks,
        allow_background_premises=allow_background_premises,
        select_multiple_candidates=select_multiple_candidates,
        system_prompt_override=system_prompt_override,
    )


def select_next_step_candidates(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm: LLMClient,
    failed_steps_context: str = "",
    *,
    num_candidates: int = 1,
    use_termination_checks: bool = True,
    allow_background_premises: bool = True,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
    prompt_session: SelectorPromptSession | None = None,
) -> tuple[TerminationCheckerDecision, List[SelectorDecision], Dict[str, Any]]:
    """
    Ask the LLM for N candidate next steps (ordered by likelihood).

    Returns:
    - A termination-only decision (termination fields filled; selection fields empty)
    - A list of candidate SelectorDecisions (length <= num_candidates)
    - Trace dict for the LLM call (prompts + raw/parsed answer)
    """
    if num_candidates <= 1:
        one, trace = select_next_step(
            problem=problem,
            premises=premises,
            answer_spec=answer_spec,
            llm=llm,
            failed_steps_context=failed_steps_context,
            use_termination_checks=use_termination_checks,
            allow_background_premises=allow_background_premises,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt_override=system_prompt_override,
            prompt_session=prompt_session,
        )
        term_only = _termination_checker_decision_from_data(one.__dict__)
        return term_only, [one], trace

    user_content = _build_user_content(
        problem,
        premises,
        answer_spec,
        failed_steps_context,
        prompt_session=prompt_session,
    )
    user_content = (
        user_content
        + "\n\n"
        + "Generate candidates:\n"
        + f"- Generate exactly {int(num_candidates)} candidate(s).\n"
        + "- Each candidate must be distinct from the others.\n"
    )
    system_prompt = _build_system_prompt(
        use_termination_checks=use_termination_checks,
        allow_background_premises=allow_background_premises,
        select_multiple_candidates=True,
        system_prompt_override=system_prompt_override,
    )

    parsed, raw = llm.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        return_raw=True,
    )

    term_only = _termination_checker_decision_from_data(parsed)
    candidates_raw = parsed.get("candidates", [])
    if not isinstance(candidates_raw, list):
        candidates_raw = []

    candidates: List[SelectorDecision] = []
    for item in candidates_raw:
        if not isinstance(item, dict):
            continue
        cand = _decision_from_data(item)
        # Ensure termination fields don't interfere with candidate evaluation in the pipeline.
        cand.is_final_solution = False
        cand.solution_premise_id = None
        cand.answer_link_rule = None
        candidates.append(cand)
        if len(candidates) >= int(num_candidates):
            break

    trace = {
        "component": "selector_select_next_step",
        "system_prompt": system_prompt,
        "user_prompt": user_content,
        "raw_answer": raw,
        "parsed_answer": parsed,
    }
    return term_only, candidates, trace


def _build_user_content(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    failed_steps_context: str = "",
    *,
    prompt_session: SelectorPromptSession | None = None,
) -> str:
    if prompt_session is None:
        premises_block = render_premises(premises, verbosity_level=PREMISE_RENDER_VERBOSITY)
        failed_block = failed_steps_context or ""

        return (
            "Problem:\n"
            f"{problem.strip()}\n\n"
            "Current premises (by ID):\n"
            f"{premises_block}\n\n"
            f"{failed_block}"
            "Answer head predicate:\n"
            f"{answer_spec.target}\n\n"
            "Decide the next reasoning step following the instructions."
        )
    return (
        prompt_session.accumulated_user_prompt
        + "\n\nDecide the next reasoning step following the instructions."
    )


def init_selector_prompt_session(
    *,
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    use_termination_checks: bool = True,
    allow_background_premises: bool = True,
    system_prompt_override: str | None = None,
) -> SelectorPromptSession:
    """
    Prefix shared across all selector turns. Step-specific state is appended per call
    in `_build_user_content` via `accumulated_user_prompt`.
    """
    base = (
        "After each step, we append **only** its outcome: one new premise "
        "(success) or one latest failed attempt.\n\n"
        "**Problem**\n"
        f"{problem.strip()}\n\n"
        "**Answer head predicate**\n"
        f"{answer_spec.target}\n\n"
        "**Premises (by ID)**\n"
        f"{render_premises(premises, verbosity_level=PREMISE_RENDER_VERBOSITY)}\n"
    )
    return SelectorPromptSession(accumulated_user_prompt=base)


async def select_next_step_async(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm_exec: LLMExecutor,
    failed_steps_context: str = "",
    *,
    use_termination_checks: bool = True,
    allow_background_premises: bool = True,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
    prompt_session: SelectorPromptSession | None = None,
) -> tuple[SelectorDecision, Dict[str, Any]]:
    """
    Async version: ask the LLM which premises to combine next via LLMExecutor.

    Always returns a second element: trace dict for the LLM call (prompts + raw/parsed answer).
    """
    user_content = _build_user_content(
        problem,
        premises,
        answer_spec,
        failed_steps_context,
        prompt_session=prompt_session,
    )
    system_prompt = _build_system_prompt(
        use_termination_checks=use_termination_checks,
        allow_background_premises=allow_background_premises,
        select_multiple_candidates=False,
        system_prompt_override=system_prompt_override,
    )
    parsed, raw = await llm_exec.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        return_raw=True,
    )
    decision = _decision_from_data(parsed)
    trace = {
        "component": "selector_select_next_step",
        "system_prompt": system_prompt,
        "user_prompt": user_content,
        "raw_answer": raw,
        "parsed_answer": parsed,
    }
    return decision, trace


async def select_next_step_candidates_async(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm_exec: LLMExecutor,
    failed_steps_context: str = "",
    *,
    num_candidates: int = 1,
    use_termination_checks: bool = True,
    allow_background_premises: bool = True,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
    prompt_session: SelectorPromptSession | None = None,
) -> tuple[TerminationCheckerDecision, List[SelectorDecision], Dict[str, Any]]:
    """
    Async version of `select_next_step_candidates`.
    """
    if num_candidates <= 1:
        one, trace = await select_next_step_async(
            problem=problem,
            premises=premises,
            answer_spec=answer_spec,
            llm_exec=llm_exec,
            failed_steps_context=failed_steps_context,
            use_termination_checks=use_termination_checks,
            allow_background_premises=allow_background_premises,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt_override=system_prompt_override,
            prompt_session=prompt_session,
        )
        term_only = _termination_checker_decision_from_data(one.__dict__)
        return term_only, [one], trace

    user_content = _build_user_content(
        problem,
        premises,
        answer_spec,
        failed_steps_context,
        prompt_session=prompt_session,
    )
    user_content = (
        user_content
        + "\n\n"
        + "Generate candidates:\n"
        + f"- Generate exactly {int(num_candidates)} candidate(s).\n"
        + "- Each candidate must be distinct from the others.\n"
    )
    system_prompt = _build_system_prompt(
        use_termination_checks=use_termination_checks,
        allow_background_premises=allow_background_premises,
        select_multiple_candidates=True,
        system_prompt_override=system_prompt_override,
    )

    parsed, raw = await llm_exec.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        return_raw=True,
    )

    term_only = _termination_checker_decision_from_data(parsed)
    candidates_raw = parsed.get("candidates", [])
    if not isinstance(candidates_raw, list):
        candidates_raw = []

    candidates: List[SelectorDecision] = []
    for item in candidates_raw:
        if not isinstance(item, dict):
            continue
        cand = _decision_from_data(item)
        cand.is_final_solution = False
        cand.solution_premise_id = None
        cand.answer_link_rule = None
        candidates.append(cand)
        if len(candidates) >= int(num_candidates):
            break
    trace = {
        "component": "selector_select_next_step",
        "system_prompt": system_prompt,
        "user_prompt": user_content,
        "raw_answer": raw,
        "parsed_answer": parsed,
    }
    return term_only, candidates, trace


def check_termination(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    *,
    recent_premise: Optional[Premise],
    llm: LLMClient,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> tuple[TerminationCheckerDecision, Dict[str, Any]]:
    """
    Ask the LLM whether we have reached a final (ground-fact) solution and, if so,
    propose a linking rule that makes the answer head derivable.

    Always returns a second element: trace dict for the LLM call (prompts + raw/parsed answer).
    """
    user_content = _build_termination_checker_user_content(
        problem=problem,
        premises=premises,
        answer_spec=answer_spec,
        recent_premise=recent_premise,
    )
    system_prompt = system_prompt_override or TERMINATION_CHECK_SYSTEM_PROMPT
    parsed, raw = llm.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        return_raw=True,
    )
    trace = {
        "component": "final_termination_check",
        "system_prompt": system_prompt,
        "user_prompt": user_content,
        "raw_answer": raw,
        "parsed_answer": parsed,
    }
    return _termination_checker_decision_from_data(parsed), trace


def _build_termination_checker_user_content(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    recent_premise: Optional[Premise],
) -> str:
    premises_block = render_premises(premises, verbosity_level=PREMISE_RENDER_VERBOSITY)
    recent_block = ""
    if recent_premise is not None:
        recent_block = (
            "Most recent newly derived premise (distinguished):\n"
            f"- {recent_premise.id}: {recent_premise.clause}\n\n"
        )
    return (
        "Problem:\n"
        f"{problem.strip()}\n\n"
        "Current premises (by ID):\n"
        f"{premises_block}\n\n"
        f"{recent_block}"
        "Target answer head predicate:\n"
        f"{answer_spec.target}\n\n"
        "Decide whether termination criteria are met and, if so, propose the linking rule."
    )


async def check_termination_async(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    *,
    recent_premise: Optional[Premise],
    llm_exec: LLMExecutor,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> tuple[TerminationCheckerDecision, Dict[str, Any]]:
    """
    Async version of :func:`check_termination`.

    Always returns a second element: trace dict for the LLM call (prompts + raw/parsed answer).
    """
    user_content = _build_termination_checker_user_content(
        problem=problem,
        premises=premises,
        answer_spec=answer_spec,
        recent_premise=recent_premise,
    )
    system_prompt = system_prompt_override or TERMINATION_CHECK_SYSTEM_PROMPT
    parsed, raw = await llm_exec.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        return_raw=True,
    )
    trace = {
        "component": "final_termination_check",
        "system_prompt": system_prompt,
        "user_prompt": user_content,
        "raw_answer": raw,
        "parsed_answer": parsed,
    }
    return _termination_checker_decision_from_data(parsed), trace

