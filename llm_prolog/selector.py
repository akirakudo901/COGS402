"""
Selector module.

Given the current set of premises and the target answer head, this module
asks an LLM to decide which premises to combine next, optionally propose
new background premises, and state whether we are aiming directly for the
answer goal.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm_client.llm_client import LLMClient
from .llm_executor import LLMExecutor
from .symbolic.types import AnswerSpec, Premise, SelectorDecision, render_premises
from .system_prompts import (
    SELECTOR_JSON_SECTION,
    SELECTOR_MAIN_SECTION,
    SYSTEM_PROMPT_INTRO,
    TERMINATION_CHECK_JSON_SECTION,
    TERMINATION_CHECK_SECTION,
    TERMINATION_CHECK_SYSTEM_PROMPT,
)


def _build_system_prompt(*, use_termination_checks: bool, system_prompt_override: str | None) -> str:
    intro = SYSTEM_PROMPT_INTRO
    termination_desc = TERMINATION_CHECK_SECTION
    selector_main = SELECTOR_MAIN_SECTION
    termination_json = TERMINATION_CHECK_JSON_SECTION
    selector_json = SELECTOR_JSON_SECTION

    if system_prompt_override:
        return system_prompt_override
    if use_termination_checks:
        return (intro + termination_desc + selector_main + termination_json + selector_json)
    return (intro + selector_main + selector_json)


@dataclass
class TerminationCheckerDecision:
    is_final_solution: bool
    solution_premise_id: Optional[int]
    answer_link_rule: Optional[str]


def select_next_step(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm: LLMClient,
    failed_steps_context: str = "",
    *,
    use_termination_checks: bool = True,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> SelectorDecision:
    """
    Ask the LLM which premises to combine next and what goal to pursue.
    """
    user_content = _build_user_content(problem, premises, answer_spec, failed_steps_context)
    system_prompt = _build_system_prompt(
        use_termination_checks=use_termination_checks,
        system_prompt_override=system_prompt_override,
    )
    data = llm.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _decision_from_data(data)


def select_next_step_candidates(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm: LLMClient,
    failed_steps_context: str = "",
    *,
    num_candidates: int = 1,
    use_termination_checks: bool = True,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> tuple[TerminationCheckerDecision, List[SelectorDecision]]:
    """
    Ask the LLM for N candidate next steps (ordered by likelihood).

    Returns:
    - A termination-only decision (termination fields filled; selection fields empty)
    - A list of candidate SelectorDecisions (length <= num_candidates)
    """
    if num_candidates <= 1:
        one = select_next_step(
            problem=problem,
            premises=premises,
            answer_spec=answer_spec,
            llm=llm,
            failed_steps_context=failed_steps_context,
            use_termination_checks=use_termination_checks,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt_override=system_prompt_override,
        )
        term_only = _termination_checker_decision_from_data(one.__dict__)
        return term_only, [one]

    user_content = _build_user_content(problem, premises, answer_spec, failed_steps_context)
    user_content = (
        user_content
        + "\n\n"
        + "Generate candidates:\n"
        + f"- Generate exactly {int(num_candidates)} candidates.\n"
        + "- Each candidate must be distinct from the others.\n"
    )
    system_prompt = _build_system_prompt(
        use_termination_checks=use_termination_checks,
        system_prompt_override=system_prompt_override,
    )
    system_prompt = system_prompt + "\n" + SELECTOR_MULTI_CANDIDATE_SECTION

    data = llm.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    term_only = _termination_checker_decision_from_data(data)
    candidates_raw = data.get("candidates", [])
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

    return term_only, candidates


def _build_user_content(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    failed_steps_context: str = "",
) -> str:
    premises_block = render_premises(premises, verbosity_level=2)
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


def _decision_from_data(data: Dict[str, Any]) -> SelectorDecision:
    selected_rule_id = data.get("selected_rule_id") or None
    selected_fact_ids = data.get("selected_fact_ids") or []
    
    try:
        selected_fact_ids = [selected_rule_id] + list(selected_fact_ids)
    except Exception:
        selected_fact_ids = []
    
    selected_ids_clean: List[int] = []
    for v in selected_fact_ids:
        try:
            selected_ids_clean.append(int(v))
        except (TypeError, ValueError):
            continue

    proposed = data.get("proposed_new_premise")
    if proposed is not None and not isinstance(proposed, str):
        proposed = None

    background = data.get("background_premises") or []
    if not isinstance(background, list):
        background = []
    background_clean = [str(x) for x in background if isinstance(x, (str, int, float))]

    is_answer_goal = bool(data.get("is_answer_goal", False))

    termination = _termination_checker_decision_from_data(data)

    # 'should_stop' and 'stop_reason' are filled by the pipeline once we checked the new premise
    # is a fact uniting with the goal predicate
    return SelectorDecision(
        selected_premise_ids=selected_ids_clean,
        proposed_new_premise=proposed,
        background_premises=background_clean,
        is_answer_goal=is_answer_goal,
        is_final_solution=termination.is_final_solution,
        solution_premise_id=termination.solution_premise_id,
        answer_link_rule=termination.answer_link_rule,
        should_stop=False,
        stop_reason=None,
    )


def _termination_checker_decision_from_data(data: Dict[str, Any]) -> TerminationCheckerDecision:
    """
    Extract termination-check fields from JSON.
    """
    raw_final_flag = data.get("is_final_solution", False)
    if isinstance(raw_final_flag, bool):
        is_final_solution = raw_final_flag
    elif isinstance(raw_final_flag, str):
        is_final_solution = raw_final_flag.strip().lower() in ("true", "1", "yes", "y")
    else:
        is_final_solution = bool(raw_final_flag)

    solution_id_raw = data.get("solution_premise_id", None)
    solution_premise_id: Optional[int] = None
    if solution_id_raw is not None:
        try:
            solution_premise_id = int(solution_id_raw)
        except (TypeError, ValueError):
            solution_premise_id = None

    rule_raw = data.get("answer_link_rule", None)
    answer_link_rule: Optional[str] = rule_raw if isinstance(rule_raw, str) else None
    if answer_link_rule is not None:
        answer_link_rule = answer_link_rule.strip()
        if not answer_link_rule:
            answer_link_rule = None

    if not is_final_solution:
        solution_premise_id = None
        answer_link_rule = None

    return TerminationCheckerDecision(
        is_final_solution=is_final_solution,
        solution_premise_id=solution_premise_id,
        answer_link_rule=answer_link_rule,
    )


async def select_next_step_async(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm_exec: LLMExecutor,
    failed_steps_context: str = "",
    *,
    use_termination_checks: bool = True,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> SelectorDecision:
    """
    Async version: ask the LLM which premises to combine next via LLMExecutor.
    """
    user_content = _build_user_content(problem, premises, answer_spec, failed_steps_context)
    system_prompt = _build_system_prompt(
        use_termination_checks=use_termination_checks,
        system_prompt_override=system_prompt_override,
    )
    data = await llm_exec.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _decision_from_data(data)


async def select_next_step_candidates_async(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm_exec: LLMExecutor,
    failed_steps_context: str = "",
    *,
    num_candidates: int = 1,
    use_termination_checks: bool = True,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> tuple[TerminationCheckerDecision, List[SelectorDecision]]:
    """
    Async version of `select_next_step_candidates`.
    """
    if num_candidates <= 1:
        one = await select_next_step_async(
            problem=problem,
            premises=premises,
            answer_spec=answer_spec,
            llm_exec=llm_exec,
            failed_steps_context=failed_steps_context,
            use_termination_checks=use_termination_checks,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt_override=system_prompt_override,
        )
        term_only = _termination_checker_decision_from_data(one.__dict__)
        return term_only, [one]

    user_content = _build_user_content(problem, premises, answer_spec, failed_steps_context)
    user_content = (
        user_content
        + "\n\n"
        + "Generate candidates:\n"
        + f"- Generate exactly {int(num_candidates)} candidates.\n"
        + "- Each candidate must be distinct from the others.\n"
    )
    system_prompt = _build_system_prompt(
        use_termination_checks=use_termination_checks,
        system_prompt_override=system_prompt_override,
    )
    system_prompt = system_prompt + "\n" + SELECTOR_MULTI_CANDIDATE_SECTION

    data = await llm_exec.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    term_only = _termination_checker_decision_from_data(data)
    candidates_raw = data.get("candidates", [])
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
    return term_only, candidates


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
) -> TerminationCheckerDecision:
    """
    Ask the LLM whether we have reached a final (ground-fact) solution and, if so,
    propose a linking rule that makes the answer head derivable.
    """
    user_content = _build_termination_checker_user_content(
        problem=problem,
        premises=premises,
        answer_spec=answer_spec,
        recent_premise=recent_premise,
    )
    system_prompt = system_prompt_override or TERMINATION_CHECK_SYSTEM_PROMPT
    data = llm.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _termination_checker_decision_from_data(data)


def _build_termination_checker_user_content(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    recent_premise: Optional[Premise],
) -> str:
    premises_block = render_premises(premises, verbosity_level=2)
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
) -> TerminationCheckerDecision:
    """
    Async version of :func:`check_termination`.
    """
    user_content = _build_termination_checker_user_content(
        problem=problem,
        premises=premises,
        answer_spec=answer_spec,
        recent_premise=recent_premise,
    )
    system_prompt = system_prompt_override or TERMINATION_CHECK_SYSTEM_PROMPT
    data = await llm_exec.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _termination_checker_decision_from_data(data)

