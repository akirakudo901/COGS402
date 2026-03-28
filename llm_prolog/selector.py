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


SYSTEM_PROMPT_INTRO = """
You are a symbolic reasoning planner working over a Prolog‑style theory.

You are given:
- A natural language problem with a question or goal.
- A list of existing premises (facts and rules) with IDs.
- A target answer head predicate we ultimately want to prove.
- A (possibly empty) list of failed past reasoning steps grouped by reason.

Your task for each step:
"""

TERMINATION_CHECK_SECTION = """
- First, perform a termination check using the provided existing premises (the current
  list of facts and rules).
- Decide whether the system already has a "final solution" ground fact available.
  A "final solution" captures the solution to the natural language problem + question / goal provided,
  even if it is NOT the answer head predicate itself.
  A "final solution" MUST a ground fact: a fact whose predicate arguments contain no variables.
- If such a ground fact exists, you MUST also propose a single Prolog rule that links the chosen
  ground-fact predicate to the answer head predicate.
  The linking rule MUST have the answer head predicate as its head and the chosen ground fact's
  predicate as the (single) body atom, with the answer variable passed through.
More concretely, if a final solution exists:
  - set "is_final_solution" to true
  - set "solution_premise_id" to the ID of the chosen "final solution" ground fact premise
  - set "answer_link_rule" to a single Prolog rule that links the chosen ground fact to the
    answer head predicate (exactly one body atom):
      <answer_head>(<AnswerVar>) :- <solution_predicate>(<AnswerVar>, <OtherConstantsIfAny>)
    e.g. if the question asks for the total for marc, and answer_head is "Answer" and solution premise is "Total(32, marc)", 
    then "Answer(X) :- Total(X, marc)."
    Use the answer head predicate exactly as provided, including its distinguished answer variable.
  - The answer head in the linking rule MUST use the same predicate name and constants as the provided
    answer head predicate, except one variable for the value in question we want to extract.
  - Use exactly one body atom in the linking rule.
  - Do not add a trailing period to the rule string.

- If termination criteria are NOT met, set "is_final_solution" to false, and set the other
  termination fields to null.

Once done with termination check:
"""

SELECTOR_MAIN_SECTION = """
- State what new premise you intend for the inference engine to derive. This premise 
  must be new among existing premises.
- Indicate whether this new premise is directly the answer head goal.
- Optionally propose new background premises (facts or rules) if the
  current theory is insufficient.
- Choose exactly ONE **consumer** rule (a clause with a head and body) by `selected_rule_id`.
- Choose ONE OR MORE **producers** by listing their premise IDs in `selected_fact_ids` in the **order**
  they should be applied. Each producer may be a **fact** (head only) or a **rule** (head and body).
  For each goal position in the consumer, in order, the inference engine consumes the **first** producer 
  in the remaining producers' 'pool' that unifies with that goal. Producers used for a goal are removed from the pool.
- You MUST NOT choose an order of premises that has been previously combined to produce an existing premise.
- Use the failed-step history to avoid repeating choices that failed for known reasons.

Output MUST be a single JSON object with the fields:
"""

SELECTOR_MULTI_CANDIDATE_SECTION = """
- Generate multiple distinct candidate next steps in decreasing order of likelihood.
- Each candidate represents ONE planned inference attempt and must include:
  - "proposed_new_premise": a premise you intend to derive (string; must be new among existing premises)
  - "is_answer_goal": boolean (true if this premise is directly the answer head goal)
  - "background_premises": optional list of facts/rules (strings) ending with a period
  - "selected_rule_id": integer ID of exactly ONE consumer rule
  - "selected_fact_ids": ordered list of integer IDs of ONE OR MORE producers
- Each candidate MUST be distinct from all others (not the same exact combination of the above fields).

If generating multiple candidates, output MUST be a single JSON object with:
- "candidates": a list of candidate objects, ordered by likelihood (best first).
"""

TERMINATION_CHECK_JSON_SECTION = """
For termination check:
- "is_final_solution": boolean.
- "solution_premise_id": integer ID of the chosen "final solution" ground fact premise, or null.
- "answer_link_rule": string Prolog rule (no trailing period) linking solution to answer head,
  or null.

For selection of next premise:
"""

SELECTOR_JSON_SECTION = """
- "proposed_new_premise": string or null (a Prolog‑style clause WITHOUT
  needing to be valid; this is an intention. It must be new from existing premises).
- "is_new_proposal": boolean.
- "is_answer_goal": boolean.
- "background_premises": list of strings, each a fact or rule ending
  with a period.
- "selected_rule_id": integer ID of the **consumer** rule premise for this step.
- "selected_fact_ids": ordered list of integer IDs of **producer** premises (each may be a fact or a
  rule). Order matters: the engine matches slot 0 against producers in this order, then slot 1 against
  what remains, and so on.
"""


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


# NOT CURRENTLY IN USE; Possibly useful in the future if we isolte the selector from termination checker
TERMINATION_CHECK_SYSTEM_PROMPT = """
You are a termination checker for a symbolic (Prolog-like) reasoning system.

Given:
- A natural language problem.
- A set of current Prolog-style premises (facts and rules) with IDs.
- The target answer head predicate we ultimately want to prove.
- The most recently derived premise (at the current step) as a distinguished item.

Your job:
- Decide whether there is a "final solution" ground fact available at the most recent step.
  A "final solution" MUST be a ground fact: a fact whose predicate arguments contain no variables.
  The ground fact should capture the constant(s) needed for the final answer, even if it is NOT
  the answer head predicate itself.
- If such a ground fact exists, you MUST also propose a single Prolog rule that links the chosen
  ground-fact predicate to the answer head predicate.
  The linking rule MUST have the answer head predicate as its head and the chosen ground fact's
  predicate as the (single) body atom, with the answer variable passed through.

Output MUST be a single JSON object with exactly these fields:
- "is_final_solution": boolean
- "solution_premise_id": integer ID of the chosen ground fact, or null
- "answer_link_rule": string of the form "<answer_head>(<AnswerVar>) :- <solution_predicate>(<AnswerVar>, <OtherConstantsIfAny>)", or null
  e.g. if the question asks for the total for marc, and answer_head is "Answer" and solution premise is "Total(32, marc)", 
  then "Answer(X) :- Total(X, marc)."

Rules:
- The answer head in the linking rule MUST use the same predicate name and constants as the provided
  answer head predicate, except one variable for the value in question we want to extract.
- Use exactly one body atom in the linking rule.
- Do not add a trailing period to the rule string.
"""


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

