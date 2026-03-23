"""
Selector module.

Given the current set of premises and the target answer head, this module
asks an LLM to decide which premises to combine next, optionally propose
new background premises, and state whether we are aiming directly for the
answer goal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .llm_client.llm_client import LLMClient
from .llm_executor import LLMExecutor
from .symbolic.types import AnswerSpec, Premise, SelectorDecision, render_premises


SYSTEM_PROMPT = """
You are a symbolic reasoning planner working over a Prolog‑style theory.

You are given:
- A natural language problem with a question or goal.
- A list of existing premises (facts and rules) with IDs.
- A target answer head predicate we ultimately want to prove.
- A (possibly empty) list of failed past reasoning steps grouped by reason.

Your task for each step:
- State what new premise you intend for the inference engine to derive. This premise 
  must be new among existing premises.
- Indicate whether this new premise is directly the answer head goal.
- Optionally propose new background premises (facts or rules) if the
  current theory is insufficient.
- Choose ONE rule (with a head and body) and ONE OR MORE facts (with a head only) by their premise IDs 
  that should be combined to derive the new premise via the inference step.
- You MUST NOT choose a set of premises that has been previously combined to produce an existing premise.
- Use the failed-step history to avoid repeating choices that failed for known reasons.

Output MUST be a single JSON object with the fields:
- "proposed_new_premise": string or null (a Prolog‑style clause WITHOUT
  needing to be valid; this is an intention. It must be new from existing premises).
- "is_new_proposal": boolean.
- "is_answer_goal": boolean.
- "background_premises": list of strings, each a fact or rule ending
  with a period.
- "selected_rule_id": an integer ID indicating the rule premise selected in this step.
- "selected_fact_ids": list of integer IDs indicating fact premises selected in this step.
"""


def select_next_step(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm: LLMClient,
    failed_steps_context: str = "",
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> SelectorDecision:
    """
    Ask the LLM which premises to combine next and what goal to pursue.
    """
    user_content = _build_user_content(problem, premises, answer_spec, failed_steps_context)
    system_prompt = system_prompt_override or SYSTEM_PROMPT
    data = llm.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _decision_from_data(data)


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

    # 'should_stop' and 'stop_reason' are filled by the pipeline once we checked the new premise
    # is a fact uniting with the goal predicate
    return SelectorDecision(
        selected_premise_ids=selected_ids_clean,
        proposed_new_premise=proposed,
        background_premises=background_clean,
        is_answer_goal=is_answer_goal,
        should_stop=False,
        stop_reason=None,
    )


async def select_next_step_async(
    problem: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm_exec: LLMExecutor,
    failed_steps_context: str = "",
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> SelectorDecision:
    """
    Async version: ask the LLM which premises to combine next via LLMExecutor.
    """
    user_content = _build_user_content(problem, premises, answer_spec, failed_steps_context)
    system_prompt = system_prompt_override or SYSTEM_PROMPT
    data = await llm_exec.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _decision_from_data(data)

