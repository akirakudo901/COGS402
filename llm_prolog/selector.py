"""
Selector module.

Given the current set of premises and the target answer head, this module
asks an LLM to decide which premises to combine next, optionally propose
new background premises, and state whether we are aiming directly for the
answer goal.
"""

from __future__ import annotations

from typing import List, Optional

from .llm_client.llm_client import LLMClient
from .symbolic.types import AnswerSpec, Premise, SelectorDecision, format_clause


SYSTEM_PROMPT = """
You are a symbolic reasoning planner working over a Prolog‑style theory.

You are given:
- A general natural language problem and its question.
- A list of existing premises (facts and rules) with IDs.
- A target answer head predicate we ultimately want to prove.
‑ A (possibly empty) list of premise‑ID sets that have already been combined in previous steps.

Your task for each step:
- Choose ONE rule (with a head and body) and SOME facts (with a head only) by their premise IDs 
  that should be combined for the next inference step.
- Optionally propose new background premises (facts or rules) if the
  current theory is insufficient.
- State what new premise you intend for the inference engine to derive.
- Indicate whether this new premise is directly the answer head goal.

Output MUST be a single JSON object with the fields:
- "selected_premise_ids": list of integer IDs.
- "proposed_new_premise": string or null (a Prolog‑style clause WITHOUT
  needing to be valid; this is an intention).
- "background_premises": list of strings, each a fact or rule ending
  with a period.
- "is_answer_goal": boolean.
You MUST NOT choose a set of "selected_premise_ids" that is exactly equal
to any of the previously combined premise‑ID sets.
"""


def _render_premises(premises: List[Premise]) -> str:
    lines = []
    for p in premises:
        clause_str = format_clause(p.clause)
        nl = f"  # {p.nl}" if p.nl else ""
        lines.append(f"{p.id}: {clause_str}{nl}")
    return "\n".join(lines)


def select_next_step(
    problem: str,
    query: str,
    premises: List[Premise],
    answer_spec: AnswerSpec,
    llm: LLMClient,
    previous_premise_sets: Optional[List[List[int]]] = None,
) -> SelectorDecision:
    """
    Ask the LLM which premises to combine next and what goal to pursue.
    """
    premises_block = _render_premises(premises)
    previous_sets_block = ""
    if previous_premise_sets:
        formatted_sets = ", ".join(
            "{" + ", ".join(str(pid) for pid in sorted(s)) + "}"
            for s in previous_premise_sets
        )
        previous_sets_block = (
            "Previously combined premise ID sets (do NOT choose any of these exact combinations again):\n"
            f"{formatted_sets}\n\n"
        )
    user_content = (
        "Problem:\n"
        f"{problem.strip()}\n\n"
        "Question:\n"
        f"{query.strip()}\n\n"
        "Current premises (by ID):\n"
        f"{premises_block}\n\n"
        f"{previous_sets_block}"
        "Answer head predicate:\n"
        f"{answer_spec.target}\n\n"
        "Decide the next reasoning step following the instructions."
    )

    data = llm.generate_json(SYSTEM_PROMPT, user_content)

    selected_ids = data.get("selected_premise_ids") or []
    if not isinstance(selected_ids, list):
        selected_ids = []
    selected_ids_clean: List[int] = []
    for v in selected_ids:
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

