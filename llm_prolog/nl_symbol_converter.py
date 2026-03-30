"""
NL‑to‑symbol converter module.

This module is responsible for taking a natural‑language problem and producing:
- an initial set of symbolic premises (facts and rules), and
- an AnswerSpec describing the target head predicate we hope to prove.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .llm_client.llm_client import LLMClient
from .llm_executor import LLMExecutor
from .symbolic.types import (
    AnswerSpec,
    Premise,
    Clause,
    parse_fact_or_rule,
    parse_predicate,
)
from .system_prompts import NL_TO_SYMBOL_SYSTEM_PROMPT


def _build_user_prompt(problem: str) -> str:
    return (
        "Problem:\n"
        f"{problem.strip()}\n\n"
        "Instructions:\n"
        "- Identify the important information and relationships.\n"
        "- Express them as Prolog‑style facts and rules.\n"
        "- Choose an answer_head predicate with one variable encoding the final outcome "
        "needed to answer the problem.\n"
        "- Keep the theory small and focused on what is needed."
    )


def convert_problem_to_symbols(
    problem: str,
    llm: LLMClient,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> Tuple[List[Premise], AnswerSpec]:
    """
    Convert a natural‑language problem into initial symbolic premises and an answer spec.
    """
    user_prompt = _build_user_prompt(problem)
    system_prompt = system_prompt_override or NL_TO_SYMBOL_SYSTEM_PROMPT
    data = llm.generate_json(
        system_prompt,
        user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _symbols_from_data(data)


def _symbols_from_data(data: Dict[str, Any]) -> Tuple[List[Premise], AnswerSpec]:
    raw_facts = data.get("facts", []) or []
    raw_rules = data.get("rules", []) or []
    answer_head_str = data.get("answer_head")
    explanations = data.get("explanations", []) or []

    clauses: List[Clause] = []
    for s in raw_facts + raw_rules:
        if not isinstance(s, str):
            continue
        clauses.append(parse_fact_or_rule(s))

    premises: List[Premise] = []
    for idx, clause in enumerate(clauses, start=1):
        gloss = explanations[idx - 1] if idx - 1 < len(explanations) else None
        premises.append(
            Premise(
                id=idx,
                clause=clause,
                nl=gloss,
                source="nl_symbol_converter",
            )
        )

    if not isinstance(answer_head_str, str):
        raise ValueError("NL‑Symbol converter did not return a valid 'answer_head' string.")

    target_pred = parse_predicate(answer_head_str)
    # The AnswerSpec enforces that the target predicate contains exactly one
    # logical variable (the final answer) and any number of constants.
    answer_spec = AnswerSpec(target=target_pred)
    return premises, answer_spec


async def convert_problem_to_symbols_async(
    problem: str,
    llm_exec: LLMExecutor,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> Tuple[List[Premise], AnswerSpec]:
    """
    Async version: convert NL problem to symbolic premises and answer spec via LLMExecutor.
    """
    user_prompt = _build_user_prompt(problem)
    system_prompt = system_prompt_override or NL_TO_SYMBOL_SYSTEM_PROMPT
    data = await llm_exec.generate_json(
        system_prompt,
        user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _symbols_from_data(data)

