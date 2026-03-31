"""
Symbol‑to‑NL converter module.

Given symbolic premises and the original problem text, this module asks an
LLM to paraphrase each clause into concise natural‑language explanations.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .llm_client.llm_client import LLMClient
from .llm_executor import LLMExecutor
from .symbolic.types import Premise, format_clause
from .system_prompts import SYMBOL_TO_NL_SYSTEM_PROMPT



def _render_premises(premises: List[Premise]) -> str:
    lines = []
    for p in premises:
        clause_str = format_clause(p.clause)
        lines.append(f"{p.id}: {clause_str}")
    return "\n".join(lines)


def symbols_to_nl(
    problem: str,
    premises: List[Premise],
    llm: LLMClient,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> Dict[int, str]:
    """
    Ask the LLM to paraphrase each symbolic premise into NL.
    """
    user_content = _build_user_content(problem, premises)
    system_prompt = system_prompt_override or SYMBOL_TO_NL_SYSTEM_PROMPT
    data = llm.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _explanations_from_data(data)


def _build_user_content(problem: str, premises: List[Premise]) -> str:
    premises_block = _render_premises(premises)
    return (
        "Problem:\n"
        f"{problem.strip()}\n\n"
        "Clauses (by ID):\n"
        f"{premises_block}\n\n"
        "Provide explanations for each ID as described."
    )


def _explanations_from_data(data: Dict[str, Any]) -> Dict[int, str]:
    raw_explanations = data.get("explanations", {}) or {}
    result: Dict[int, str] = {}
    if isinstance(raw_explanations, dict):
        for k, v in raw_explanations.items():
            try:
                pid = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, str):
                result[pid] = v
    return result


async def symbols_to_nl_async(
    problem: str,
    premises: List[Premise],
    llm_exec: LLMExecutor,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt_override: str | None = None,
) -> Dict[int, str]:
    """
    Async version: ask the LLM to paraphrase each symbolic premise into NL via LLMExecutor.
    """
    user_content = _build_user_content(problem, premises)
    system_prompt = system_prompt_override or SYMBOL_TO_NL_SYSTEM_PROMPT
    data = await llm_exec.generate_json(
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _explanations_from_data(data)

