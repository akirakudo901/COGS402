"""
Symbol‑to‑NL converter module.

Given symbolic premises and the original problem/query, this module asks an
LLM to paraphrase each clause into concise natural‑language explanations.
"""

from __future__ import annotations

from typing import Dict, List

from .llm_client.llm_client import LLMClient
from .symbolic.types import Premise, format_clause


SYSTEM_PROMPT = """
You are a logic tutor.

You receive:
- A word problem and its question.
- A list of Prolog‑style clauses with IDs.

For each clause, provide a short, precise natural‑language explanation of
what it states, suitable for a reasoning trace. Be concrete and focus on
quantities and relationships, not on Prolog syntax.

Output MUST be a single JSON object with:
- "explanations": an object mapping string IDs to explanation strings.
"""


def _render_premises(premises: List[Premise]) -> str:
    lines = []
    for p in premises:
        clause_str = format_clause(p.clause)
        lines.append(f"{p.id}: {clause_str}")
    return "\n".join(lines)


def symbols_to_nl(
    problem: str,
    query: str,
    premises: List[Premise],
    llm: LLMClient,
) -> Dict[int, str]:
    """
    Ask the LLM to paraphrase each symbolic premise into NL.
    """
    premises_block = _render_premises(premises)
    user_content = (
        "Problem:\n"
        f"{problem.strip()}\n\n"
        "Question:\n"
        f"{query.strip()}\n\n"
        "Clauses (by ID):\n"
        f"{premises_block}\n\n"
        "Provide explanations for each ID as described."
    )

    data = llm.generate_json(SYSTEM_PROMPT, user_content)
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

