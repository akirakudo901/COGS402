"""
NL‑to‑symbol converter module.

This module is responsible for taking a natural‑language problem and query
and producing:
- an initial set of symbolic premises (facts and rules), and
- an AnswerSpec describing the target head predicate we hope to prove.
"""

from __future__ import annotations

from typing import List, Tuple

from .llm_client.llm_client import LLMClient
from .symbolic.types import (
    AnswerSpec,
    Premise,
    Clause,
    parse_fact_or_rule,
    parse_predicate,
)


SYSTEM_PROMPT = """
You are a reasoning assistant that converts general natural language problems into a small
Prolog‑style Horn clause theory.

Language:
- Facts: predicate(constant1, constant2, ...).
- Rules: head(X, Y) :- body1(X), body2(X, Y).
- Variables start with an uppercase letter.
- Constants are lowercase identifiers or numbers.

Goal:
- Extract base facts from the problem statement.
- Introduce simple rules that connect those facts to the question or target query.
- Define a single answer head predicate with exactly one variable representing 
  the final answer, such as answer(Value) or eq(Lhs, rhs).

Output format:
- You MUST return a single JSON object with the keys:
  - "facts": list of strings, each a fact ending with a period.
  - "rules": list of strings, each a rule ending with a period.
  - "answer_head": a single predicate string WITHOUT a trailing period.
  - "explanations": list of strings of same length as facts+rules, giving
    a short natural‑language gloss for each clause.
"""


def _build_user_prompt(problem: str, query: str) -> str:
    return (
        "Problem:\n"
        f"{problem.strip()}\n\n"
        "Question:\n"
        f"{query.strip()}\n\n"
        "Instructions:\n"
        "- Identify the important information and relationships.\n"
        "- Express them as Prolog‑style facts and rules.\n"
        "- Choose an answer_head predicate with one variable encoding the final outcome "
        "needed to answer the question.\n"
        "- Keep the theory small and focused on what is needed."
    )


def convert_problem_to_symbols(
    problem: str,
    query: str,
    llm: LLMClient,
) -> Tuple[List[Premise], AnswerSpec]:
    """
    Convert a problem‑query pair into initial symbolic premises and an answer spec.
    """
    user_prompt = _build_user_prompt(problem, query)
    data = llm.generate_json(SYSTEM_PROMPT, user_prompt)

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

