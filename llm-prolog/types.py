"""
Core symbolic types for the LLM‑Prolog pipeline.

This module defines a small Horn‑clause style language:
- Terms (variables vs constants)
- Predicates
- Facts and rules
- Premises (facts or rules with IDs and optional NL gloss)

It also provides very small helper parsers and formatters for a Prolog‑like
syntax sufficient for the project (no nested function symbols, no lists).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Term:
    """A term can be a variable or a constant."""

    name: str
    is_variable: bool = False

    @staticmethod
    def variable(name: str) -> "Term":
        return Term(name=name, is_variable=True)

    @staticmethod
    def constant(name: str) -> "Term":
        return Term(name=name, is_variable=False)


@dataclass(frozen=True)
class Predicate:
    name: str
    args: Tuple[Term, ...]

    def __str__(self) -> str:
        if not self.args:
            return self.name
        arg_str = ", ".join(t.name for t in self.args)
        return f"{self.name}({arg_str})"


@dataclass(frozen=True)
class Fact:
    predicate: Predicate

    def __str__(self) -> str:
        return f"{self.predicate}."


@dataclass(frozen=True)
class Rule:
    head: Predicate
    body: Tuple[Predicate, ...]

    def __str__(self) -> str:
        if not self.body:
            return f"{self.head}."
        body_str = ", ".join(str(p) for p in self.body)
        return f"{self.head} :- {body_str}."


Clause = Union[Fact, Rule]


@dataclass
class Premise:
    """A fact or rule with a unique ID and optional metadata."""

    id: int
    clause: Clause
    nl: Optional[str] = None
    source: Optional[str] = None


@dataclass(frozen=True)
class AnswerSpec:
    """Target predicate/head we hope to derive."""

    target: Predicate


@dataclass
class SelectorDecision:
    selected_premise_ids: List[int]
    proposed_new_premise: Optional[str]
    background_premises: List[str]
    is_answer_goal: bool
    should_stop: bool
    stop_reason: Optional[str] = None


@dataclass
class PipelineStep:
    step_index: int
    used_premise_ids: List[int]
    new_premise_id: Optional[int]
    decision: SelectorDecision
    success: bool
    note: Optional[str] = None


@dataclass
class PipelineResult:
    success: bool
    answer_premise: Optional[Premise]
    steps: List[PipelineStep]
    reason: Optional[str] = None


#
# Parsing helpers
#

def _parse_term(token: str) -> Term:
    token = token.strip()
    if not token:
        raise ValueError("Empty term token")
    # Simple heuristic: Prolog‑style variables start with uppercase.
    if token[0].isupper():
        return Term.variable(token)
    return Term.constant(token)


def parse_predicate(text: str) -> Predicate:
    """
    Parse a simple predicate of the form `name(arg1, arg2, ...)`.

    This intentionally supports only a limited subset: no nested function
    symbols or lists. Arguments are split on commas at the top level.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty predicate string")

    if "(" not in text:
        return Predicate(name=text, args=())

    name_part, rest = text.split("(", 1)
    name = name_part.strip()
    if not rest.endswith(")"):
        raise ValueError(f"Invalid predicate string (missing ')'): {text}")
    arg_str = rest[:-1]
    raw_args = [a.strip() for a in arg_str.split(",") if a.strip()]
    args = tuple(_parse_term(a) for a in raw_args)
    return Predicate(name=name, args=args)


def parse_fact_or_rule(text: str) -> Clause:
    """
    Parse a fact or rule from a Prolog‑like string.

    Examples:
      - 'has(apples, 3).'
      - 'sum(X, Y, Z) :- add(X, Y, Z).'
    """
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    if ":-" in text:
        head_str, body_str = text.split(":-", 1)
        head = parse_predicate(head_str.strip())
        body_atoms: List[Predicate] = []
        for atom_str in body_str.split(","):
            atom_str = atom_str.strip()
            if not atom_str:
                continue
            body_atoms.append(parse_predicate(atom_str))
        return Rule(head=head, body=tuple(body_atoms))
    else:
        pred = parse_predicate(text)
        return Fact(predicate=pred)


def format_clause(clause: Clause) -> str:
    """Render a Clause back into a canonical Prolog‑like string."""
    return str(clause)


def same_predicate_shape(a: Predicate, b: Predicate) -> bool:
    """Check if two predicates have the same name and arity."""
    return a.name == b.name and len(a.args) == len(b.args)

