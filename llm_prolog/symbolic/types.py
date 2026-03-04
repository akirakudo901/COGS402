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

from dataclasses import dataclass, field
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

    def __repr__(self) -> str:
        return f"{self.name.capitalize() if self.is_variable 
                  else self.name.lower()}"


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

    def __repr__(self) -> str:
        parts = [f"\nclause='{self.clause}'"]
        if self.nl is not None:
            parts.append(f"\nnl={self.nl}")
        if self.source is not None:
            parts.append(f"\nsource={self.source}")
        inner = ", ".join(parts)
        return f"Premise_{self.id}({inner})"


@dataclass(frozen=True)
class AnswerSpec:
    """
    Target predicate/head we hope to derive.

    Invariant:
    - `target` contains exactly one *logical* variable name (which may appear
      in one or more argument positions).
    - All other arguments are constants.

    This encodes that the final answer is a single value that will unify with
    this distinguished variable, while other arguments can pin down context
    via constants.
    """

    target: Predicate
    # Name of the single logical variable that the final answer will bind to.
    variable_name: str = field(init=False)

    def __post_init__(self) -> None:
        # Collect distinct logical variable names across all arguments.
        var_names = {t.name for t in self.target.args if t.is_variable}
        if not var_names:
            raise ValueError(
                "AnswerSpec.target must contain exactly one logical variable, "
                "but found none."
            )
        if len(var_names) > 1:
            raise ValueError(
                "AnswerSpec.target must contain exactly one logical variable, "
                f"but found multiple: {sorted(var_names)}"
            )
        # Freeze the single distinguished variable name.
        object.__setattr__(self, "variable_name", next(iter(var_names)))

    @property
    def variable(self) -> Term:
        """Return the distinguished answer variable as a Term."""
        return Term.variable(self.variable_name)

    def __repr__(self) -> str:
        return f"AnswerSpec(target={self.target}, variable={self.variable_name})"


@dataclass
class SelectorDecision:
    selected_premise_ids: List[int]
    proposed_new_premise: Optional[str]
    background_premises: List[str]
    is_answer_goal: bool
    should_stop: bool
    stop_reason: Optional[str] = None

    def __repr__(self) -> str:
        return (
            "SelectorDecision("
            f"selected_premise_ids={self.selected_premise_ids}, "
            f"proposed_new_premise={self.proposed_new_premise}, "
            f"background_premises={self.background_premises}, "
            f"is_answer_goal={self.is_answer_goal}, "
            f"should_stop={self.should_stop}, "
            f"stop_reason={self.stop_reason})"
        )


@dataclass
class PipelineStep:
    step_index: int
    used_premise_ids: List[int]
    new_premise: Optional[Premise]
    decision: SelectorDecision
    success: bool
    note: Optional[str] = None

    def __repr__(self) -> str:
        parts = [
            f"step_index={self.step_index}",
            f"used_premise_ids={self.used_premise_ids}",
            f"new_premise={self.new_premise}",
            f"decision={self.decision}",
            f"success={self.success}",
        ]
        if self.note is not None:
            parts.append(f"note={self.note}")
        inner = ", ".join(parts)
        return f"PipelineStep({inner})"


@dataclass
class PipelineResult:
    success: bool
    answer_premise: Optional[Premise]
    steps: List[PipelineStep]
    reason: Optional[str] = None

    def __repr__(self) -> str:
        parts = [
            f"success={self.success}",
            f"answer_premise={self.answer_premise}",
            f"steps={self.steps}",
        ]
        if self.reason is not None:
            parts.append(f"reason={self.reason}")
        inner = ", ".join(parts)
        return f"PipelineResult({inner})"


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

    # Prolog arithmetic evaluation: `X is Expr.`
    # We represent it internally as a normal predicate `mathIs(LHS, RHS_EXPR)`.
    # RHS_EXPR is stored as a constant Term containing the expression string.
    if " is " in text and "(" not in text:
        lhs_str, rhs_str = text.split(" is ", 1)
        lhs_str = lhs_str.strip()
        rhs_str = rhs_str.strip()
        if not lhs_str or not rhs_str:
            raise ValueError(f"Invalid is/2 predicate string: {text}")
        return Predicate(
            name="mathIs",
            args=(
                _parse_term(lhs_str),
                Term.constant(rhs_str),
            ),
        )

    if "(" not in text:
        return Predicate(name=text, args=())

    name_part, rest = text.split("(", 1)
    name = name_part.strip()
    if not rest.endswith(")"):
        raise ValueError(f"Invalid predicate string (missing ')'): {text}")
    arg_str = rest[:-1]
    raw_args = [a.strip() for a in arg_str.split(",") if a.strip()]
    args = tuple(_parse_term(a) for a in raw_args)
    # Also accept functional form `is(LHS, RHS_EXPR)` and normalize it.
    if name == "is":
        if len(args) != 2:
            raise ValueError(f"Invalid is/2 predicate arity: {text}")
        lhs, rhs = args
        if rhs.is_variable:
            # `is/2` evaluates the RHS expression; a bare variable RHS is not
            # a supported expression in this project representation.
            raise ValueError(f"Invalid is/2 RHS expression: {text}")
        return Predicate(name="mathIs", args=(lhs, rhs))

    return Predicate(name=name, args=args)


def _split_predicate_atoms(body_str: str) -> List[str]:
    """
    Split a rule body string into predicate substrings, ignoring commas that
    occur inside parentheses.
    """
    parts: List[str] = []
    current: List[str] = []
    depth = 0

    for ch in body_str:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            # We are lenient here; if depth would go negative we just clamp at 0.
            if depth > 0: depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            segment = "".join(current).strip()
            if segment:
                parts.append(segment)
            current = []
        else:
            current.append(ch)

    # Add the final segment if any.
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    return parts


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
        for atom_str in _split_predicate_atoms(body_str):
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

def render_premises(premises: List[Premise]) -> str:
    """Render a list of Premise in order, one by line."""
    lines = []
    sorted_premises = sorted(premises, key=lambda x: x.id)
    for p in sorted_premises:
        clause_str = format_clause(p.clause)
        nl = f"  # {p.nl}" if p.nl else ""
        lines.append(f"{p.id}: {clause_str}{nl}")
    return "\n".join(lines)