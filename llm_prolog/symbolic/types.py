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
from typing import Dict, List, Optional, Tuple, Union


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
        return f"Term(name={self.name!r}, is_variable={self.is_variable!r})"

    def __str__(self) -> str:
        # Prolog-style: variables are uppercase, constants are lowercase.
        return self.name.capitalize() if self.is_variable else self.name.lower()


@dataclass(frozen=True)
class Predicate:
    name: str
    args: Tuple[Term, ...]

    def __repr__(self) -> str:
        return f"Predicate(name={self.name!r}, args={self.args!r})"

    def __str__(self) -> str:
        if not self.args:
            return self.name
        arg_str = ", ".join(str(t) for t in self.args)
        return f"{self.name}({arg_str})"


@dataclass(frozen=True)
class Fact:
    predicate: Predicate

    def __repr__(self) -> str:
        return f"Fact(predicate={self.predicate!r})"

    def __str__(self) -> str:
        return f"{self.predicate}."


@dataclass(frozen=True)
class Rule:
    head: Predicate
    body: Tuple[Predicate, ...]

    def __repr__(self) -> str:
        return f"Rule(head={self.head!r}, body={self.body!r})"

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
    parent_ids: Optional[List[int]] = None

    def __repr__(self) -> str:
        return (
            "Premise("
            f"id={self.id!r}, "
            f"clause={self.clause!r}, "
            f"nl={self.nl!r}, "
            f"source={self.source!r}, "
            f"parent_ids={self.parent_ids!r})"
        )

    def __str__(self) -> str:
        return self.str_verbose(level=3)
    
    def str_verbose(self, *, level : int) -> str:
        """
        Produces string with different verbosity.
        Level 0: clause
        Level 1: clause + natural language description
        Level 2: clause + nl desc + parent ids
        Level 3: clause + nl desc + p ids + source
        """
        if level not in range(4):
            raise Exception("str_verbose accepts verbosity levels from 0 to 3 only.")
        clause_str = format_clause(self.clause)
        lines = [f"{self.id}: {clause_str}"]
        if self.nl and level >= 1:
            lines[0] += f" # {self.nl}"
        if self.parent_ids is not None and len(self.parent_ids) > 0 and level >= 2:
            lines.append(f"  (from premises {', '.join(str(pid) for pid in self.parent_ids)})")
        if self.source and level >= 3:
            lines.append(f"  Source: {self.source}")
        return "\n".join(lines)


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
        return (
            "AnswerSpec("
            f"target={self.target!r}, "
            f"variable_name={self.variable_name!r})"
        )

    def __str__(self) -> str:
        return (
            f"'{self.variable_name}' in '{self.target}'"
        )


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
    
    def __str__(self) -> str:
        lines = ["Selector decision:"]

        if self.background_premises:
            lines.append("  Proposed new background premises:")
            for premise in self.background_premises:
                lines.append(f"    * {premise}")
        
        if self.proposed_new_premise:
            spec = 'goal' if self.is_answer_goal else 'non-goal'
            lines.append(f"  Proposed to combine IDs {self.selected_premise_ids} to deduce a {spec} premise:")
            lines.append(f"    {self.proposed_new_premise}")
        else:
            lines.append("  Proposed no new premise.")
        
        if self.should_stop and self.stop_reason:
            lines.append(f"  Decided we must stop because: {self.stop_reason}")
        return "\n".join(lines)


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
            f"step_index={self.step_index!r}",
            f"used_premise_ids={self.used_premise_ids!r}",
            f"new_premise={self.new_premise!r}",
            f"decision={self.decision!r}",
            f"success={self.success!r}",
        ]
        if self.note is not None:
            parts.append(f"note={self.note!r}")
        inner = ", ".join(parts)
        return f"PipelineStep({inner})"
    
    def __str__(self) -> str:
        lines = []
        lines.append(f"Step {self.step_index} ({'succeeded' if self.success else 'failed'}):")
        lines.append(f"."*20)
        lines.append(f"{self.decision}")
        lines.append(f"."*20)
        if self.new_premise is not None:
            lines.append(f"  Used premise IDs {self.used_premise_ids} to deduce the new premise:")
            lines.append(f"    {self.new_premise}")
        else:
            lines.append(f"  Used premise IDs {self.used_premise_ids} to deduce NO new premise.")

        if self.note is not None:
            lines.append(f"  Note: {self.note}")
        return "\n".join(lines)


@dataclass
class PipelineResult:
    success: bool
    answer_premise: Optional[Premise]
    steps: List[PipelineStep]
    answer_spec: AnswerSpec
    # All premises available at the end of the pipeline run, including
    # originals, selector‑provided background premises, and inferred ones.
    final_premises: List[Premise]
    reason: Optional[str] = None

    def __repr__(self) -> str:
        parts = [
            f"success={self.success!r}",
            f"answer_premise={self.answer_premise!r}",
            f"steps={self.steps!r}",
            f"answer_spec={self.answer_spec!r}",
            f"final_premises={self.final_premises!r}",
        ]
        if self.reason is not None:
            parts.append(f"reason={self.reason!r}")
        inner = ", ".join(parts)
        return f"PipelineResult({inner})"
    
    def __str__(self) -> str:
        status = "succeeded" if self.success else "failed"
        lines = [f"Pipeline {status}."]
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        
        lines.append("Premises:")
        lines.append(render_premises(self.final_premises))

        if self.answer_premise:
            lines.append(f"Answer premise: {self.answer_premise}")
        else:
            lines.append("Answer premise: None")

        lines.append(f"Answer spec: {self.answer_spec}")

        # Show obtained steps
        lines.append("="*20)
        lines.append("Pipeline steps:")
        for s in self.steps:
            lines.append("-"*20)
            lines.append(f"{s}")
        return "\n".join(lines)

def extract_premise_derivation_dict(
    result: PipelineResult,
) -> Dict[int, Tuple[List[int], int]]:
    """
    Build a dictionary summarizing which premises were used to derive new ones.

    The returned mapping has:
    - key: step_index for each step that produced a new premise
    - value: (used_premise_ids, new_premise_id)
    """
    derivations: Dict[int, Tuple[List[int], int]] = {}
    for step in result.steps:
        if step.new_premise is None:
            continue
        derivations[step.step_index] = (sorted(list(step.used_premise_ids)), 
                                        step.new_premise.id)
    return derivations


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
    if " is " in text:
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

def render_premises(premises: List[Premise], verbosity_level : int=1) -> str:
    """
    Render a list of Premise in order, one by line. 
    Can adjust verbosity_level (see Premise.str_verbose)
    """
    lines = []
    sorted_premises = sorted(premises, key=lambda x: x.id)
    for p in sorted_premises:
        lines.append(p.str_verbose(level=verbosity_level))
    return "\n".join(lines)