"""
Symbolic inference engine for the LLM‑Prolog pipeline.

This module implements a very small Horn‑clause engine:
- Unification between terms and predicates.
- Deriving a new clause (Fact or Rule) from one rule and one or more facts.
"""

from __future__ import annotations

import ast
import os
import re
import threading
from typing import Dict, List, Optional, Tuple

from .types import Clause, Fact, Predicate, Premise, Rule, Term, _parse_term

try:
    from pyswip import Prolog  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - import error path is tested via policy handling
    Prolog = None  # type: ignore[assignment]


Substitution = Dict[str, Term]

_PROLOG_VAR_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")


def _inference_policy() -> str:
    """
    Backend policy:
    - strict (default): require pySwip + SWI-Prolog and raise if unavailable.
    - fallback: silently fallback to Python implementation when unavailable.
    """
    raw = os.getenv("LLM_PROLOG_INFERENCE_POLICY", "strict").strip().lower()
    return "fallback" if raw == "fallback" else "strict"


class _PrologBackend:
    def __init__(self) -> None:
        if Prolog is None:
            raise RuntimeError(
                "pySwip is not available. Install `pyswip` and SWI-Prolog, "
                "or set LLM_PROLOG_INFERENCE_POLICY=fallback."
            )
        self._lock = threading.Lock()
        self._prolog = Prolog()

    def unify_predicates(
        self, a: Predicate, b: Predicate, subst: Optional[Substitution] = None
    ) -> Optional[Substitution]:
        def _value_to_term(value: object) -> Optional[Term]:
            if value is None:
                return None
            text = str(value)
            if not text:
                return None
            return _parse_term(text)
        
        if a.name != b.name or len(a.args) != len(b.args):
            return None

        initial = {} if subst is None else dict(subst)
        pred_a = a.to_prolog_text()
        pred_b = b.to_prolog_text()
        goals = []
        for var_name, bound_term in initial.items():
            goals.append(f"{var_name} = {bound_term.to_prolog_text()}")
        goals.append(f"{pred_a} = {pred_b}")
        query = ", ".join(goals)

        try:
            with self._lock:
                solutions = list(self._prolog.query(query, maxresult=1))
        except Exception:
            return None
        if not solutions:
            return None

        resolved = dict(initial)
        solution = solutions[0]
        for var_name, value in solution.items():
            term = _value_to_term(value)
            if term is not None:
                resolved[str(var_name)] = term
        return resolved


_PROLOG_BACKEND: Optional[_PrologBackend] = None
_PROLOG_BACKEND_ERROR: Optional[Exception] = None


def _get_prolog_backend() -> _PrologBackend:
    global _PROLOG_BACKEND, _PROLOG_BACKEND_ERROR
    if _PROLOG_BACKEND is not None:
        return _PROLOG_BACKEND
    if _PROLOG_BACKEND_ERROR is not None:
        raise _PROLOG_BACKEND_ERROR
    try:
        _PROLOG_BACKEND = _PrologBackend()
        return _PROLOG_BACKEND
    except Exception as exc:
        _PROLOG_BACKEND_ERROR = exc
        raise


def _prefer_python_fallback_on_backend_error() -> bool:
    return _inference_policy() == "fallback"


def _py_unify_terms(a: Term, b: Term, subst: Substitution) -> Optional[Substitution]:
    """
    Unify two terms under an existing substitution, returning an extended
    substitution or None if unification fails.
    """
    # Apply existing substitution.
    if a.is_variable and a.name in subst:
        a = subst[a.name]
    if b.is_variable and b.name in subst:
        b = subst[b.name]

    # Identical constants.
    if not a.is_variable and not b.is_variable:
        return subst if a.name == b.name else None

    # Variable cases: bind variable to the other term.
    if a.is_variable and not b.is_variable:
        subst[a.name] = b
        return subst
    if b.is_variable and not a.is_variable:
        subst[b.name] = a
        return subst

    # Both variables: arbitrarily choose to bind a to b.
    if a.is_variable and b.is_variable:
        if a.name != b.name:
            subst[a.name] = b
        return subst

    return None


def _py_unify_predicates(a: Predicate, b: Predicate, subst: Optional[Substitution] = None) -> Optional[Substitution]:
    """
    Unify two predicates with the same name and arity.
    """
    if a.name != b.name or len(a.args) != len(b.args):
        return None
    subst = {} if subst is None else dict(subst)
    for ta, tb in zip(a.args, b.args):
        subst = _py_unify_terms(ta, tb, subst)
        if subst is None:
            return None
    return subst


def unify_predicates(a: Predicate, b: Predicate, subst: Optional[Substitution] = None) -> Optional[Substitution]:
    """
    Unify two predicates using SWI-Prolog (pySwip) by default.
    Falls back to legacy Python unifier only when policy is `fallback`.
    """
    try:
        backend = _get_prolog_backend()
        return backend.unify_predicates(a, b, subst)
    except Exception as exc:
        if _prefer_python_fallback_on_backend_error():
            return _py_unify_predicates(a, b, subst)
        raise RuntimeError(
            "SWI-Prolog inference backend unavailable while "
            "LLM_PROLOG_INFERENCE_POLICY=strict. "
            "Install SWI-Prolog and pyswip, or set "
            "LLM_PROLOG_INFERENCE_POLICY=fallback."
        ) from exc


def apply_subst_predicate(pred: Predicate, subst: Substitution) -> Predicate:
    """Apply a substitution to a predicate, returning a new predicate."""
    # Special-case our internal arithmetic builtin.
    if pred.name == "mathIs" and len(pred.args) == 2:
        lhs, rhs_expr = pred.args
        # Apply substitution to LHS term.
        if lhs.is_variable and lhs.name in subst:
            lhs = subst[lhs.name]

        # Rewrite RHS expression string by substituting known variables.
        expr = rhs_expr.name
        for var_name, bound_term in subst.items():
            if not bound_term.is_variable:
                expr = re.sub(rf"\b{re.escape(var_name)}\b", bound_term.name, expr)
        return Predicate(name="mathIs", args=(lhs, Term.constant(expr)))

    new_args: List[Term] = []
    for t in pred.args:
        if t.is_variable and t.name in subst:
            new_args.append(subst[t.name])
        else:
            new_args.append(t)
    return Predicate(name=pred.name, args=tuple(new_args))


def _as_fact(clause: Clause) -> Optional[Fact]:
    return clause if isinstance(clause, Fact) else None


def _as_rule(clause: Clause) -> Optional[Rule]:
    return clause if isinstance(clause, Rule) else None


def _is_ground_arith_expr(expr: str) -> bool:
    """True iff expr contains no Prolog-style variables."""
    return _PROLOG_VAR_RE.search(expr) is None


def _term_to_number(t: Term) -> Optional[float]:
    """Convert a constant Term to a number if possible."""
    if t.is_variable:
        return None
    s = t.name.strip()
    if re.fullmatch(r"[+-]?\d+", s):
        return float(int(s))
    if re.fullmatch(r"[+-]?\d+\.\d+", s):
        return float(s)
    return None


def _safe_eval_arith(expr: str) -> Optional[float]:
    """
    Evaluate a restricted arithmetic expression.

    Allowed:
    - integers / floats
    - unary +/-
    - binary +, -, *, /, //, %, **
    - parentheses (via AST)
    """
    expr = expr.strip()
    if not expr:
        return None

    # Lightweight Prolog-ish normalization.
    expr = re.sub(r"\bmod\b", "%", expr)
    expr = re.sub(r"\bdiv\b", "//", expr)

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def eval_node(n: ast.AST) -> Optional[float]:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = eval_node(n.operand)
            if v is None:
                return None
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp) and isinstance(
            n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        ):
            l = eval_node(n.left)
            r = eval_node(n.right)
            if l is None or r is None:
                return None
            if isinstance(n.op, ast.Add):
                return l + r
            if isinstance(n.op, ast.Sub):
                return l - r
            if isinstance(n.op, ast.Mult):
                return l * r
            if isinstance(n.op, ast.Div):
                return l / r
            if isinstance(n.op, ast.FloorDiv):
                return l // r
            if isinstance(n.op, ast.Mod):
                return l % r
            if isinstance(n.op, ast.Pow):
                return l ** r
        return None

    return eval_node(node)


def _reduce_mathis_in_rule(rule: Rule) -> Clause:
    """
    Reduce any `mathIs/2` atoms in the body that have a ground RHS expression.

    This emulates Prolog `is/2`: evaluate RHS, then unify LHS with the result.
    On success, the `mathIs` atom is removed; on failure, no reduction occurs.
    """
    current: Rule = rule
    subst: Substitution = {}

    changed = True
    while changed:
        changed = False
        new_body: List[Predicate] = []

        for atom in current.body:
            if atom.name != "mathIs" or len(atom.args) != 2:
                new_body.append(atom)
                continue

            lhs, rhs_expr_term = atom.args
            expr = rhs_expr_term.name

            # Apply any known substitutions into the atom (including expression rewrite).
            atom = apply_subst_predicate(atom, subst)
            lhs, rhs_expr_term = atom.args
            expr = rhs_expr_term.name

            if not _is_ground_arith_expr(expr):
                new_body.append(atom)
                continue

            value = _safe_eval_arith(expr)
            if value is None:
                new_body.append(atom)
                continue

            # Prefer integer rendering when possible.
            if float(value).is_integer():
                value_term = Term.constant(str(int(value)))
            else:
                value_term = Term.constant(str(value))

            extended = _py_unify_terms(lhs, value_term, subst)
            if extended is None:
                # Prolog would fail this branch; we represent this as "no reduction".
                new_body.append(atom)
                continue

            subst = extended
            changed = True
            # Drop the satisfied mathIs atom.

        if changed:
            # Apply newly learned substitutions across head/body and iterate again,
            # since bindings can make more mathIs expressions ground.
            head2 = apply_subst_predicate(current.head, subst)
            body2 = tuple(apply_subst_predicate(a, subst) for a in new_body)
            current = Rule(head=head2, body=body2)
        else:
            current = Rule(head=current.head, body=tuple(new_body))

    if not current.body:
        return Fact(predicate=current.head)
    return current


def _infer_rule_fact(rule: Rule, fact: Fact) -> Optional[Clause]:
    """
    Derive a new clause from a rule and a fact by unifying the fact with one
    body atom. If that body atom was the only one, return a Fact; otherwise
    return a new Rule with the instantiated head and remaining body atoms.
    """
    for i, body_atom in enumerate(rule.body):
        subst = unify_predicates(body_atom, fact.predicate)
        if subst is None:
            continue
        head_instantiated = apply_subst_predicate(rule.head, subst)
        remaining = [
            apply_subst_predicate(p, subst)
            for j, p in enumerate(rule.body)
            if j != i
        ]
        if not remaining:
            return Fact(predicate=head_instantiated)
        reduced: Clause = Rule(head=head_instantiated, body=tuple(remaining))
        if isinstance(reduced, Rule):
            reduced = _reduce_mathis_in_rule(reduced)
        return reduced
    return None


def _maybe_reduce_mathis(clause: Clause) -> Tuple[Clause, bool]:
    """
    If the clause is a Rule, attempt to reduce mathIs/2 builtins in its body.
    Returns the possibly reduced clause and a boolean indicating whether reduction occurred.
    """
    if isinstance(clause, Rule):
        reduced = _reduce_mathis_in_rule(clause)
        if reduced != clause:
            return reduced, True
    return clause, False

def reduce_rule_by_facts(premises: Tuple[Premise, ...]) -> Optional[Clause]:
    """
    Derive a new clause from a tuple of premises containing exactly one rule,
    zero or more facts, and possibly reduce built-in math expressions.

    - Math expressions (mathIs/2) in the rule body are opportunistically reduced
      even before considering facts. If any binding or rewriting can be done through
      mathIs/2, it is performed at each iteration, even if no fact is consumed.
    - One rule + one fact: unify the fact with one body atom. If it was the
      only body atom, return a Fact (instantiated head); otherwise return a
      new Rule with instantiated head and remaining body atoms. Each reduction 
      may further allow math expression reduction in the rule.
    - One rule + multiple facts: reduce the rule by each fact in turn; after each 
      reduction of a fact, also reduce mathIs/2 in the rule. Return the final 
      derived clause (Fact or Rule) if at least one reduction occurred.

    Example (bird / swims / flightless):
      [1] bird(penguin).  [2] swims(penguin).  [3] flightless(B) :- bird(B), swims(B).
      Step 1: premises = ([1], [3]) -> flightless(penguin) :- swims(penguin).
      Step 2: premises = ([2], [4]) -> flightless(penguin).

    Example with math (lives/2):
      [1] mathIs(A, 2+3). [2] foo(A). [3] bar(B) :- foo(B), mathIs(B, 2+3).
      Step 1: mathIs(B, 2+3) is reduced to mathIs(B, 5), and, if B unifies, B=5.
    """
    rules = [p for p in premises if _as_rule(p.clause) is not None]
    facts = [p for p in premises if _as_fact(p.clause) is not None]
    if len(rules) != 1:
        return None
    rule = _as_rule(rules[0].clause)
    assert rule is not None
    fact_clauses: List[Fact] = []
    for p in facts:
        f = _as_fact(p.clause)
        if f is not None:
            fact_clauses.append(f)
    current: Clause = rule
    any_reduction = False

    # Opportunistically reduce math expressions before using facts.
    current, reduced = _maybe_reduce_mathis(current)
    any_reduction = any_reduction or reduced

    for fact in fact_clauses:
        if not isinstance(current, Rule):
            break
        derived = _infer_rule_fact(current, fact)
        if derived is not None:
            current = derived
            any_reduction = True

            # Reduce math expressions again after applying fact.
            current, reduced = _maybe_reduce_mathis(current)
            any_reduction = any_reduction or reduced
    return current if any_reduction else None


def infer_new_premise(premises: List[Premise]) -> Optional[Clause]:
    """
    Public entry point: attempt to derive a new clause from a list of premises
    containing exactly one rule and one or more facts. Returns a Fact or Rule,
    or None.
    """
    return reduce_rule_by_facts(tuple(premises))