"""
Symbolic inference engine for the LLM‑Prolog pipeline.

This module implements a very small Horn‑clause engine:
- Unification between terms and predicates.
- Deriving a new clause (Fact or Rule) from one rule and one or more facts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .types import Clause, Fact, Predicate, Premise, Rule, Term


Substitution = Dict[str, Term]


def unify_terms(a: Term, b: Term, subst: Substitution) -> Optional[Substitution]:
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


def unify_predicates(a: Predicate, b: Predicate, subst: Optional[Substitution] = None) -> Optional[Substitution]:
    """
    Unify two predicates with the same name and arity.
    """
    if a.name != b.name or len(a.args) != len(b.args):
        return None
    subst = {} if subst is None else dict(subst)
    for ta, tb in zip(a.args, b.args):
        subst = unify_terms(ta, tb, subst)
        if subst is None:
            return None
    return subst


def apply_subst_predicate(pred: Predicate, subst: Substitution) -> Predicate:
    """Apply a substitution to a predicate, returning a new predicate."""
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
        return Rule(head=head_instantiated, body=tuple(remaining))
    return None


def reduce_rule_by_facts(premises: Tuple[Premise, ...]) -> Optional[Clause]:
    """
    Derive a new clause from a tuple of premises containing exactly one rule
    and the rest facts.

    - One rule + one fact: unify the fact with one body atom. If it was the
      only body atom, return a Fact (instantiated head); otherwise return a
      new Rule with instantiated head and remaining body atoms.
    - One rule + multiple facts: reduce the rule by each fact in turn; return
      the final derived clause (Fact or Rule) if at least one reduction
      occurred.

    Example (bird / swims / flightless):
      [1] bird(penguin).  [2] swims(penguin).  [3] flightless(B) :- bird(B), swims(B).
      Step 1: premises = ([1], [3]) -> flightless(penguin) :- swims(penguin).
      Step 2: premises = ([2], [4]) -> flightless(penguin).
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
    for fact in fact_clauses:
        if not isinstance(current, Rule):
            break
        derived = _infer_rule_fact(current, fact)
        if derived is not None:
            current = derived
            any_reduction = True
    return current if any_reduction else None


def infer_new_premise(premises: List[Premise]) -> Optional[Clause]:
    """
    Public entry point: attempt to derive a new clause from a list of premises
    containing exactly one rule and one or more facts. Returns a Fact or Rule,
    or None.
    """
    return reduce_rule_by_facts(tuple(premises))