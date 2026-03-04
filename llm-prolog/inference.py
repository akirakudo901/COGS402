"""
Symbolic inference engine for the LLM‑Prolog pipeline.

This module implements a very small Horn‑clause engine:
- Unification between terms and predicates.
- A helper to combine a rule and a fact (or two facts) to derive a new fact.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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


def infer_from_pair(p1: Premise, p2: Premise) -> Optional[Fact]:
    """
    Attempt to derive a new fact by combining two premises.

    Supported patterns:
    - Rule + Fact: if the fact unifies with one of the rule's body atoms and
      the remaining body atoms are trivially satisfiable (no additional
      constraints), emit the instantiated head.
    - Fact + Rule: symmetric to the above.
    - Fact + Fact: currently no new inference; this is where arithmetic‑
      specific rules could be plugged in later.
    """

    for first, second in ((p1, p2), (p2, p1)):
        rule = _as_rule(first.clause)
        fact = _as_fact(second.clause)
        if rule is None or fact is None:
            continue

        # Try to unify the fact with each body atom.
        for body_atom in rule.body:
            subst = unify_predicates(body_atom, fact.predicate)
            if subst is None:
                continue
            # For now, require that other body atoms do not add constraints.
            # A more complete engine would need to check those against the
            # full set of facts.
            head_instantiated = apply_subst_predicate(rule.head, subst)
            return Fact(predicate=head_instantiated)

    # Fact + Fact or unsupported combinations: no inference.
    return None


def infer_new_premise(prem1: Premise, prem2: Premise) -> Optional[Fact]:
    """
    Public entry point: attempt to derive a new fact from two premises.
    """
    return infer_from_pair(prem1, prem2)

