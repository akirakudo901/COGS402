"""
Symbolic inference engine for the LLM‑Prolog pipeline.

This module implements a very small Horn‑clause engine:
- Unification between terms and predicates.
- Deriving a new clause (Fact or Rule) from one consumer rule and ordered
  producers (facts or rules), using fixed body slots and two-pass global
  simplification (start and end only).
"""

from __future__ import annotations

import ast
import os
import re
import threading
from typing import Dict, List, Optional, Tuple

from .types import Clause, Fact, Predicate, Premise, Rule, Term, _parse_term, format_clause, parse_fact_or_rule

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


def _py_unify_predicate_args(
    a: Term | Predicate,
    b: Term | Predicate,
    subst: Substitution,
) -> Optional[Substitution]:
    if isinstance(a, Term) and isinstance(b, Term):
        return _py_unify_terms(a, b, subst)
    if isinstance(a, Predicate) and isinstance(b, Predicate):
        return _py_unify_predicates(a, b, subst)
    return None


def _py_unify_predicates(a: Predicate, b: Predicate, subst: Optional[Substitution] = None) -> Optional[Substitution]:
    """
    Unify two predicates with the same name and arity.
    """
    if a.name != b.name or len(a.args) != len(b.args):
        return None
    subst = {} if subst is None else dict(subst)
    for ta, tb in zip(a.args, b.args):
        subst = _py_unify_predicate_args(ta, tb, subst)
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
    def apply_subst_arg(arg: Term | Predicate) -> Term | Predicate:
        if isinstance(arg, Predicate):
            return apply_subst_predicate(arg, subst)
        if arg.is_variable and arg.name in subst:
            return subst[arg.name]
        return arg

    # Special-case our internal arithmetic builtin.
    if pred.name == "mathIs" and len(pred.args) == 2:
        lhs, rhs_expr = pred.args
        lhs = apply_subst_arg(lhs)
        rhs_expr = apply_subst_arg(rhs_expr)

        # Rewrite RHS expression string by substituting known variables.
        # TODO might be obsolete now that we can resursively define Predicate for
        #      inner math expressions as well
        if isinstance(rhs_expr, Term):
            expr = rhs_expr.name
            for var_name, bound_term in subst.items():
                if not bound_term.is_variable:
                    expr = re.sub(rf"\b{re.escape(var_name)}\b", bound_term.name, expr)
        return Predicate(name="mathIs", args=(lhs, Term.constant(expr)))

    new_args: List[Term | Predicate] = []
    for t in pred.args:
        new_args.append(apply_subst_arg(t))
    return Predicate(name=pred.name, args=tuple(new_args))


def _as_fact(clause: Clause) -> Optional[Fact]:
    return clause if isinstance(clause, Fact) else None


def _as_rule(clause: Clause) -> Optional[Rule]:
    return clause if isinstance(clause, Rule) else None


def _collect_variable_names_in_rule(rule: Rule) -> set[str]:
    names: set[str] = set()
    for arg in rule.head.args:
        names.update(_collect_variable_names_from_arg(arg))
    for atom in rule.body:
        for arg in atom.args:
            names.update(_collect_variable_names_from_arg(arg))
    return names


def _rename_term_vars(term: Term, mapping: Dict[str, str]) -> Term:
    if term.is_variable and term.name in mapping:
        return Term.variable(mapping[term.name])
    return term


def _rename_arg_vars(arg: Term | Predicate, mapping: Dict[str, str]) -> Term | Predicate:
    if isinstance(arg, Predicate):
        return _rename_predicate_vars(arg, mapping)
    return _rename_term_vars(arg, mapping)


def _rename_predicate_vars(pred: Predicate, mapping: Dict[str, str]) -> Predicate:
    return Predicate(name=pred.name, args=tuple(_rename_arg_vars(a, mapping) for a in pred.args))


def _standardize_rule_apart(producer: Rule, forbidden: set[str]) -> Rule:
    """
    Rename all variables in `producer` to fresh names not in `forbidden`
    (or each other), preserving structure.
    """
    src_vars = sorted(_collect_variable_names_in_rule(producer))
    if not src_vars:
        return producer
    used = set(forbidden)
    mapping: Dict[str, str] = {}
    for v in src_vars:
        i = 0
        while True:
            cand = f"Fresh{i}"
            if cand not in used:
                mapping[v] = cand
                used.add(cand)
                break
            i += 1
    new_head = _rename_predicate_vars(producer.head, mapping)
    new_body = tuple(_rename_predicate_vars(p, mapping) for p in producer.body)
    return Rule(head=new_head, body=new_body)


def _vars_from_head_and_slots(head: Predicate, slots: List[List[Predicate]]) -> set[str]:
    names: set[str] = set()
    for arg in head.args:
        names.update(_collect_variable_names_from_arg(arg))
    for sl in slots:
        for pred in sl:
            for arg in pred.args:
                names.update(_collect_variable_names_from_arg(arg))
    return names


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


def _predicate_to_prolog_goal_text(pred: Predicate) -> str:
    """
    Render an internal predicate as an executable Prolog goal text.

    Internally we store arithmetic evaluation as `mathIs/2`; SWI executes it as
    the `is/2` operator.
    """
    if pred.name == "mathIs" and len(pred.args) == 2:
        lhs, rhs = pred.args
        return f"{lhs.to_prolog_text()} is {rhs.name.strip()}"
    return pred.to_prolog_text()


def _prolog_value_to_term(value: object) -> Optional[Term]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _parse_term(text)


def _collect_variable_names_from_arg(arg: Term | Predicate) -> List[str]:
    if isinstance(arg, Term):
        return [arg.name] if arg.is_variable else []
    names: List[str] = []
    for nested in arg.args:
        names.extend(_collect_variable_names_from_arg(nested))
    return names


def _reduce_rule_via_prolog_truth_and_constants(rule: Rule) -> Clause:
    """
    General SWI-Prolog simplifier for rule bodies.

    For each body atom, execute it as a Prolog goal under currently known
    substitutions:
    - If the goal is provably true and fully ground, remove it.
    - If the goal yields only ground bindings for its free variables, apply the
      bindings and remove that goal.
    """
    try:
        backend = _get_prolog_backend()
    except Exception:
        if _prefer_python_fallback_on_backend_error():
            return rule
        raise

    current: Rule = rule
    subst: Substitution = {}

    changed = True
    while changed:
        changed = False
        new_body: List[Predicate] = []

        for atom in current.body:
            atom = apply_subst_predicate(atom, subst)
            free_vars = list(dict.fromkeys(
                name
                for arg in atom.args
                for name in _collect_variable_names_from_arg(arg)
            ))

            goals = [
                f"{var_name} = {bound_term.to_prolog_text()}"
                for var_name, bound_term in subst.items()
            ]
            goals.append(_predicate_to_prolog_goal_text(atom))
            query = ", ".join(goals)

            try:
                with backend._lock:
                    solutions = list(backend._prolog.query(query, maxresult=1))
            except Exception:
                new_body.append(atom)
                continue

            if not solutions:
                new_body.append(atom)
                continue

            solution = solutions[0]
            candidate_bindings: List[Tuple[str, Term]] = []
            reducible = True

            for var_name in free_vars:
                raw = solution.get(var_name)
                term = _prolog_value_to_term(raw)
                # Not uniquely reduced to a ground constant yet.
                if term is None or term.is_variable:
                    reducible = False
                    break
                candidate_bindings.append((var_name, term))

            if not reducible:
                new_body.append(atom)
                continue

            local_subst = dict(subst)
            ok = True
            for var_name, term in candidate_bindings:
                unified = _py_unify_terms(Term.variable(var_name), term, local_subst)
                if unified is None:
                    ok = False
                    break
                local_subst = unified

            if not ok:
                new_body.append(atom)
                continue

            subst = local_subst
            changed = True
            # Goal satisfied and reduced; omit from new body.

        if changed:
            head2 = apply_subst_predicate(current.head, subst)
            body2 = tuple(apply_subst_predicate(a, subst) for a in new_body)
            current = Rule(head=head2, body=body2)
        else:
            current = Rule(head=current.head, body=tuple(new_body))

    if not current.body:
        return Fact(predicate=current.head)
    return current


def _reduce_clause_text_by_prolog(clause_text: str) -> Clause:
    """
    Parse and simplify a Prolog clause represented as text.

    This accepts any parseable fact/rule text. Facts are returned unchanged.
    Rules are simplified by removing goals that are provably true and by
    propagating single-goal constant reductions.
    """
    clause = parse_fact_or_rule(clause_text)
    if isinstance(clause, Rule):
        return _reduce_rule_via_prolog_truth_and_constants(clause)
    return clause


def _maybe_reduce_clause_with_prolog(clause: Clause) -> Tuple[Clause, bool]:
    """
    Try to simplify a clause via SWI-Prolog semantics.
    Returns the possibly reduced clause and whether reduction occurred.
    """
    reduced = _reduce_clause_text_by_prolog(format_clause(clause))
    return reduced, (reduced != clause)


def _global_simplify_clause(clause: Clause) -> Tuple[Clause, bool]:
    """
    One global simplify pass: reduce mathIs/2 on rules, then SWI-based
    simplification. Used only at the start and end of inference (not between
    slot iterations).
    """
    any_changed = False
    current = clause
    if isinstance(current, Rule):
        m = _reduce_mathis_in_rule(current)
        if m != current:
            any_changed = True
            current = m
    cur2, r2 = _maybe_reduce_clause_with_prolog(current)
    if r2:
        any_changed = True
        current = cur2
    return current, any_changed

def _clause_to_slots_rule(clause: Clause) -> Optional[Tuple[Predicate, List[List[Predicate]]]]:
    """Return (head, slots) where each slot is a list of predicates; None if not a Rule."""
    if not isinstance(clause, Rule):
        return None
    slots = [[p] for p in clause.body]
    return clause.head, slots


def _slots_to_clause(head: Predicate, slots: List[List[Predicate]]) -> Clause:
    body = tuple(p for sl in slots for p in sl)
    if not body:
        return Fact(predicate=head)
    return Rule(head=head, body=body)


def validate_inference_premise_selection(premises: List[Premise]) -> Optional[str]:
    """
    Return an error message if ``premises`` cannot be used for multi-premise
    inference (first must be a consumer ``Rule``, rest ``Fact`` or ``Rule``).
    Return ``None`` if valid. Single-premise calls are not validated here.
    """
    if len(premises) < 2:
        return None
    first = premises[0].clause
    if not isinstance(first, Rule):
        return (
            "Inference requires the first selected premise to be a rule (consumer); "
            f"got {type(first).__name__}."
        )
    for p in premises[1:]:
        c = p.clause
        if not isinstance(c, (Fact, Rule)):
            return (
                "Inference requires all premises after the consumer to be facts or rules "
                f"(producers); got {type(c).__name__} for premise id {p.id}."
            )
    return None


def infer_new_premise(premises: List[Premise]) -> Optional[Clause]:
    """
    Derive a new clause from premises.

    - If a single premise is given (a Rule), run global simplification twice
      (start/end) and return the result if anything changed.
    - Otherwise: first premise must be the **consumer** ``Rule``; remaining
      premises are **producers** (``Fact`` or ``Rule``) in order.

    After an opening global simplify pass, **N** body slots are frozen (one per
    body atom). For each slot index ``i`` in order, the first producer in the
    pool that unifies with that slot's atom (when the slot has exactly one
    atom) is applied and removed from the pool. A fact removes the atom; a rule
    replaces it with the producer's body (standardized apart). Slots with
    multiple atoms (from a rule splice) are skipped for producer matching.
    No global simplification runs between slot iterations.

    Returns a ``Fact`` or ``Rule``, or ``None`` if nothing changed.
    """
    if not premises:
        return None

    if len(premises) == 1:
        only = premises[0].clause
        if not isinstance(only, Rule):
            return None
        cur, r1 = _global_simplify_clause(only)
        if not isinstance(cur, Rule):
            return cur if r1 else None
        cur2, r2 = _global_simplify_clause(cur)
        return cur2 if (r1 or r2) else None

    consumer = premises[0].clause
    if not isinstance(consumer, Rule):
        return None
    producer_clauses: List[Clause] = []
    for p in premises[1:]:
        c = p.clause
        if not isinstance(c, (Fact, Rule)):
            return None
        producer_clauses.append(c)

    any_reduction = False
    current: Clause = consumer
    cur, reduced_open = _global_simplify_clause(current)
    if reduced_open:
        any_reduction = True
    current = cur

    if not isinstance(current, Rule):
        return current if any_reduction else None

    parsed = _clause_to_slots_rule(current)
    if parsed is None:
        return None
    head, slots = parsed
    n = len(slots)
    pool: List[Clause] = producer_clauses

    slot_applied = False
    for i in range(n):
        if len(slots[i]) != 1:
            continue
        atom = slots[i][0]
        for pool_idx, prod in enumerate(pool):
            # Handle fact slotting
            if isinstance(prod, Fact):
                subst = unify_predicates(atom, prod.predicate)
                if subst is None:
                    continue
                slots[i] = []
            # Handle rule slotting
            else:
                subst_rule = _as_rule(prod)
                assert subst_rule is not None
                forbidden = _vars_from_head_and_slots(head, slots)
                prod_f = _standardize_rule_apart(subst_rule, forbidden)
                subst = unify_predicates(atom, prod_f.head)
                if subst is None:
                    continue
                slots[i] = [apply_subst_predicate(p, subst) for p in prod_f.body]
            # Apply substitution to head & other slots
            head = apply_subst_predicate(head, subst)
            for j in range(n):
                slots[j] = [apply_subst_predicate(p, subst) for p in slots[j]]
            # Remove applied object from pool_idx
            pool.pop(pool_idx)
            slot_applied = True
            break

    out = _slots_to_clause(head, slots)
    cur_end, reduced_end = _global_simplify_clause(out)
    if reduced_end:
        any_reduction = True
    out = cur_end

    if any_reduction or slot_applied:
        return out
    return None


def reduce_rule_by_facts(premises: Tuple[Premise, ...]) -> Optional[Clause]:
    """
    Backward-compatible alias for :func:`infer_new_premise`.
    """
    return infer_new_premise(list(premises))