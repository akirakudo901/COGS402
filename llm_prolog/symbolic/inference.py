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
from typing import Dict, List, Literal, Optional, Tuple, Union

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
    if pred.name in ["mathIs", "is"] and len(pred.args) == 2:
        lhs, rhs_expr = pred.args
        lhs = apply_subst_arg(lhs)
        rhs_expr = apply_subst_arg(rhs_expr)

        # Legacy RHS format: `mathIs/2` storing the RHS as a constant Term whose
        # `name` is an infix expression string (e.g. `"2*X+1"`). Substitution
        # doesn't rewrite inside that string, so we do a best-effort word-boundary
        # replacement here; the reduction pass will then convert it into nested
        # arithmetic Predicates.
        if isinstance(rhs_expr, Term) and not rhs_expr.is_variable:
            expr = rhs_expr.name
            for var_name, bound_term in subst.items():
                expr = re.sub(rf"\b{re.escape(var_name)}\b", bound_term.name, expr)

        return Predicate(name=pred.name, args=(lhs, Term.constant(expr)))

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


def _arith_normalize_source(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"\bmod\b", "%", expr)
    expr = re.sub(r"\bdiv\b", "//", expr)
    return expr


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

    expr = _arith_normalize_source(expr)

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


_MATHIS_POLY_EPS = 1e-9


def _fold_constants_in_ast(node: ast.AST) -> ast.AST:
    """
    Fold arithmetic subexpressions that contain no variable names (ast.Name).
    Reuses the same operator set as _safe_eval_arith.
    """

    def contains_name(n: ast.AST) -> bool:
        if isinstance(n, ast.Name):
            return True
        if isinstance(n, ast.Expression):
            return contains_name(n.body)
        if isinstance(n, ast.UnaryOp):
            return contains_name(n.operand)
        if isinstance(n, ast.BinOp):
            return contains_name(n.left) or contains_name(n.right)
        if isinstance(n, ast.Constant):
            return False
        return True

    def fold_inner(n: ast.AST) -> ast.AST:
        if isinstance(n, ast.Expression):
            inner = fold_inner(n.body)
            return ast.Expression(body=inner)
        if isinstance(n, ast.Name):
            return n
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            op = fold_inner(n.operand)
            if isinstance(op, ast.Constant) and isinstance(op.value, (int, float)):
                v = float(op.value)
                v = v if isinstance(n.op, ast.UAdd) else -v
                if float(v).is_integer():
                    return ast.Constant(value=int(v))
                return ast.Constant(value=v)
            return ast.UnaryOp(op=n.op, operand=op)
        if isinstance(n, ast.BinOp) and isinstance(
            n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        ):
            left = fold_inner(n.left)
            right = fold_inner(n.right)
            if (
                isinstance(left, ast.Constant)
                and isinstance(right, ast.Constant)
                and isinstance(left.value, (int, float))
                and isinstance(right.value, (int, float))
            ):
                a = float(left.value)
                b = float(right.value)
                if isinstance(n.op, ast.Add):
                    out = a + b
                elif isinstance(n.op, ast.Sub):
                    out = a - b
                elif isinstance(n.op, ast.Mult):
                    out = a * b
                elif isinstance(n.op, ast.Div):
                    out = a / b
                elif isinstance(n.op, ast.FloorDiv):
                    out = a // b
                elif isinstance(n.op, ast.Mod):
                    out = a % b
                elif isinstance(n.op, ast.Pow):
                    out = a ** b
                else:
                    return ast.BinOp(left=left, op=n.op, right=right)
                if float(out).is_integer():
                    return ast.Constant(value=int(out))
                return ast.Constant(value=float(out))
            return ast.BinOp(left=left, op=n.op, right=right)
        return n

    folded = fold_inner(node)
    if contains_name(folded):
        return folded
    to_unparse = folded.body if isinstance(folded, ast.Expression) else folded
    v = _safe_eval_arith(ast.unparse(to_unparse))
    if v is None:
        return folded
    if float(v).is_integer():
        return ast.Constant(value=int(v))
    return ast.Constant(value=float(v))


def _substitute_bound_vars_in_arith_ast(node: ast.AST, subst: Substitution) -> ast.AST:
    """Inline variables that are already bound to ground numeric constants."""

    def walk(n: ast.AST) -> ast.AST:
        if isinstance(n, ast.Expression):
            inner = walk(n.body)
            return ast.Expression(body=inner)
        if isinstance(n, ast.Name):
            t = subst.get(n.id)
            if t is not None and not t.is_variable:
                num = _term_to_number(t)
                if num is not None:
                    if float(num).is_integer():
                        return ast.Constant(value=int(num))
                    return ast.Constant(value=float(num))
            return n
        if isinstance(n, ast.Constant):
            return n
        if isinstance(n, ast.UnaryOp):
            return ast.UnaryOp(op=n.op, operand=walk(n.operand))
        if isinstance(n, ast.BinOp):
            return ast.BinOp(left=walk(n.left), op=n.op, right=walk(n.right))
        return n

    return walk(node)


def _poly_trim(coeffs: List[float]) -> List[float]:
    out = list(coeffs)
    while len(out) > 1 and abs(out[-1]) < _MATHIS_POLY_EPS:
        out.pop()
    return out


def _poly_neg(coeffs: List[float]) -> List[float]:
    return [-c for c in coeffs]


def _poly_add(a: List[float], b: List[float]) -> List[float]:
    m = max(len(a), len(b))
    out = [(a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0) for i in range(m)]
    return _poly_trim(out)


def _poly_sub(a: List[float], b: List[float]) -> List[float]:
    m = max(len(a), len(b))
    out = [(a[i] if i < len(a) else 0.0) - (b[i] if i < len(b) else 0.0) for i in range(m)]
    return _poly_trim(out)


def _poly_mul(a: List[float], b: List[float]) -> List[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return _poly_trim(out)


def _poly_scalar_div(p: List[float], k: float) -> Optional[List[float]]:
    if abs(k) < _MATHIS_POLY_EPS:
        return None
    return _poly_trim([c / k for c in p])


def _poly_pow(base: List[float], exp: int) -> Optional[List[float]]:
    if exp < 0:
        return None
    if exp == 0:
        return [1.0]
    acc = base
    for _ in range(1, exp):
        acc = _poly_mul(acc, base)
    return acc


def _ast_to_polynomial(var: str, n: ast.AST) -> Optional[List[float]]:
    """
    Map an AST expression to dense coefficients in ``var`` (lowest degree first),
    or None if the expression is not a polynomial in ``var`` (e.g. multivariate).
    """
    if isinstance(n, ast.Expression):
        return _ast_to_polynomial(var, n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return [float(n.value)]
    if isinstance(n, ast.Name):
        if n.id == var:
            return [0.0, 1.0]
        return None
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        inner = _ast_to_polynomial(var, n.operand)
        return None if inner is None else _poly_neg(inner)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
        return _ast_to_polynomial(var, n.operand)
    if isinstance(n, ast.BinOp):
        if isinstance(n.op, ast.Add):
            pl = _ast_to_polynomial(var, n.left)
            pr = _ast_to_polynomial(var, n.right)
            if pl is None or pr is None:
                return None
            return _poly_add(pl, pr)
        if isinstance(n.op, ast.Sub):
            pl = _ast_to_polynomial(var, n.left)
            pr = _ast_to_polynomial(var, n.right)
            if pl is None or pr is None:
                return None
            return _poly_sub(pl, pr)
        if isinstance(n.op, ast.Mult):
            pl = _ast_to_polynomial(var, n.left)
            pr = _ast_to_polynomial(var, n.right)
            if pl is None or pr is None:
                return None
            return _poly_mul(pl, pr)
        if isinstance(n.op, ast.Div):
            pr = _ast_to_polynomial(var, n.right)
            if pr is None or len(pr) != 1:
                return None
            pl = _ast_to_polynomial(var, n.left)
            if pl is None:
                return None
            return _poly_scalar_div(pl, pr[0])
        if isinstance(n.op, ast.Pow):
            if not isinstance(n.right, ast.Constant) or not isinstance(n.right.value, (int, float)):
                return None
            exp = int(n.right.value)
            if exp != float(n.right.value) or exp < 0:
                return None
            pl = _ast_to_polynomial(var, n.left)
            if pl is None:
                return None
            return _poly_pow(pl, exp)
    return None


def _solve_polynomial_real_unique(coeffs: List[float]) -> Union[Literal["tautology"], Literal["none"], float]:
    """
    ``coeffs[k]`` is the coefficient of var^k. Solve sum_k coeffs[k] * var^k = 0.
    Returns a single real root, 'tautology' if identically zero, or 'none'.
    """
    c = _poly_trim([float(x) for x in coeffs])
    if not c:
        return "tautology"
    if len(c) == 1:
        if abs(c[0]) < _MATHIS_POLY_EPS:
            return "tautology"
        return "none"
    if len(c) == 2:
        return -c[0] / c[1]

    import numpy as np

    high_first = c[::-1]
    roots = np.roots(high_first)
    reals: List[float] = []
    for r in roots:
        if abs(r.imag) < 1e-8:
            reals.append(float(r.real))
    if not reals:
        return "none"
    uniq: List[float] = []
    for x in sorted(reals):
        if not uniq or abs(x - uniq[-1]) > 1e-6:
            uniq.append(x)
    if len(uniq) != 1:
        return "none"
    return uniq[0]


def _collect_prolog_var_names_from_ast(node: ast.AST) -> set[str]:
    names: set[str] = set()

    def walk(n: ast.AST) -> None:
        if isinstance(n, ast.Expression):
            walk(n.body)
            return
        if isinstance(n, ast.Name):
            if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", n.id):
                names.add(n.id)
            return
        if isinstance(n, ast.UnaryOp):
            walk(n.operand)
            return
        if isinstance(n, ast.BinOp):
            walk(n.left)
            walk(n.right)
            return

    walk(node)
    return names


def _term_to_arith_ast(term: Term, subst: Substitution) -> Optional[ast.AST]:
    if term.is_variable:
        if term.name in subst:
            return _term_to_arith_ast(subst[term.name], subst)
        return ast.Name(id=term.name, ctx=ast.Load())
    s = term.name.strip()
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        v = float(s)
        if float(v).is_integer():
            return ast.Constant(value=int(v))
        return ast.Constant(value=v)
    try:
        node = ast.parse(_arith_normalize_source(s), mode="eval")
    except SyntaxError:
        return None
    folded = _fold_constants_in_ast(node)
    return folded if isinstance(folded, ast.Expression) else ast.Expression(body=folded)


def _arith_expr_arg_to_py_source(expr: Term | Predicate) -> Optional[str]:
    """
    Convert a nested arithmetic expression (as Term or operator Predicates)
    into a Python-infix string that `_safe_eval_arith` / `_try_fold_mathis_rhs`
    can parse.
    """
    if isinstance(expr, Term):
        if expr.is_variable:
            return expr.name
        s = expr.name.strip()
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
            return s
        # For legacy string expressions stored as constants, keep the raw text.
        if any(ch in s for ch in "+-*/()%") or re.search(r"[A-Z][A-Za-z0-9_]*", s):
            return s
        # Otherwise treat it like an identifier.
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
            return s
        return None

    # Predicate operators (prefix functor form like '+(A,B)').
    if len(expr.args) == 1 and expr.name in {"-", "neg"}:
        inner = _arith_expr_arg_to_py_source(expr.args[0])
        if inner is None:
            return None
        return f"(-{inner})"

    op_map: Dict[str, str] = {
        "+": "+",
        "-": "-",
        "*": "*",
        "/": "/",
        "//": "//",
        "%": "%",
        "**": "**",
        "div": "//",
        "mod": "%",
    }
    if expr.name in op_map:
        if len(expr.args) != 2:
            return None
        l = _arith_expr_arg_to_py_source(expr.args[0])
        r = _arith_expr_arg_to_py_source(expr.args[1])
        if l is None or r is None:
            return None
        return f"({l}{op_map[expr.name]}{r})"

    return None


def _py_source_to_arith_expr_arg(expr: str) -> Optional[Term | Predicate]:
    """
    Convert a Python-style arithmetic source string into our nested arithmetic
    representation (operator Predicates + Term variables/constants).
    """
    expr = expr.strip()
    if not expr:
        return None
    try:
        node = ast.parse(_arith_normalize_source(expr), mode="eval")
    except SyntaxError:
        return None

    def conv(n: ast.AST) -> Optional[Term | Predicate]:
        if isinstance(n, ast.Expression):
            return conv(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            v = float(n.value)
            if v.is_integer():
                return Term.constant(str(int(v)))
            return Term.constant(str(v))
        if isinstance(n, ast.Name):
            if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", n.id):
                return Term.variable(n.id)
            return Term.constant(n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            inner = conv(n.operand)
            if inner is None:
                return None
            return Predicate(name="-", args=(inner,))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
            return conv(n.operand)
        if isinstance(n, ast.BinOp):
            left = conv(n.left)
            right = conv(n.right)
            if left is None or right is None:
                return None
            if isinstance(n.op, ast.Add):
                return Predicate(name="+", args=(left, right))
            if isinstance(n.op, ast.Sub):
                return Predicate(name="-", args=(left, right))
            if isinstance(n.op, ast.Mult):
                return Predicate(name="*", args=(left, right))
            if isinstance(n.op, ast.Div):
                return Predicate(name="/", args=(left, right))
            if isinstance(n.op, ast.FloorDiv):
                return Predicate(name="//", args=(left, right))
            if isinstance(n.op, ast.Mod):
                return Predicate(name="%", args=(left, right))
            if isinstance(n.op, ast.Pow):
                return Predicate(name="**", args=(left, right))
            return None
        return None

    return conv(node)


def _try_fold_mathis_rhs(expr: str) -> Optional[str]:
    """
    If purely arithmetic subexpressions can be evaluated, return a new RHS
    string, else None.
    """
    try:
        raw = ast.parse(_arith_normalize_source(expr), mode="eval")
    except SyntaxError:
        return None
    raw_unparsed = ast.unparse(raw.body)
    folded = _fold_constants_in_ast(raw)
    fb = folded.body if isinstance(folded, ast.Expression) else folded
    folded_unparsed = ast.unparse(fb)
    if folded_unparsed != raw_unparsed:
        return folded_unparsed
    return None


def _try_reduce_mathis_as_relation(
    lhs: Term, rhs_expr: str, subst: Substitution
) -> Tuple[Optional[Substitution], bool]:
    """
    Treat ``mathIs(LHS, RHS)`` as the relation LHS = RHS when SWI ``is/2``
    semantics (evaluate RHS only) are too narrow.

    Returns (extended_subst_or_None, drop_atom).
    """
    rhs_expr = rhs_expr.strip()
    if not rhs_expr:
        return None, False

    try:
        rhs_ast_full = ast.parse(_arith_normalize_source(rhs_expr), mode="eval")
    except SyntaxError:
        return None, False

    rhs_ast = _fold_constants_in_ast(_substitute_bound_vars_in_arith_ast(rhs_ast_full, subst))
    rhs_body = rhs_ast.body if isinstance(rhs_ast, ast.Expression) else rhs_ast

    lhs_ast_full = _term_to_arith_ast(lhs, subst)
    if lhs_ast_full is None:
        return None, False
    lhs_ast = lhs_ast_full.body if isinstance(lhs_ast_full, ast.Expression) else lhs_ast_full
    lhs_ast = _fold_constants_in_ast(_substitute_bound_vars_in_arith_ast(lhs_ast, subst))
    lhs_ast = lhs_ast.body if isinstance(lhs_ast, ast.Expression) else lhs_ast

    diff = ast.BinOp(left=lhs_ast, op=ast.Sub(), right=rhs_body)
    diff = _fold_constants_in_ast(ast.Expression(body=diff))
    diff_body = diff.body if isinstance(diff, ast.Expression) else diff

    free_names = _collect_prolog_var_names_from_ast(diff_body)

    if not free_names:
        vnum = _safe_eval_arith(ast.unparse(diff_body))
        if vnum is None:
            return None, False
        if abs(float(vnum)) < _MATHIS_POLY_EPS:
            return subst, True
        return None, False

    if len(free_names) != 1:
        return None, False

    (var_name,) = tuple(free_names)
    poly = _ast_to_polynomial(var_name, diff_body)
    if poly is None:
        return None, False

    sol = _solve_polynomial_real_unique(poly)
    if sol == "none":
        return None, False
    if sol == "tautology":
        return subst, True

    if float(sol).is_integer():
        val_term: Term = Term.constant(str(int(sol)))
    else:
        val_term = Term.constant(str(float(sol)))

    extended = _py_unify_terms(Term.variable(var_name), val_term, dict(subst))
    if extended is None:
        return None, False
    return extended, True


def _reduce_mathis_in_rule(rule: Rule) -> Clause:
    """
    Simplify ``mathIs/2`` body atoms: Prolog-style ``is/2`` when the RHS is
    ground, constant-folding on the RHS when possible, and otherwise relational
    solving when the equality is a polynomial in a single Prolog variable.
    """
    current: Rule = rule
    subst: Substitution = {}

    changed = True
    while changed:
        changed = False
        new_body: List[Predicate] = []

        for atom in current.body:
            if atom.name not in ["mathIs", "is"] or len(atom.args) != 2:
                new_body.append(atom)
                continue

            lhs, rhs_expr_term = atom.args
            # Apply any known substitutions into the atom.
            atom = apply_subst_predicate(atom, subst)
            lhs, rhs_expr_term = atom.args

            expr_arg: Term | Predicate = rhs_expr_term  # for typing clarity
            # Deprecation handling: if RHS is the legacy "expression string as Term name"
            # format, convert it to nested arithmetic Predicates now.
            if isinstance(expr_arg, Term) and not expr_arg.is_variable:
                converted = _py_source_to_arith_expr_arg(expr_arg.name)
                if converted is not None:
                    expr_arg = converted
                    atom = Predicate(name=atom.name, args=(lhs, expr_arg))

            expr = _arith_expr_arg_to_py_source(expr_arg)
            if expr is None:
                new_body.append(atom)
                continue

            folded = _try_fold_mathis_rhs(expr)
            if folded is not None:
                folded_arg = _py_source_to_arith_expr_arg(folded)
                expr = folded
                if folded_arg is not None:
                    atom = Predicate(name=atom.name, args=(lhs, folded_arg))
                changed = True

            if not _is_ground_arith_expr(expr):
                # TODO STILL RESTRICTING LHS TO TERM DUE TO _try_reduce_mathis_as_relation's parameters
                if not isinstance(lhs, Term):
                    new_body.append(atom)
                    continue
                # TODO END
                rel_subst, drop = _try_reduce_mathis_as_relation(lhs, expr, subst)
                if drop:
                    assert rel_subst is not None
                    subst = rel_subst
                    changed = True
                    continue
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

            if not isinstance(lhs, Term):
                new_body.append(atom)
                continue
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
    if pred.name in ["mathIs", "is"] and len(pred.args) == 2:
        lhs, rhs = pred.args
        if isinstance(rhs, Term):
            rhs_text = rhs.name.strip()
        else:
            rhs_text = rhs.to_prolog_text()
        return f"{lhs.to_prolog_text()} is {rhs_text}"
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