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
import re
import threading
from typing import Dict, List, Optional, Tuple, Union

try:
    from pyswip import Prolog as _PySwipProlog  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    _PySwipProlog = None  # type: ignore[assignment,misc]

_PARSE_PROLOG_LOCK = threading.Lock()
_PARSE_PROLOG: object = None  # Prolog() instance, or False if unavailable

_NUMERIC_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")

def _escape_atom(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


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
    
    def to_prolog_text(self) -> str:
        if self.is_variable:
            return self.name
        if _NUMERIC_RE.fullmatch(self.name.strip()):
            return self.name.strip()
        return f"'{_escape_atom(self.name)}'"


PredicateArg = Union[Term, "Predicate"]


@dataclass(frozen=True)
class Predicate:
    name: str
    args: Tuple[PredicateArg, ...]

    def __repr__(self) -> str:
        return f"Predicate(name={self.name!r}, args={self.args!r})"

    def __str__(self) -> str:
        if not self.args:
            return self.name
        arg_str = ", ".join(str(t) for t in self.args)
        return f"{self.name}({arg_str})"
    
    def to_prolog_text(self) -> str:
        if not self.args:
            return self.name
        args = ", ".join(t.to_prolog_text() for t in self.args)
        return f"{self.name}({args})"


@dataclass(frozen=True)
class Fact:
    predicate: Predicate

    def __repr__(self) -> str:
        return f"Fact(predicate={self.predicate!r})"

    def __str__(self) -> str:
        return f"{self.predicate}."
    
    def to_prolog_text(self) -> str:
        return self.predicate.to_prolog_text()


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
    
    def to_prolog_text(self) -> str:
        body = ", ".join(p.to_prolog_text() for p in self.body)
        return f"{self.head.to_prolog_text()} :- {body}"


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
        Level 4: clause + parent ids
        """
        if level not in range(5):
            raise Exception("str_verbose accepts verbosity levels from 0 to 4 only.")
        clause_str = format_clause(self.clause)
        lines = [f"{self.id}: {clause_str}"]
        if self.nl and level >= 1 and level != 4:
            lines[0] += f" # {self.nl}"
        if self.parent_ids is not None and len(self.parent_ids) > 0 and level >= 2:
            lines.append(f"  (from premises {', '.join(str(pid) for pid in self.parent_ids)})")
        if self.source and level >= 3 and level != 4:
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
        def collect_var_names(arg: PredicateArg) -> set[str]:
            if isinstance(arg, Term):
                return {arg.name} if arg.is_variable else set()
            names: set[str] = set()
            for nested in arg.args:
                names.update(collect_var_names(nested))
            return names

        # Collect distinct logical variable names across all arguments.
        var_names: set[str] = set()
        for arg in self.target.args:
            var_names.update(collect_var_names(arg))
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
    # Termination-check results computed by the LLM (same single call as selection).
    # If `is_final_solution` is true, `solution_premise_id` names a ground fact in the
    # current premises, and `answer_link_rule` is a linking rule that maps that ground
    # fact to the configured answer head predicate.
    should_stop: bool
    is_final_solution: bool = False
    solution_premise_id: Optional[int] = None
    answer_link_rule: Optional[str] = None
    stop_reason: Optional[str] = None

    def __repr__(self) -> str:
        return (
            "SelectorDecision("
            f"selected_premise_ids={self.selected_premise_ids}, "
            f"proposed_new_premise={self.proposed_new_premise}, "
            f"background_premises={self.background_premises}, "
            f"is_answer_goal={self.is_answer_goal}, "
            f"is_final_solution={self.is_final_solution}, "
            f"solution_premise_id={self.solution_premise_id}, "
            f"answer_link_rule={self.answer_link_rule}, "
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


def _failed_step_groups_in_order(steps: List["PipelineStep"]) -> Dict[str, List["PipelineStep"]]:
    """Group failed steps by note, preserving first-seen note order."""
    groups: Dict[str, List["PipelineStep"]] = {}
    for s in steps:
        if s.success:
            continue
        key = s.note if s.note is not None else ""
        if key not in groups:
            groups[key] = []
        groups[key].append(s)
    return groups


def _initial_premises_for_report(final_premises: List[Premise]) -> List[Premise]:
    """Premises that are not pipeline-added (inference or selector background)."""
    return sorted(
        (
            p
            for p in final_premises
            if (p.source or "")
            not in (
                "inference",
                "selector_background",
                "termination_checker",
                "final_termination_check",
                "termination_checker_inference",
            )
        ),
        key=lambda p: p.id,
    )


def _append_new_premises_report_lines(
    lines: List[str],
    *,
    steps: List["PipelineStep"],
    final_premises: List[Premise],
) -> None:
    """
    Append New Premises section body: selector backgrounds (in step order), then
    inference results per successful step, using monotonic id assignment like the pipeline.
    """
    initial = _initial_premises_for_report(final_premises)
    cursor = max((p.id for p in initial), default=0)
    id_by_final = {p.id: p for p in final_premises}

    for step in sorted(steps, key=lambda s: s.step_index):
        for _ in step.decision.background_premises:
            cursor += 1
            p = id_by_final.get(cursor)
            if p is None:
                continue
            lines.append(
                f"{p.id} (from selector proposal, step {step.step_index}):"
            )
            body = f"  {format_clause(p.clause)}"
            if p.nl:
                body += f" # {p.nl}"
            lines.append(body)
        if step.new_premise is not None:
            np = step.new_premise
            cursor = np.id
            src = np.source or "inference"
            compact_id_list = ",".join(str(i) for i in step.used_premise_ids)
            prop = step.decision.proposed_new_premise
            lines.append(
                f"{np.id} (from {src}, [{compact_id_list}], "
                f"step {step.step_index}, proposed='{prop}'): "
            )
            body = f"  {format_clause(np.clause)}"
            if np.nl:
                body += f" # {np.nl}"
            lines.append(body)


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
        compact_id_list = ",".join(str(i) for i in self.used_premise_ids)
        return (
            f"Step {self.step_index}: Combined=[{compact_id_list}], "
            f"Proposed='{self.decision.proposed_new_premise}'."
        )


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

        lines.append("")
        original = _initial_premises_for_report(self.final_premises)
        lines.append("Original Premises:")
        lines.append(render_premises(original, verbosity_level=1))

        lines.append("")
        lines.append("New Premises:")
        _append_new_premises_report_lines(
            lines,
            steps=self.steps,
            final_premises=self.final_premises,
        )

        lines.append("")
        if self.answer_premise is not None:
            lines.append("Answer premise: " + self.answer_premise.str_verbose(level=3))
        else:
            lines.append("Answer premise: None")
        
        lines.append(f"Answer spec: {self.answer_spec}")
        lines.append("")
        lines.append("Failed step:")
        groups = list(_failed_step_groups_in_order(self.steps).items())
        if not groups:
            lines.append("  None")
        else:
            for gi, (note, group_steps) in enumerate(groups):
                lines.append(f"- {note}")
                for s in sorted(group_steps, key=lambda x: x.step_index):
                    lines.append(str(s))
        return "\n".join(lines)

    def extract_answer_constant(self) -> Optional[str]:
        """
        Determine which variable in the answer premise is the one of interest
        (from AnswerSpec), then return the constant bound to it by unification.

        Returns the constant's name (string) if the answer premise is a Fact
        that unifies with the answer_spec target and the distinguished variable
        is bound to a constant; otherwise None.
        """
        if self.answer_premise is None:
            return None
        clause = self.answer_premise.clause
        if not isinstance(clause, Fact):
            return None
        from .inference import unify_predicates

        subst = unify_predicates(self.answer_spec.target, clause.predicate)
        if subst is None:
            return None
        bound = subst.get(self.answer_spec.variable_name)
        if bound is None or bound.is_variable:
            return None
        return bound.name

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
    if token[0].isupper() or token.startswith("_"):
        return Term.variable(token)
    return Term.constant(token)


def _atom_literal_for_read_term_from_atom(text: str) -> str:
    """Embed *text* as a Prolog single-quoted atom literal (for read_term_from_atom/3)."""
    inner = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{inner}'"


def _get_parse_prolog():
    """Lazily construct one pySwip Prolog instance for parsing, or False if unavailable."""
    global _PARSE_PROLOG
    if _PARSE_PROLOG is not None:
        return _PARSE_PROLOG
    with _PARSE_PROLOG_LOCK:
        if _PySwipProlog is None:
            _PARSE_PROLOG = False
            return _PARSE_PROLOG
        try:
            _PARSE_PROLOG = _PySwipProlog()
        except Exception:
            _PARSE_PROLOG = False
        return _PARSE_PROLOG


def _prolog_atom_to_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _parse_predicate_arg_text(token: str) -> PredicateArg:
    token = token.strip()
    if not token:
        raise ValueError("Empty predicate argument")
    # Assumes nested predicates are in prefix format with brackets
    if "(" in token and token.endswith(")"):
        return parse_predicate(token)
    return _parse_term(token)


def _ensure_cogs402_parse_helpers(prolog) -> None:
    """
    Register small SWI predicates used only for parse_predicate.

    `term_string/2` turns variables into internal names (_NNN); we combine
    `read_term_from_atom/3` with `variable_names/1` so named variables keep
    their source spellings.
    """
    if getattr(prolog, "_cogs402_parse_helpers_loaded", False):
        return
    # Extend helpers to handle mathIs/is/2 predicates; handle arbitrary expressions by rewriting variables with VN names.
    prolog.assertz(
        "cogs402_arg_source_string(VN, Arg, Out) :- "
        "var(Arg), member(N=Var, VN), Arg == Var, !, Out = N"
    )
    # Special-case: if the argument is a recursive compound, we want to show the 
    # entire argument with variable names preserved; e.g. RHS of is/2, mathIs/2.
    prolog.assertz(
        "cogs402_arg_source_string(VN, Arg, Out) :- "
        "compound(Arg), "
        # "(compound_name_arity(Arg, is, 2); compound_name_arity(Arg, mathIs, 2)), "
        "Arg =.. [F, Lhs, Rhs], "
        "cogs402_arg_source_string(VN, Lhs, LhsS), "
        "cogs402_arg_source_string(VN, Rhs, RhsS), "
        "atomic_list_concat([F, '(', LhsS, ', ', RhsS, ')'], Out)"
    )
    # Default: use term_string if no special handling.
    prolog.assertz(
        "cogs402_arg_source_string(VN, Arg, Out) :- term_string(Arg, Out)"
    )
    prolog.assertz(
        "cogs402_args_source_strings(_, [], [])"
    )
    prolog.assertz(
        "cogs402_args_source_strings(VN, [A|As], [S|Ss]) :- "
        "cogs402_arg_source_string(VN, A, S), "
        "cogs402_args_source_strings(VN, As, Ss)"
    )
    prolog.assertz(
        "cogs402_parse_predicate_text(AtomStr, Name, ArgStrs) :- "
        "read_term_from_atom(AtomStr, T, "
        "[syntax_errors(error), variable_names(VN)]), "
        "( compound(T) -> compound_name_arguments(T, Name, Args) "
        "; atom(T) -> Name = T, Args = [] ), "
        "cogs402_args_source_strings(VN, Args, ArgStrs)"
    )
    prolog._cogs402_parse_helpers_loaded = True  # type: ignore[attr-defined]


def _parse_predicate_swi(text: str, prolog) -> Predicate:
    """
    Use SWI-Prolog's tokenizer/parser (read_term_from_atom/3) to obtain the
    functor and argument substrings, then map them into Predicate / Term.
    """
    _ensure_cogs402_parse_helpers(prolog)
    atom_lit = _atom_literal_for_read_term_from_atom(text)
    
    goal = f"cogs402_parse_predicate_text({atom_lit}, Name, ArgStrs)"
    solutions = list(prolog.query(goal, maxresult=1))
    if not solutions:
        raise ValueError(f"SWI-Prolog could not parse predicate string: {text!r}")
    sol = solutions[0]
    name = _prolog_atom_to_str(sol.get("Name")).strip()
    raw_strs = sol.get("ArgStrs")
    if raw_strs is None:
        raise ValueError(f"Unexpected SWI parse result for: {text!r}")
    if not isinstance(raw_strs, (list, tuple)):
        arg_strs = [raw_strs]
    else:
        arg_strs = list(raw_strs)
    arg_strs_py = [_prolog_atom_to_str(s).strip() for s in arg_strs]
    args = tuple(_parse_predicate_arg_text(s) for s in arg_strs_py)
    return Predicate(name=name, args=args)


def _parse_predicate_manual(text: str) -> Predicate:
    """
    Regex/comma-split fallback when SWI-Prolog is not available or fails.

    Supports only a limited subset: no nested function symbols or lists at
    the comma-split level. Prefer `_parse_predicate_swi` when possible.
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

    def _split_top_level_commas(raw: str) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        for ch in raw:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                if depth > 0:
                    depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                segment = "".join(current).strip()
                if segment:
                    parts.append(segment)
                current = []
            else:
                current.append(ch)
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    if "(" not in text:
        return Predicate(name=text, args=())

    name_part, rest = text.split("(", 1)
    name = name_part.strip()
    if not rest.endswith(")"):
        raise ValueError(f"Invalid predicate string (missing ')'): {text}")
    arg_str = rest[:-1]
    raw_args = _split_top_level_commas(arg_str)
    args = tuple(_parse_predicate_arg_text(a) for a in raw_args)
    # Also accept functional form `is(LHS, RHS_EXPR)` and normalize it.
    if name == "is":
        if len(args) != 2:
            raise ValueError(f"Invalid is/2 predicate arity: {text}")
        lhs, rhs = args
        if not isinstance(lhs, Term):
            raise ValueError(f"Invalid is/2 LHS expression: {text}")
        if rhs.is_variable:
            # `is/2` evaluates the RHS expression; a bare variable RHS is not
            # a supported expression in this project representation.
            raise ValueError(f"Invalid is/2 RHS expression: {text}")
        return Predicate(name="mathIs", args=(lhs, rhs))

    return Predicate(name=name, args=args)


def parse_predicate(text: str) -> Predicate:
    """
    Parse a predicate of the form `name(arg1, arg2, ...)` or `name`.

    When pySwip and SWI-Prolog are available, uses the engine's
    `read_term_from_atom/3` and `compound_name_arguments/3` so functor and
    arguments follow Prolog syntax (parentheses, commas, operators such as
    `is/2`). Otherwise falls back to a small manual parser.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty predicate string")

    backend = _get_parse_prolog()
    if backend is not False:
        try:
            return _parse_predicate_swi(text, backend)
        except Exception:
            pass
    return _parse_predicate_manual(text)


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


def _demo_pipeline_results() -> Tuple["PipelineResult", "PipelineResult"]:
    """
    Minimal PipelineResult pair exercising all __str__ sections (success vs failure).
    Run: python -m llm_prolog.symbolic.types
    """
    # --- Success: step 0 adds two background premises, then inference ---
    p1 = Premise(
        id=1,
        clause=Fact(predicate=Predicate("collected_cans", (Term.constant("144"),))),
        nl="Collected 144 cans.",
        source="nl_to_symbol",
    )
    p2 = Premise(
        id=2,
        clause=parse_fact_or_rule("reward(Count, R) :- mathIs(R, count * 2)."),
        nl="Reward rule.",
        source="nl_to_symbol",
    )
    p3 = Premise(
        id=3,
        clause=parse_fact_or_rule("bonus_one."),
        nl=None,
        source="selector_background",
    )
    p4 = Premise(
        id=4,
        clause=parse_fact_or_rule("bonus_two."),
        nl=None,
        source="selector_background",
    )
    p5 = Premise(
        id=5,
        clause=Fact(predicate=Predicate("reward", (Term.constant("144"), Term.constant("288")))),
        nl="Reward for 144 cans.",
        source="inference",
        parent_ids=[1, 2],
    )
    dec0 = SelectorDecision(
        selected_premise_ids=[1, 2],
        proposed_new_premise="reward(144, R)",
        background_premises=["bonus_one.", "bonus_two."],
        is_answer_goal=True,
        should_stop=False,
    )
    step0 = PipelineStep(
        step_index=0,
        used_premise_ids=[1, 2],
        new_premise=p5,
        decision=dec0,
        success=True,
        note=None,
    )
    answer_spec = AnswerSpec(target=parse_predicate("reward(144, R)"))
    success_result = PipelineResult(
        success=True,
        answer_premise=p5,
        steps=[step0],
        answer_spec=answer_spec,
        final_premises=[p1, p2, p3, p4, p5],
        reason="answer_head_matched",
    )

    # --- Failure: step 0 adds two backgrounds then skips inference; later failures ---
    q1 = Premise(
        id=1,
        clause=Fact(predicate=Predicate("fact_a", ())),
        nl="A.",
        source="nl_to_symbol",
    )
    q2 = Premise(
        id=2,
        clause=Fact(predicate=Predicate("fact_b", ())),
        nl="B.",
        source="nl_to_symbol",
    )
    q3 = Premise(
        id=3,
        clause=parse_fact_or_rule("bonus_one."),
        nl=None,
        source="selector_background",
    )
    q4 = Premise(
        id=4,
        clause=parse_fact_or_rule("bonus_two."),
        nl=None,
        source="selector_background",
    )
    q5 = Premise(
        id=5,
        clause=Fact(predicate=Predicate("merged", ())),
        nl="Merged.",
        source="inference",
        parent_ids=[1, 2],
    )
    d_bg = SelectorDecision(
        selected_premise_ids=[1],
        proposed_new_premise=None,
        background_premises=["bonus_one.", "bonus_two."],
        is_answer_goal=False,
        should_stop=False,
    )
    s_bg = PipelineStep(
        step_index=0,
        used_premise_ids=[1],
        new_premise=None,
        decision=d_bg,
        success=False,
        note="Selector did not choose two premises; skipping inference.",
    )
    d_ok = SelectorDecision(
        selected_premise_ids=[1, 2],
        proposed_new_premise="merged",
        background_premises=[],
        is_answer_goal=False,
        should_stop=False,
    )
    s_ok = PipelineStep(
        step_index=1,
        used_premise_ids=[1, 2],
        new_premise=q5,
        decision=d_ok,
        success=True,
        note=None,
    )
    note_reuse = "Inference step failed due to selecting premises already combined previously."
    d_fail = SelectorDecision(
        selected_premise_ids=[1, 2],
        proposed_new_premise="merged",
        background_premises=[],
        is_answer_goal=False,
        should_stop=False,
    )
    s_fail1 = PipelineStep(
        step_index=2,
        used_premise_ids=[1, 2],
        new_premise=None,
        decision=d_fail,
        success=False,
        note=note_reuse,
    )
    note_infer = "Inference failed to derive a new clause from selected premises."
    d_fail2 = SelectorDecision(
        selected_premise_ids=[2, 5],
        proposed_new_premise="nope(x)",
        background_premises=[],
        is_answer_goal=False,
        should_stop=False,
    )
    s_fail2 = PipelineStep(
        step_index=3,
        used_premise_ids=[2, 5],
        new_premise=None,
        decision=d_fail2,
        success=False,
        note=note_infer,
    )
    fail_spec = AnswerSpec(target=parse_predicate("answer(X)"))
    fail_result = PipelineResult(
        success=False,
        answer_premise=None,
        steps=[s_bg, s_ok, s_fail1, s_fail2],
        answer_spec=fail_spec,
        final_premises=[q1, q2, q3, q4, q5],
        reason="max_steps_exhausted",
    )
    return success_result, fail_result


if __name__ == "__main__":
    ok, bad = _demo_pipeline_results()
    print("=== demo success ===")
    print(ok)
    print("=== demo failure ===")
    print(bad)