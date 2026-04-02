"""
Metrics for NL→symbol conversion quality vs SWI-Prolog and the hybrid selector.

*Well-defined* (for one example): load the premises returned by
``premises_for_nl_symbol_validity_check`` — NL→symbol premises, plus (only when
``PipelineResult.success``) any verified rule(s) the **post-loop** final
termination checker induced (``source == "final_termination_check"``). Querying
SWI-Prolog for the AnswerSpec goal must yield a **unique** binding matching
ground truth. The FTC rule is omitted when the run failed, since it may be wrong.

*Failed under well-defined symbols*: NL→symbol is well-defined, but the pipeline
did not finish with ``PipelineResult.success``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Set, Union

from llm_prolog.symbolic.inference import _predicate_to_prolog_goal_text
from llm_prolog.symbolic.types import (
    AnswerSpec,
    Clause,
    Fact,
    PipelineResult,
    Premise,
    Rule,
)

try:
    from pyswip import Prolog as _PySwipProlog  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _PySwipProlog = None


# Premises produced by the NL→symbol step use this source; legacy/empty also treated as initial.
_NL_SYMBOL_SOURCES = frozenset({"nl_symbol_converter", "nl_to_symbol", ""})

# Verified linking rule appended when the post-loop final termination check succeeds.
_FINAL_TERMINATION_CHECK_SOURCE = "final_termination_check"

# Safety cap: if this many distinct answer bindings appear, treat as not well-defined.
_MAX_ANSWER_BINDINGS = 10_000

# Cap each SWI-Prolog query duration to avoid pathological non-termination.
_SWIPL_QUERY_TIME_LIMIT_SECONDS = 5


def _pyswip_load_clp_extensions(prolog: Any) -> None:
    """
    Load SWI-Prolog CLP libraries so constraint arithmetic is available.

    ``library(clpfd)`` provides finite-domain constraints (e.g. ``#=/2``).
    ``library(clpqr)`` provides CLP over rationals and reals (e.g. ``{}/1``).
    Each load is best-effort so a missing optional install does not break metrics.
    """
    for q in (
        "use_module(library(clpfd))",
        "use_module(library(clpqr))",
    ):
        try:
            list(prolog.query(q, maxresult=1))
        except Exception:
            pass


def _pyswip_prepare_embedded_engine(prolog: Any) -> None:
    """
    Batch-style SWI-Prolog settings for PySwip.

    With the default ``debug_on_error`` flag, syntax/load errors during ``assertz``
    can start the interactive tracer (``Call: ... ?``) and block on stdin.
    Disabling that flag keeps failures non-interactive so Python can treat them
    as errors and return.
    """
    try:
        list(prolog.query("set_prolog_flag(debug_on_error, false)", maxresult=1))
    except Exception:
        pass
    _pyswip_load_clp_extensions(prolog)


def _pyswip_retract_clause(prolog: Any, clause_text: str) -> None:
    """
    Undo one ``assertz(clause_text)`` on the shared embedded engine.

    PySwip attaches to a process-wide SWI-Prolog database, so clauses survive
    across ``Prolog()`` constructors unless retracted. Rules must be passed to
    ``retract/1`` as a single term: ``retract((Head :- Body))``.
    """
    if ":- " in clause_text:
        q = f"catch(retract(({clause_text})), _, true)"
    else:
        q = f"catch(retract({clause_text}), _, true)"
    try:
        list(prolog.query(q, maxresult=1))
    except Exception:
        pass


def initial_premises_from_nl_symbol_converter(premises: Sequence[Premise]) -> List[Premise]:
    """
    Premises whose ``source`` marks them as coming from the NL→symbol converter
    (same convention as ``PipelineResult`` string rendering).
    """
    return sorted(
        (p for p in premises if (p.source or "") in _NL_SYMBOL_SOURCES),
        key=lambda p: p.id,
    )


def premises_for_nl_symbol_validity_check(result: PipelineResult) -> List[Premise]:
    """
    Premises used for the SWI-Prolog “well-defined” check on an artifact row.

    Starts from NL→symbol premises. If ``result.success`` is True, also includes
    any premise whose ``source`` is ``final_termination_check`` (the verified rule
    the post-loop final termination checker induced). If the pipeline did not
    succeed, FTC premises are **not** included — they may be incorrect.
    """
    base = initial_premises_from_nl_symbol_converter(result.final_premises)
    if not result.success:
        return list(base)
    extras = [
        p
        for p in result.final_premises
        if (p.source or "") == _FINAL_TERMINATION_CHECK_SOURCE
    ]
    if not extras:
        return list(base)
    by_id = {p.id: p for p in base}
    for p in sorted(extras, key=lambda x: x.id):
        by_id[p.id] = p
    return sorted(by_id.values(), key=lambda p: p.id)


def _clause_to_swipl_clause_text(clause: Clause) -> str:
    """Render a stored clause as SWI-Prolog source (``mathIs/2`` → ``is/2`` in bodies)."""
    if isinstance(clause, Fact):
        return _predicate_to_prolog_goal_text(clause.predicate) + "."
    if isinstance(clause, Rule):
        if not clause.body:
            return _predicate_to_prolog_goal_text(clause.head) + "."
        head = clause.head.to_prolog_text()
        body = ", ".join(_predicate_to_prolog_goal_text(p) for p in clause.body)
        return f"{head} :- {body}."
    raise TypeError(f"Unsupported clause: {type(clause)}")


def _normalize_answer_string(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, float):
        if math.isfinite(raw) and raw == round(raw):
            return str(int(raw))
        return repr(raw)
    if isinstance(raw, int):
        return str(raw)
    s = str(raw).strip()
    try:
        v = float(s)
        if math.isfinite(v) and v == round(v):
            return str(int(v))
    except (TypeError, ValueError):
        pass
    return s


def _ground_truth_matches_value(ground_truth: Any, prolog_answer: str) -> bool:
    if ground_truth is None:
        return False
    got = _normalize_answer_string(prolog_answer)
    if got == "":
        return False

    exp = _normalize_answer_string(ground_truth)
    # TODO DEBUG REMOVE
    print(f"Post-normalization, actual: {got}, expected: {exp}")
    # TODO DEBUG REMOVE END
    if exp == got:
        return True

    try:
        gf = float(ground_truth)
        ga = float(got)
        if math.isfinite(gf) and math.isfinite(ga):
            return math.isclose(gf, ga, rel_tol=0.0, abs_tol=1e-6)
    except (TypeError, ValueError):
        pass

    return str(ground_truth).strip() == got


def _binding_for_var(solution: dict, var_name: str) -> Any:
    if var_name in solution:
        return solution[var_name]
    for k, v in solution.items():
        if str(k) == var_name:
            return v
    return None


def nl_symbol_conversion_is_well_defined(
    initial_premises: Sequence[Premise],
    answer_spec: AnswerSpec,
    ground_truth: Any,
) -> bool:
    """
    Return True iff SWI-Prolog, given only ``initial_premises`` as the program,
    proves the AnswerSpec goal with a **unique** binding for the distinguished
    variable, and that binding matches ``ground_truth``.

    If pySwip/SWI-Prolog is unavailable or the program/query errors, returns False.
    """
    if _PySwipProlog is None:
        return False
    
    texts = [_clause_to_swipl_clause_text(p.clause) for p in initial_premises]
    goal = _predicate_to_prolog_goal_text(answer_spec.target)
    var_name = answer_spec.variable_name

    prolog = _PySwipProlog()
    _pyswip_prepare_embedded_engine(prolog)
    assert_stack: List[str] = []
    try:
        # TODO DEBUG REMOVE
        print("Asserted:")
        # TODO DEBUG REMOVE END
        for t in texts:
            clause_text = t.strip()
            if clause_text.endswith('.'):
                clause_text = clause_text[:-1]
            # TODO DEBUG REMOVE
            print(f"  {clause_text}")
            # TODO DEBUG REMOVE END
            prolog.assertz(clause_text)
            assert_stack.append(clause_text)
        # TODO DEBUG REMOVE
        print(f"goal:{goal}")
        # TODO DEBUG REMOVE END
        timed_goal = f"call_with_time_limit({_SWIPL_QUERY_TIME_LIMIT_SECONDS}, {goal})"
        solutions = list(prolog.query(timed_goal, maxresult=_MAX_ANSWER_BINDINGS + 1))
    except Exception as e:
        # TODO DEBUG REMOVE
        print(f"Failing query due to exception: {e}")
        # TODO DEBUG REMOVE END
        return False
    finally:
        for clause_text in reversed(assert_stack):
            _pyswip_retract_clause(prolog, clause_text)

    if not solutions or len(solutions) > _MAX_ANSWER_BINDINGS:
        # TODO DEBUG REMOVE
        print(
            f"We have no solution: {solutions}" if not solutions else
            f"We have too many solutions: {len(solutions)} vs. {_MAX_ANSWER_BINDINGS} allowed"
            )
        # TODO DEBUG REMOVE END
        return False

    distinct: Set[str] = set()
    for sol in solutions:
        raw = _binding_for_var(sol, var_name)
        if raw is None:
            # TODO DEBUG REMOVE
            print(f"We failed to obtain the value bound to {var_name} within {sol}")
            # TODO DEBUG REMOVE END
            return False
        distinct.add(_normalize_answer_string(raw))

    if len(distinct) != 1:
        # TODO DEBUG REMOVE
        print(f"We don't have a distinct answer: {distinct}")
        # TODO DEBUG REMOVE END
        return False

    only = next(iter(distinct))
    # TODO DEBUG REMOVE
    print(f"Pre-normalization, actual: {only}, expected: {ground_truth}")
    # TODO DEBUG REMOVE END
    return _ground_truth_matches_value(ground_truth, only)


def selector_failed_under_well_defined_symbols(
    nl_symbol_well_defined: bool,
    pipeline_success: bool,
) -> bool:
    """
    True when the NL→symbol step left a unique Prolog answer matching ground truth,
    but the hybrid pipeline still did not succeed (selector + inference path).
    """
    return nl_symbol_well_defined and not pipeline_success


@dataclass(frozen=True)
class WellDefinedNlSymbolSummary:
    """Aggregated counts from ``summarize_well_defined_nl_symbol_metrics``."""

    total_rows: int
    symbolic_hybrid_rows: int
    skipped_rows: int
    well_defined_count: int
    not_well_defined_count: int
    failed_under_well_defined_symbols_count: int
    prolog_unavailable: bool
    final_termination_checker_premise_rows: int
    well_defined_only_after_including_ftc_premise_count: int


def summarize_well_defined_nl_symbol_metrics(
    artifact_dir: Union[str, Path],
) -> WellDefinedNlSymbolSummary:
    """
    Read ``artifact_dir / examples.jsonl`` (eval artifact layout), and for each
    symbolic-hybrid ``PipelineResult`` tally well-defined symbol validity (using
    ``premises_for_nl_symbol_validity_check``), selector failure under
    well-defined symbols, FTC rule rows, and rows where adding the FTC rule was
    necessary for a well-defined verdict.

    Rows without a deserializable ``PipelineResult`` in ``output`` are skipped
    (counted in ``skipped_rows``).
    """
    artifact_dir = Path(artifact_dir)
    path = artifact_dir / "examples.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"examples.jsonl not found under {artifact_dir!s}")

    prolog_unavailable = _PySwipProlog is None

    total_rows = 0
    symbolic_hybrid_rows = 0
    skipped = 0
    well_defined_n = 0
    not_well_defined_n = 0
    failed_under_wd = 0
    ftc_premise_rows = 0
    wd_flipped_by_ftc = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total_rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        out = row.get("output")
        if not isinstance(out, dict) or out.get("result_type") != "PipelineResult":
            skipped += 1
            continue

        try:
            result = PipelineResult.from_json_dict(out)
        except Exception:
            skipped += 1
            continue

        symbolic_hybrid_rows += 1
        ground_truth = row.get("ground_truth")
        nl_only = initial_premises_from_nl_symbol_converter(result.final_premises)
        check_premises = premises_for_nl_symbol_validity_check(result)

        has_ftc_premise = any(
            (p.source or "") == _FINAL_TERMINATION_CHECK_SOURCE
            for p in result.final_premises
        )
        if has_ftc_premise:
            ftc_premise_rows += 1

        wd = nl_symbol_conversion_is_well_defined(
            check_premises,
            result.answer_spec,
            ground_truth,
        )
        # TODO DEBUG REMOVE
        print(f"wd: {wd}")
        print("~"*20)
        # TODO DEBUG REMOVE END
        if result.success and has_ftc_premise:
            wd_nl = nl_symbol_conversion_is_well_defined(
                nl_only,
                result.answer_spec,
                ground_truth,
            )
            if not wd_nl and wd:
                wd_flipped_by_ftc += 1

        if wd:
            well_defined_n += 1
        else:
            not_well_defined_n += 1

        if selector_failed_under_well_defined_symbols(wd, result.success):
            failed_under_wd += 1

    return WellDefinedNlSymbolSummary(
        total_rows=total_rows,
        symbolic_hybrid_rows=symbolic_hybrid_rows,
        skipped_rows=skipped,
        well_defined_count=well_defined_n,
        not_well_defined_count=not_well_defined_n,
        failed_under_well_defined_symbols_count=failed_under_wd,
        prolog_unavailable=prolog_unavailable,
        final_termination_checker_premise_rows=ftc_premise_rows,
        well_defined_only_after_including_ftc_premise_count=wd_flipped_by_ftc,
    )

if __name__ == "__main__":
    from pathlib import Path

    path = Path(
        r"/Users/akirakudo/Desktop/code/python/Class/2025W2/COGS402/artifacts/where-it-breaks-50EXS/phase-1/run_20260331_204856_102463ab_gpt-5-mini"
        )
    path2 = Path(
        r"/Users/akirakudo/Desktop/code/python/Class/2025W2/COGS402/artifacts/where-it-breaks-50EXS/phase-1/run_20260331_205112_233c02f1_gpt-4.1-mini"
    )

    summary = summarize_well_defined_nl_symbol_metrics(path2)
    print(summary)