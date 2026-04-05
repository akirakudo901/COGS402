"""
Interactive console editor for premises embedded in persisted PipelineResult rows.

Loads ``examples.jsonl`` from a run directory, lets the user revise NL→symbol premises
and ``final_termination_check`` premises, and appends one row per finished example to a JSONL file.
"""

from __future__ import annotations

import argparse
import builtins
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from eval.metrics.well_defined_nl_symbol import (
    nl_symbol_conversion_assess,
    premises_for_nl_symbol_validity_check,
)
from llm_prolog.symbolic.types import (
    PipelineResult,
    Premise,
    format_clause,
    initial_nl_symbol_premises_from_final,
    parse_fact_or_rule,
    render_premises,
)

_FINAL_TERMINATION_CHECK = "final_termination_check"

# Verbosity for list rendering (matches Premise.str_verbose levels).
_RENDER_VERBOSITY = 3

# User can type these at the premise-id prompt or field menu to run the NL→symbol well-defined check.
_NL_SYMBOL_CHECK_ALIASES = frozenset({"check", "wd"})

_FIELD_LABELS = (
    (1, "id", "integer premise id"),
    (2, "clause", "Prolog-like fact or rule (parsed with parse_fact_or_rule)"),
    (3, "nl", "optional natural language (empty → unset)"),
    (4, "source", "optional source string (empty → unset)"),
    (5, "parent_ids", "comma-separated ints, or empty for unset"),
)


def _combined_initial_and_ftc_premises(pr: PipelineResult) -> List[Premise]:
    """
    NL→symbol initial premises plus any ``final_termination_check`` premise(s), merged by id.
    """
    initial = initial_nl_symbol_premises_from_final(pr.final_premises)
    ftc = [p for p in pr.final_premises if (p.source or "") == _FINAL_TERMINATION_CHECK]
    by_id: Dict[int, Premise] = {p.id: p for p in initial}
    for p in ftc:
        by_id[p.id] = p
    return sorted(by_id.values(), key=lambda p: p.id)


def _print_reserved_final_premise_ids_line(pr: PipelineResult) -> None:
    """
    All ids in ``final_premises`` are occupied; a new premise cannot use any of these ids
    (including premises not shown in the combined NL→symbol + FTC view).
    """
    ids = sorted({p.id for p in pr.final_premises})
    if not ids:
        print(
            "Ids already in PipelineResult.final_premises "
            "(cannot add a new premise with any of these ids): (none)"
        )
        return
    joined = ", ".join(str(i) for i in ids)
    print(
        "Ids already in PipelineResult.final_premises "
        "(cannot add a new premise with any of these ids, including non-initial premises "
        f"not listed above): {joined}"
    )


def _print_combined_premises_render_and_reserved(pr: PipelineResult) -> None:
    print(
        render_premises(
            _combined_initial_and_ftc_premises(pr),
            verbosity_level=_RENDER_VERBOSITY,
        )
    )
    _print_reserved_final_premise_ids_line(pr)


def _load_examples_by_id(examples_jsonl: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for line in examples_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        eid = row.get("example_id")
        out[str(eid)] = row
    return out


def _parse_pipeline_result(row: Mapping[str, Any]) -> PipelineResult:
    payload = row.get("output")
    if not isinstance(payload, dict) or payload.get("result_type") != "PipelineResult":
        detail = repr(payload) if isinstance(payload, dict) else ""
        raise ValueError(
            "Row needs output.result_type == 'PipelineResult'; got "
            f"{type(payload).__name__} / {detail}"
        )
    return PipelineResult.from_json_dict(payload)


def _parse_parent_ids_line(line: str) -> Optional[List[int]]:
    s = line.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(x) for x in parts]


def _replace_premise_in_list(final_premises: List[Premise], updated: Premise) -> List[Premise]:
    return [updated if p.id == updated.id else p for p in final_premises]


def _remove_premise_id(final_premises: List[Premise], premise_id: int) -> List[Premise]:
    return [p for p in final_premises if p.id != premise_id]


def _append_or_replace_premise(final_premises: List[Premise], premise: Premise) -> List[Premise]:
    if any(p.id == premise.id for p in final_premises):
        return _replace_premise_in_list(final_premises, premise)
    return list(final_premises) + [premise]


def _apply_premise_change(
    pr: PipelineResult,
    *,
    old_premise_id: Optional[int],
    new_premise: Premise,
) -> PipelineResult:
    """
    Insert or replace a premise in ``final_premises``. If ``old_premise_id`` differs from
    ``new_premise.id``, remove the old id and add the new premise.
    """
    fps = list(pr.final_premises)
    if old_premise_id is not None and old_premise_id != new_premise.id:
        fps = _remove_premise_id(fps, old_premise_id)
        if any(p.id == new_premise.id for p in fps):
            raise ValueError(f"Premise id {new_premise.id} already exists in final_premises.")
        fps.append(new_premise)
    else:
        fps = _append_or_replace_premise(fps, new_premise)
    return replace(pr, final_premises=sorted(fps, key=lambda p: p.id))


def _print_field_menu(pr: PipelineResult) -> None:
    print("="*30)
    _print_combined_premises_render_and_reserved(pr)
    print("="*30)
    print("Fields (enter number):")
    for num, name, desc in _FIELD_LABELS:
        print(f"  {num}. {name} — {desc}")


def _prompt_preview_confirm(
    premise: Premise,
    read_fn: Callable[[str], str],
) -> bool:
    print()
    print("Preview (verbosity 3):")
    print(premise.str_verbose(level=_RENDER_VERBOSITY))
    print()
    while True:
        ans = read_fn('Reflect this in the pipeline result? Type "yes" or "no": ').strip().lower()
        if ans in ("yes", "y"):
            return True
        if ans in ("no", "n"):
            return False
        print('Please type "yes" or "no".')


def _read_new_premise_interactive(
    premise_id: int,
    read_fn: Callable[[str], str],
) -> Premise:
    print(f"Creating new premise id={premise_id}. Enter each field in order.")
    clause_s = read_fn("Clause (Prolog-like, parsed with parse_fact_or_rule): ").strip()
    clause = parse_fact_or_rule(clause_s)
    nl_raw = read_fn("nl (empty for unset): ").strip()
    nl: Optional[str] = nl_raw if nl_raw else None
    src_raw = read_fn("source (empty for unset): ").strip()
    source: Optional[str] = src_raw if src_raw else None
    parents_raw = read_fn("parent_ids (comma-separated ints, empty for unset): ")
    parent_ids = _parse_parent_ids_line(parents_raw)
    return Premise(
        id=premise_id,
        clause=clause,
        nl=nl,
        source=source,
        parent_ids=parent_ids,
    )


def _parent_ids_default_for_edit(parent_ids: Optional[List[int]]) -> str:
    """Literal text to place in the edit buffer (not the '(unset)' label)."""
    if not parent_ids:
        return ""
    return ", ".join(str(x) for x in parent_ids)


def _read_line_with_editable_default(
    read_fn: Callable[[str], str],
    prompt: str,
    default: str,
) -> str:
    """
    Read one line of input. When ``read_fn`` is the real :func:`input` and ``default``
    has no newlines, pre-fill the readline buffer (GNU readline / libedit) so the
    current value is editable in place before Enter.

    Multiline defaults, missing readline, or a custom ``read_fn`` (e.g. tests) fall
    back to printing the current value then reading a replacement line.
    Editable pre-fill requires passing the built-in ``input`` (not a wrapper).
    """
    if read_fn is not builtins.input:
        return read_fn(f"{prompt}\n[default: {default!r}]\n").strip()

    if "\n" in default:
        print(prompt)
        print("--- current ---")
        print(default, end="" if default.endswith("\n") else "\n")
        print("---")
        return input("New value (empty to clear): ").strip()

    try:
        import readline as readline_mod
    except ImportError:
        readline_mod = None

    if readline_mod is None:
        print(prompt)
        print(f"(readline unavailable; current value: {default!r})")
        return input("New value: ").strip()

    def _hook() -> None:
        readline_mod.insert_text(default)

    readline_mod.set_startup_hook(_hook)
    try:
        return input(prompt).strip()
    except Exception:
        print(prompt)
        print(f"(could not pre-fill; current value: {default!r})")
        return input("New value: ").strip()
    finally:
        readline_mod.set_startup_hook(None)


def _edit_existing_field(
    premise: Premise,
    field_num: int,
    read_fn: Callable[[str], str],
) -> Tuple[Premise, Optional[int]]:
    """
    Returns (updated_premise, old_id_if_id_changed_else_None).
    """
    if field_num == 1:
        raw = _read_line_with_editable_default(
            read_fn,
            "New id (integer): ",
            str(premise.id),
        )
        new_id = int(raw)
        if new_id == premise.id:
            return premise, None
        return replace(premise, id=new_id), premise.id
    if field_num == 2:
        cur_clause = format_clause(premise.clause)
        raw = _read_line_with_editable_default(
            read_fn,
            "New clause (Prolog-like): ",
            cur_clause,
        )
        return replace(premise, clause=parse_fact_or_rule(raw)), None
    if field_num == 3:
        raw = _read_line_with_editable_default(
            read_fn,
            "nl (empty for unset): ",
            premise.nl if premise.nl is not None else "",
        )
        return replace(premise, nl=raw if raw else None), None
    if field_num == 4:
        raw = _read_line_with_editable_default(
            read_fn,
            "source (empty for unset): ",
            premise.source if premise.source is not None else "",
        )
        return replace(premise, source=raw if raw else None), None
    if field_num == 5:
        raw = _read_line_with_editable_default(
            read_fn,
            "parent_ids (comma-separated ints, empty for unset): ",
            _parent_ids_default_for_edit(premise.parent_ids),
        )
        return replace(premise, parent_ids=_parse_parent_ids_line(raw)), None
    raise ValueError(f"Unknown field number: {field_num}")


def _print_well_defined_symbol_check(row: Mapping[str, Any], pr: PipelineResult) -> None:
    """
    Run the same SWI-Prolog well-defined check as metrics (``premises_for_nl_symbol_validity_check``)
    and print ground truth, obtained answer when available, success, and failure category / details.
    """
    premises = premises_for_nl_symbol_validity_check(pr)
    ground_truth = row.get("ground_truth")
    outcome = nl_symbol_conversion_assess(premises, pr.answer_spec, ground_truth)
    print()
    print("--- Well-defined symbol check ---")
    print(f"ground_truth: {ground_truth!r}")
    if "success" in row:
        print(f"artifact row success (task): {row.get('success')!r}")
    if outcome.obtained_answer is not None:
        print(f"obtained (unique normalized answer): {outcome.obtained_answer!r}")
    print(f"well-defined check success: {outcome.ok}")
    print(f"category: {outcome.category.value}")
    if outcome.detail:
        print(f"detail: {outcome.detail}")
    if outcome.message:
        print(f"message: {outcome.message}")
    print("---------------------------------")
    print()


def _prompt_optional_well_defined_check(
    row: Mapping[str, Any],
    pr: PipelineResult,
    read_fn: Callable[[str], str],
) -> None:
    while True:
        ans = read_fn(
            'Run well-defined symbol check before the next example? Type "yes" or "no": '
        ).strip().lower()
        if ans in ("yes", "y"):
            _print_well_defined_symbol_check(row, pr)
            return
        if ans in ("no", "n"):
            return
        print('Please type "yes" or "no".')


def _append_jsonl_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def interactive_edit_premises(
    examples_dir: Union[str, Path],
    example_ids: Sequence[Union[int, str]],
    destination_jsonl: Union[str, Path],
    *,
    read_fn: Optional[Callable[[str], str]] = None,
) -> None:
    """
    For each ``example_id``, load ``examples_dir / "examples.jsonl"``, deserialize
    ``PipelineResult``, show initial NL→symbol premises plus ``final_termination_check``
    premises (merged by id), and interactively edit. Accepted changes update the in-memory
    row only; the full example row (with updated ``output``) is appended to
    ``destination_jsonl`` once when the user finishes that example (``done``). At any time
    while choosing a premise id or a field, the user may type ``check`` or ``wd`` to run
    the NL→symbol well-defined check on the current premises.

    Parameters
    ----------
    examples_dir:
        Directory containing ``examples.jsonl``.
    example_ids:
        Examples to process in order.
    destination_jsonl:
        JSONL file to append one row per example when the user types ``done``.
    read_fn:
        Optional replacement for ``input`` (for tests).
    """
    ex_dir = Path(examples_dir).resolve()
    ex_path = ex_dir / "examples.jsonl"
    if not ex_path.is_file():
        raise FileNotFoundError(f"Missing examples.jsonl: {ex_path}")

    dest = Path(destination_jsonl).resolve()
    by_id = _load_examples_by_id(ex_path)
    read = read_fn or input

    for raw_eid in example_ids:
        key = str(raw_eid)
        row = by_id.get(key)
        if row is None:
            print(f"[skip] example_id={raw_eid!r} not found in {ex_path}")
            continue

        try:
            pr = _parse_pipeline_result(row)
        except ValueError as e:
            print(f"[skip] example_id={raw_eid!r}: {e}")
            continue

        print()
        print("=" * 72)
        print(f"example_id={row.get('example_id')}")
        print("Initial premises + final termination check premise(s) (verbosity 3):")
        _print_combined_premises_render_and_reserved(pr)
        print("=" * 72)

        combined_ids = {p.id for p in _combined_initial_and_ftc_premises(pr)}

        while True:
            raw = read(
                'Premise id to edit / add, "check" (or "wd") for NL symbol well-defined check, '
                'or "done" for next example: '
            ).strip()
            low = raw.lower()
            if low == "done":
                row["output"] = pr.to_json_dict()
                out_row = dict(row)
                _append_jsonl_row(dest, out_row)
                by_id[key] = out_row
                print(f"Appended snapshot to {dest}")
                break
            if low in _NL_SYMBOL_CHECK_ALIASES:
                _print_well_defined_symbol_check(row, pr)
                continue

            try:
                pid = int(raw)
            except ValueError:
                print(
                    "Enter an integer premise id, 'check' (or 'wd') for NL symbol check, or 'done'."
                )
                continue

            if pid not in combined_ids:
                # New premise: collect all fields in order, then confirm.
                if any(p.id == pid for p in pr.final_premises):
                    print(f"Id {pid} exists in final_premises but not in the displayed combined list; refusing.")
                    continue
                try:
                    candidate = _read_new_premise_interactive(pid, read)
                except Exception as e:
                    print(f"Invalid input: {e}")
                    continue
                if _prompt_preview_confirm(candidate, read):
                    pr = _apply_premise_change(pr, old_premise_id=None, new_premise=candidate)
                    combined_ids.add(pid)
                    row["output"] = pr.to_json_dict()
                    print()
                    print("Updated list (verbosity 3):")
                    _print_combined_premises_render_and_reserved(pr)
                continue

            # Existing premise in combined view: locate full Premise from final_premises.
            current_list = [p for p in pr.final_premises if p.id == pid]
            if not current_list:
                print(f"No premise with id {pid} in final_premises.")
                continue
            current = current_list[0]

            while True:
                _print_field_menu(pr)
                field_raw = read(
                    "Field number (or 'back' to pick another id, 'check' / 'wd' for NL symbol check): "
                ).strip().lower()
                if field_raw == "back":
                    break
                if field_raw in _NL_SYMBOL_CHECK_ALIASES:
                    _print_well_defined_symbol_check(row, pr)
                    continue
                try:
                    field_num = int(field_raw)
                except ValueError:
                    print("Enter 1–5, 'back', or 'check' / 'wd'.")
                    continue
                if field_num not in range(1, 6):
                    print("Enter a number from 1 to 5.")
                    continue

                try:
                    candidate, old_id = _edit_existing_field(current, field_num, read)
                except Exception as e:
                    print(f"Invalid input: {e}")
                    continue

                if _prompt_preview_confirm(candidate, read):
                    pr = _apply_premise_change(
                        pr,
                        old_premise_id=old_id,
                        new_premise=candidate,
                    )
                    if old_id is not None:
                        combined_ids.discard(old_id)
                    combined_ids.add(candidate.id)
                    row["output"] = pr.to_json_dict()
                    current = candidate
                # On 'no', keep editing this premise (same id unless id was unchanged).


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Interactive premise editor for PipelineResult JSONL rows.")
    p.add_argument(
        "examples_dir",
        type=Path,
        help="Directory containing examples.jsonl",
    )
    p.add_argument(
        "destination_jsonl",
        type=Path,
        help="JSONL file to append confirmed rows to",
    )
    p.add_argument(
        "example_ids",
        nargs="+",
        help="example_id values to process, in order",
    )
    args = p.parse_args(argv)
    interactive_edit_premises(
        args.examples_dir,
        args.example_ids,
        args.destination_jsonl,
    )


if __name__ == "__main__":
    main()
