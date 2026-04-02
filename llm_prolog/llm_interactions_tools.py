"""
Save / load pipeline `llm_interactions` and inspect LLM prompts per component.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO

# Trace `component` values (see nl_symbol_converter, symbol_nl_converter, selector, cot_baseline).
NL_TO_SYMBOL_COMPONENT = "nl_to_symbol"
SYMBOL_TO_NL_COMPONENT = "symbol_to_nl"
SELECTOR_COMPONENT = "selector_select_next_step"
FINAL_TERMINATION_CHECK_COMPONENT = "final_termination_check"
COT_SOLVER_COMPONENT = "cot_solver"


def save_llm_interactions_to_json(
    interactions: Optional[Iterable[Dict[str, Any]]],
    path: str | Path,
) -> None:
    """
    Persist `PipelineResult.llm_interactions` (or any equivalent list of trace dicts)
    to a JSON file for later reload with `load_llm_interactions_from_json`.
    """
    p = Path(path)
    payload = list(interactions) if interactions is not None else []
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_llm_interactions_from_json(path: str | Path) -> List[Dict[str, Any]]:
    """Reload interactions written by `save_llm_interactions_to_json`."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"interactions[{i}] must be an object, got {type(item).__name__}")
        out.append(item)
    return out


def _filter_component(
    interactions: Optional[Iterable[Dict[str, Any]]],
    component: str,
) -> List[Dict[str, Any]]:
    if interactions is None:
        return []
    return [
        it
        for it in interactions
        if isinstance(it, dict) and it.get("component") == component
    ]


def _print_prompt_sections(
    *,
    title: str,
    system_prompt: str,
    user_prompt: str,
    stream: TextIO,
    extra_sections: Optional[List[tuple[str, str]]] = None,
) -> None:
    sep = "=" * 72
    print(sep, file=stream)
    print(title, file=stream)
    print(sep, file=stream)
    print("--- system_prompt ---", file=stream)
    print(system_prompt, file=stream)
    print("--- user_prompt ---", file=stream)
    print(user_prompt, file=stream)
    if extra_sections:
        for heading, body in extra_sections:
            print(f"--- {heading} ---", file=stream)
            print(body, file=stream)
    print(file=stream)


def _display_component_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
    component: str,
    *,
    section_header: str,
    empty_message: str,
    label_fn: Callable[[int, Dict[str, Any]], str],
    stream: TextIO,
    extra_sections_fn: Optional[Callable[[Dict[str, Any]], List[tuple[str, str]]]] = None,
) -> None:
    entries = _filter_component(interactions, component)
    if not entries:
        print(empty_message, file=stream)
        return
    for i, it in enumerate(entries):
        title = f"{section_header} {label_fn(i, it)}"
        extra = extra_sections_fn(it) if extra_sections_fn else None
        _print_prompt_sections(
            title=title,
            system_prompt=str(it.get("system_prompt", "")),
            user_prompt=str(it.get("user_prompt", "")),
            stream=stream,
            extra_sections=extra,
        )


def extract_nl_to_symbol_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for it in _filter_component(interactions, NL_TO_SYMBOL_COMPONENT):
        rows.append(
            {
                "system_prompt": str(it.get("system_prompt", "")),
                "user_prompt": str(it.get("user_prompt", "")),
            }
        )
    return rows


def display_nl_to_symbol_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
    *,
    stream: TextIO = sys.stdout,
) -> None:
    """Print prompts for traces with ``component == 'nl_to_symbol'``."""

    def label(i: int, _: Dict[str, Any]) -> str:
        return f"call #{i}"

    _display_component_prompts(
        interactions,
        NL_TO_SYMBOL_COMPONENT,
        section_header="nl_to_symbol",
        empty_message="(no nl_to_symbol entries in interactions)",
        label_fn=label,
        stream=stream,
    )


def extract_symbol_to_nl_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for it in _filter_component(interactions, SYMBOL_TO_NL_COMPONENT):
        rows.append(
            {
                "system_prompt": str(it.get("system_prompt", "")),
                "user_prompt": str(it.get("user_prompt", "")),
            }
        )
    return rows


def display_symbol_to_nl_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
    *,
    stream: TextIO = sys.stdout,
) -> None:
    """Print prompts for traces with ``component == 'symbol_to_nl'``."""

    def label(i: int, _: Dict[str, Any]) -> str:
        return f"call #{i}"

    _display_component_prompts(
        interactions,
        SYMBOL_TO_NL_COMPONENT,
        section_header="symbol_to_nl",
        empty_message="(no symbol_to_nl entries in interactions)",
        label_fn=label,
        stream=stream,
    )


def extract_selector_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Return one record per selector LLM call, in pipeline order.

    Each record includes `step_index` (when present), `system_prompt`, and
    `user_prompt` from traces whose `component` is `selector_select_next_step`.
    """
    rows: List[Dict[str, Any]] = []
    for it in _filter_component(interactions, SELECTOR_COMPONENT):
        rows.append(
            {
                "step_index": it.get("step_index"),
                "system_prompt": str(it.get("system_prompt", "")),
                "user_prompt": str(it.get("user_prompt", "")),
            }
        )
    return rows


def display_selector_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
    *,
    stream: TextIO = sys.stdout,
) -> None:
    """
    Print system and user prompts sent to the selector at each symbolic step.

    Uses traces with ``component == 'selector_select_next_step'`` (see
    `llm_prolog.selector` and `pipeline._run_symbolic_steps`).
    """

    def label(_i: int, it: Dict[str, Any]) -> str:
        si = it.get("step_index")
        return f"step_index={si}" if si is not None else f"call #{_i}"

    _display_component_prompts(
        interactions,
        SELECTOR_COMPONENT,
        section_header="selector",
        empty_message="(no selector_select_next_step entries in interactions)",
        label_fn=label,
        stream=stream,
    )


def _selector_traces_in_step_order(
    interactions: Optional[Iterable[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Selector traces sorted by ``step_index`` (missing values sort last, preserve order)."""
    rows = _filter_component(interactions, SELECTOR_COMPONENT)
    if not rows:
        return []

    def sort_key(i: int, it: Dict[str, Any]) -> tuple[int, int]:
        si = it.get("step_index")
        if isinstance(si, int):
            return (0, si)
        try:
            return (0, int(si))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return (1, i)

    indexed = list(enumerate(rows))
    indexed.sort(key=lambda pair: sort_key(pair[0], pair[1]))
    return [pair[1] for pair in indexed]


def display_selector_user_prompt_prefix_deltas(
    interactions: Optional[Iterable[Dict[str, Any]]],
    *,
    stream: TextIO = sys.stdout,
) -> None:
    """
    Print selector ``user_prompt`` in order: full text for the first step, then only
    the suffix added at each later step when the previous step's ``user_prompt`` is
    a prefix of the current one (as in ``append_cache`` / ``SelectorPromptSession``).

    If step *i*'s prompt is not prefixed by step *i-1*'s, prints a notice and the
    full ``user_prompt`` for step *i* (e.g. ``legacy_rebuild`` full rebuilds).
    ``system_prompt`` is printed once from the first trace (and again if it changes).
    """
    traces = _selector_traces_in_step_order(interactions)
    if not traces:
        print("(no selector_select_next_step entries in interactions)", file=stream)
        return

    sep = "=" * 72
    print(sep, file=stream)
    print("selector user_prompt — base + per-step deltas (prefix extension)", file=stream)
    print(sep, file=stream)

    prev_system: Optional[str] = None
    prev_user: Optional[str] = None

    for i, it in enumerate(traces):
        si = it.get("step_index")
        step_label = f"step_index={si}" if si is not None else f"ordinal={i}"
        sys_p = str(it.get("system_prompt", ""))
        usr = str(it.get("user_prompt", ""))

        if prev_system is None or sys_p != prev_system:
            print("--- system_prompt ---", file=stream)
            print(sys_p, file=stream)
            prev_system = sys_p

        if i == 0:
            print(f"--- user_prompt BASE ({step_label}) — full initial ---", file=stream)
            print(usr, file=stream)
        else:
            assert prev_user is not None
            if usr.startswith(prev_user):
                delta = usr[len(prev_user) :]
                print(
                    f"--- user_prompt DELTA ({step_label}) — suffix after step i-1 ---",
                    file=stream,
                )
                print(delta, file=stream)
            else:
                print(
                    f"--- user_prompt ({step_label}) — previous user_prompt is NOT a prefix; "
                    f"full prompt follows ---",
                    file=stream,
                )
                print(usr, file=stream)
        print(file=stream)
        prev_user = usr


def extract_final_termination_check_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for it in _filter_component(interactions, FINAL_TERMINATION_CHECK_COMPONENT):
        rows.append(
            {
                "system_prompt": str(it.get("system_prompt", "")),
                "user_prompt": str(it.get("user_prompt", "")),
            }
        )
    return rows


def display_final_termination_check_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
    *,
    stream: TextIO = sys.stdout,
) -> None:
    """Print prompts for traces with ``component == 'final_termination_check'``."""

    def label(i: int, _: Dict[str, Any]) -> str:
        return f"call #{i}"

    _display_component_prompts(
        interactions,
        FINAL_TERMINATION_CHECK_COMPONENT,
        section_header="final_termination_check",
        empty_message="(no final_termination_check entries in interactions)",
        label_fn=label,
        stream=stream,
    )


def extract_cot_solver_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for it in _filter_component(interactions, COT_SOLVER_COMPONENT):
        rows.append(
            {
                "system_prompt": str(it.get("system_prompt", "")),
                "user_prompt": str(it.get("user_prompt", "")),
                "raw_answer": str(it.get("raw_answer", "")),
            }
        )
    return rows


def display_cot_solver_prompts(
    interactions: Optional[Iterable[Dict[str, Any]]],
    *,
    stream: TextIO = sys.stdout,
) -> None:
    """
    Print prompts (and raw model output) for traces with ``component == 'cot_solver'``.

    CoT baselines store the completion as `raw_answer` without a separate `parsed_answer`.
    """

    def extra(it: Dict[str, Any]) -> List[tuple[str, str]]:
        return [("raw_answer", str(it.get("raw_answer", "")))]

    def label(i: int, _: Dict[str, Any]) -> str:
        return f"call #{i}"

    _display_component_prompts(
        interactions,
        COT_SOLVER_COMPONENT,
        section_header="cot_solver",
        empty_message="(no cot_solver entries in interactions)",
        label_fn=label,
        stream=stream,
        extra_sections_fn=extra,
    )
