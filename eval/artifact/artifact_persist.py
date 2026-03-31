"""
Persist evaluation runs to disk using the artifact layout from the automation spec:
  artifacts/run_<run_id>/run_meta.json
  artifacts/run_<run_id>/examples.jsonl
  artifacts/run_<run_id>/failures.jsonl (optional)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval.eval_suite import (
    EvaluationSuite,
    ExampleOutcome,
    ModelMapping,
    ModelSpec,
    PipelineMode,
    SimpleEvalTask,
    SuiteReport,
)
from llm_prolog.system_prompts import (
    SYSTEM_PROMPTS_BY_NAME,
    SYSTEM_PROMPT_HASHES_BY_NAME,
    hash_system_prompt_text,
)


def new_run_id() -> str:
    """Timestamp + short hash for a unique run directory name."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:8]
    return f"{ts}_{h}"


def git_code_version(repo_root: Optional[Path] = None) -> str:
    try:
        cwd = str(repo_root) if repo_root else None
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _get_prompt_override(
    prompt_overrides: Optional[Mapping[Any, str]],
    role: str,
) -> Optional[str]:
    """
    Find the override string for a given role.

    Mirrors the pipeline's `_role_key` behavior.
    """
    def _role_key(role: Any) -> str:
        """
        Normalize a role key (Enum or string) to a stable string.
        """
        if isinstance(role, str):
            return role
        value = getattr(role, "value", None)
        if isinstance(value, str):
            return value
        return str(role)

    if not prompt_overrides:
        return None
    for k, v in prompt_overrides.items():
        if _role_key(k) == role:
            return v
    return None


def _maybe_add_used_prompt_entry(
    entries: List[Dict[str, Any]],
    *,
    component: str,
    prompt_name: str,
    prompt_text: str,
) -> None:
    entries.append(
        {
            "component": component,
            "prompt_name": prompt_name,
            "prompt_hash": hash_system_prompt_text(prompt_text),
        }
    )


def _system_prompts_used_by_role(
    *,
    suite: EvaluationSuite
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute which system prompts were used for each role/model in this suite run.

    We record:
    - which role used which system prompt (component-level)
    - the SHA-256 hash of that prompt text
    """
    prompt_overrides = suite.prompt_overrides or {}

    used_roles: Dict[str, List[Dict[str, Any]]] = {}
    required_roles = [r.value for r in suite.pipeline_mode.get_required_roles()]
    for rr in required_roles:
        used_roles[rr] = []
    

    if suite.pipeline_mode == PipelineMode.SYMBOLIC_HYBRID:
        # nl_to_symbol
        nl_override = _get_prompt_override(prompt_overrides, "nl_to_symbol")
        if nl_override is not None:
            _maybe_add_used_prompt_entry(
                used_roles["nl_to_symbol"],
                component="nl_to_symbol",
                prompt_name="override:nl_to_symbol",
                prompt_text=nl_override,
            )
        else:
            _maybe_add_used_prompt_entry(
                used_roles["nl_to_symbol"],
                component="nl_to_symbol",
                prompt_name="nl_to_symbol",
                prompt_text=SYSTEM_PROMPTS_BY_NAME["nl_to_symbol"],
            )

        # selector (select_next_step)
        selector_override = _get_prompt_override(prompt_overrides, "selector")
        if selector_override is not None:
            _maybe_add_used_prompt_entry(
                used_roles["selector"],
                component="selector_select_next_step",
                prompt_name="override:selector",
                prompt_text=selector_override,
            )
        else:
            if suite.pipeline_cfg.use_termination_checks:
                cname = "selector_with_termination_checks"
            else:
                cname = "selector_no_termination_checks"
            _maybe_add_used_prompt_entry(
                used_roles["selector"],
                component="selector_select_next_step",
                prompt_name=cname,
                prompt_text=SYSTEM_PROMPTS_BY_NAME[cname],
            )

        # selector (final termination-check)
        if suite.pipeline_cfg.use_final_termination_check:
            if selector_override is not None:
                _maybe_add_used_prompt_entry(
                    used_roles["selector"],
                    component="final_termination_check",
                    prompt_name="override:selector",
                    prompt_text=selector_override,
                )
            else:
                _maybe_add_used_prompt_entry(
                    used_roles["selector"],
                    component="final_termination_check",
                    prompt_name="final_termination_check",
                    prompt_text=SYSTEM_PROMPTS_BY_NAME["final_termination_check"],
                )

        # symbol_to_nl (only if explain is enabled)
        if suite.pipeline_cfg.explain:
            sym2nl_override = _get_prompt_override(prompt_overrides, "symbol_to_nl")
            if sym2nl_override is not None:
                _maybe_add_used_prompt_entry(
                    used_roles["symbol_to_nl"],
                    component="symbol_to_nl",
                    prompt_name="override:symbol_to_nl",
                    prompt_text=sym2nl_override,
                )
            else:
                _maybe_add_used_prompt_entry(
                    used_roles["symbol_to_nl"],
                    component="symbol_to_nl",
                    prompt_name="symbol_to_nl",
                    prompt_text=SYSTEM_PROMPTS_BY_NAME["symbol_to_nl"],
                )

    elif suite.pipeline_mode == PipelineMode.COT_BASELINE:
        # cot_solver (only for cot_baseline)
        cot_override = _get_prompt_override(prompt_overrides, "cot_solver")
        if cot_override is not None:
            _maybe_add_used_prompt_entry(
                used_roles["cot_solver"],
                component="cot_solver",
                prompt_name="override:cot_solver",
                prompt_text=cot_override,
            )
        else:
            COT_PROMPT_NAME = "cot_solver_fewshot"
            _maybe_add_used_prompt_entry(
                used_roles["cot_solver"],
                component="cot_solver",
                prompt_name=COT_PROMPT_NAME,
                prompt_text=SYSTEM_PROMPTS_BY_NAME[COT_PROMPT_NAME],
            )

    return used_roles


def model_specs_by_role_json(model_by_role: ModelMapping) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for role, spec in model_by_role.items():
        out[role.value] = model_spec_to_json(spec)
    return out


def model_spec_to_json(spec: ModelSpec) -> Dict[str, Any]:
    return {
        "model": spec.model,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }


def build_output_summary(result: Any, pipeline_mode: PipelineMode) -> Dict[str, Any]:
    if pipeline_mode == PipelineMode.COT_BASELINE:
        answer_text = getattr(result, "answer_text", None)
        model = getattr(result, "model", None)
        return {
            "result_type": "CoTResult",
            "answer_text": answer_text if isinstance(answer_text, str) else None,
            "model": model,
        }
    if pipeline_mode == PipelineMode.SYMBOLIC_HYBRID:
        if result is None:
            return {
                "result_type": "PipelineResult",
                "success": False,
                "answer_premise_text": None,
                "extracted_answer_constant": None,
                "termination": {
                    "reason": None,
                    "route": None,
                    "termination_rule_source": None,
                    "steps_count": 0,
                },
            }
        ap = getattr(result, "answer_premise", None)
        ap_text = str(ap) if ap is not None else None
        ext = None
        reason = getattr(result, "reason", None)
        steps = getattr(result, "steps", None)
        steps_count = len(steps) if isinstance(steps, list) else 0
        termination_rule_source = None
        if isinstance(steps, list) and steps:
            last_new = getattr(steps[-1], "new_premise", None)
            if last_new is not None:
                src = getattr(last_new, "source", None)
                termination_rule_source = src if isinstance(src, str) else None
        termination_route = None
        if reason == "termination_checker_verified":
            termination_route = "inline_selector_termination_check"
        elif termination_rule_source == "final_termination_check":
            termination_route = "post_loop_final_termination_check"
        elif reason == "answer_head_matched":
            termination_route = "answer_head_match"
        elif reason == "max_steps_exhausted":
            termination_route = "max_steps_exhausted"
        if hasattr(result, "extract_answer_constant"):
            try:
                ext = result.extract_answer_constant()
            except Exception:
                ext = None
        return {
            "result_type": "PipelineResult",
            "success": bool(getattr(result, "success", False)),
            "answer_premise_text": ap_text,
            "extracted_answer_constant": ext,
            "termination": {
                "reason": reason if isinstance(reason, str) else None,
                "route": termination_route,
                "termination_rule_source": termination_rule_source,
                "steps_count": steps_count,
            },
        }
    return {"result_type": type(result).__name__ if result is not None else "None"}


def _validator_label(pipeline_mode: PipelineMode) -> str:
    return f"gsm8k_main_validator:{pipeline_mode.value}"


def _example_lookup(examples: Sequence[Any]) -> Dict[str, Any]:
    by_id: Dict[str, Any] = {}
    for i, ex in enumerate(examples):
        eid = str(getattr(ex, "id", i))
        by_id[eid] = ex
    return by_id


def _obtained_float(
    outcome: ExampleOutcome,
    task: SimpleEvalTask,
    pipeline_mode: PipelineMode,
) -> Optional[float]:
    if outcome.error:
        return None
    if outcome.result is None:
        return None
    val = task.validator(outcome.result, pipeline_mode)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _cot_failure_category(outcome: ExampleOutcome, obtained: Optional[float]) -> Tuple[str, str]:
    if outcome.error:
        err = outcome.error or ""
        if "429" in err or "rate" in err.lower():
            return "llm_api_error", err
        return "llm_unknown_exception", err
    if obtained is None:
        return "extraction_failed", "Could not extract numeric answer from CoT output"
    return "wrong_numeric_answer", "Extracted value does not match ground truth"


# TODO THIS FUNCTION IS VERY SPECIFIC TO GSM8K
def _failure_rows_for_outcome(
    *,
    outcome: ExampleOutcome,
    obtained: Optional[float],
    pipeline_mode: PipelineMode,
    model_by_role: ModelMapping,
    failure_prefix: str,
) -> List[Dict[str, Any]]:
    if outcome.correct and not outcome.error:
        return []
    if pipeline_mode == PipelineMode.COT_BASELINE:
        cat, note = _cot_failure_category(outcome, obtained)
    elif pipeline_mode == PipelineMode.SYMBOLIC_HYBRID:
        if outcome.error:
            cat, note = "llm_unknown_exception", outcome.error or ""
        else:
            cat, note = "wrong_numeric_answer", "Symbolic pipeline did not yield correct numeric answer"
    else:
        cat, note = "llm_unknown_exception", outcome.error or "unknown failure"
    fid = f"{failure_prefix}_{outcome.example_id}"
    row = {
        "example_id": outcome.example_id,
        "failure_id": fid,
        "failure_category": cat,
        "failure_note": note,
        "component_context": {
            "roles": model_specs_by_role_json(model_by_role),
            "step_index": None,
            "used_premise_ids": None,
        },
        "debug_snapshot": None,
    }
    return [row]


def _ordered_example_ids_from_report(
    suite_report: SuiteReport
) -> List[Any]:
    """IDs in the same order as persisted example rows (per task, then per outcome)."""
    ordered: List[Any] = []
    for trep in suite_report.task_reports:
        for outcome in trep.outcomes:
            raw = outcome.example_id
            try:
                ordered.append(int(raw))
            except (TypeError, ValueError):
                ordered.append(raw)
    return ordered
 

def _timing_llm_harness_for_run_meta(
    suite: EvaluationSuite,
    suite_report: SuiteReport,
    dataset_out: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Normalize SuiteReport.run_metadata into persisted run_meta sections."""
    rm = dict(suite_report.run_metadata) if suite_report.run_metadata else {}
    timing = dict(rm.get("run_timing") or {})
    if not timing:
        timing = {"started_at": None, "finished_at": None, "duration_seconds": None}
    llm_usage = dict(rm.get("llm_usage") or {})
    _usage_defaults: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "n_requests": 0,
        "cost_usd": 0.0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
    }
    for k, v in _usage_defaults.items():
        if k not in llm_usage:
            llm_usage[k] = v
    raw_sub = dataset_out.get("subset_spec")
    subset = raw_sub if isinstance(raw_sub, dict) else {}
    harness = {
        "max_in_flight": rm.get("max_in_flight"),
        "suite_seed": suite.seed,
        "dataset_subset_seed": subset.get("seed"),
    }
    return timing, llm_usage, harness


def persist_evaluation_run(
    *,
    artifacts_root: Path,
    run_id: str,
    suite: EvaluationSuite,
    suite_report: SuiteReport,
    dataset: Mapping[str, Any],
    ablation: Mapping[str, Any],
    code_version: Optional[str] = None,
    write_failures: bool = True,
) -> Path:
    """
    Write run_meta.json, examples.jsonl, and optionally failures.jsonl under
    artifacts_root / run_<run_id>.
    """
    run_dir = artifacts_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if code_version is None:
        code_version = git_code_version()

    overall_accuracy = suite_report.overall_accuracy
    example_ids = _ordered_example_ids_from_report(suite_report)
    dataset_out = dict(dataset)
    subset_spec = dict(dataset_out.get("subset_spec") or {})
    subset_spec["example_ids"] = example_ids
    dataset_out["subset_spec"] = subset_spec

    run_timing, llm_usage, harness = _timing_llm_harness_for_run_meta(
        suite, suite_report, dataset_out
    )

    example_lines: List[str] = []
    failure_lines: List[str] = []
    failure_counts: Dict[str, int] = {}
    failure_prefix = f"f_{run_id[:16]}"

    for task, trep in zip(suite.tasks, suite_report.task_reports):
        if not isinstance(task, SimpleEvalTask):
            raise TypeError("persist_evaluation_run expects SimpleEvalTask items in suite.tasks")
        examples = list(task.load_examples())
        by_id = _example_lookup(examples)

        for outcome in trep.outcomes:
            ex = by_id.get(outcome.example_id)
            if ex is None:
                ex = examples[outcome.idx]
            # TODO This might be specific to GSM8K
            gt = getattr(ex, "ground_truth", None)
            if gt is not None:
                try:
                    ground_truth: Any = float(gt)
                except (TypeError, ValueError):
                    ground_truth = gt
            else:
                ground_truth = None

            obtained = _obtained_float(outcome, task, suite.pipeline_mode)
            # TODO END
            success = bool(outcome.correct)
            reason: Optional[str] = None
            if outcome.error:
                reason = outcome.error
            elif outcome.result is not None:
                r = getattr(outcome.result, "reason", None)
                reason = r if isinstance(r, str) or r is None else str(r)

            rec = {
                "example_id": outcome.example_id,
                "problem": outcome.problem,
                # TODO This might be specific to GSM8K
                "ground_truth": ground_truth,
                "validator": _validator_label(suite.pipeline_mode),
                # TODO END
                "obtained": obtained,
                "success": success,
                "reason": reason if isinstance(reason, str) or reason is None else str(reason),
                "output_summary": build_output_summary(outcome.result, suite.pipeline_mode),
            }
            example_lines.append(json.dumps(rec, ensure_ascii=False))

            if write_failures:
                for fr in _failure_rows_for_outcome(
                    outcome=outcome,
                    obtained=obtained,
                    pipeline_mode=suite.pipeline_mode,
                    model_by_role=suite.model_by_role,
                    failure_prefix=failure_prefix,
                ):
                    cat = fr.get("failure_category")
                    if isinstance(cat, str) and cat:
                        failure_counts[cat] = failure_counts.get(cat, 0) + 1
                    failure_lines.append(json.dumps(fr, ensure_ascii=False))

    run_meta = {
        "run_id": run_id,
        "pipeline_mode": suite.pipeline_mode.value,
        "dataset": dataset_out,
        "pipeline_config": {
            "max_steps": suite.pipeline_cfg.max_steps,
            "explain": suite.pipeline_cfg.explain,
            "use_termination_checks": suite.pipeline_cfg.use_termination_checks,
            "use_final_termination_check": suite.pipeline_cfg.use_final_termination_check,
        },
        "seed": suite.seed,
        "model_specs_by_role": model_specs_by_role_json(suite.model_by_role),
        "ablation": dict(ablation),
        "code_version": code_version,
        "suite_name": suite.name,
        "overall_accuracy": overall_accuracy,
        "run_timing": run_timing,
        "llm_usage": llm_usage,
        "failure_counts_by_category": dict(sorted(failure_counts.items(), key=lambda x: (-x[1], x[0]))),
        "harness": harness,
        "system_prompts_hashes_by_canonical_name": dict(SYSTEM_PROMPT_HASHES_BY_NAME),
        "system_prompts_used_by_role": _system_prompts_used_by_role(
            suite=suite
        ),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (run_dir / "examples.jsonl").write_text("\n".join(example_lines) + ("\n" if example_lines else ""), encoding="utf-8")
    if write_failures and failure_lines:
        (run_dir / "failures.jsonl").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    return run_dir
