"""
Gate B: validate persisted run directories (run_meta.json, examples.jsonl, failures.jsonl).

Usage:
  python -m eval.artifact.validate_artifacts path/to/artifacts/run_<id>
  python -m eval.artifact.validate_artifacts path/to/artifacts  --latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# TODO THINK OF A BETTER WAY TO IMPLEMENT THIS! FOR EXAMPLE, DECLARE EACH FIELD AS NEEDING A CERTAIN CHECK,
# AND THE SUIT CAN AUTOMATICALLY BUILD THIS FOR YOU (E.G. KNOWING SOMETHING MUST BE AN INTEGER, IT CHECKS AUTOMATICALLY)

RUN_META_REQUIRED = (
    "run_id",
    "pipeline_mode",
    "dataset",
    "pipeline_config",
    "seed",
    "model_specs_by_role",
    "ablation",
    "code_version",
    "overall_accuracy",
    "run_timing",
    "llm_usage",
    "failure_counts_by_category",
    "harness",
)

RUN_TIMING_REQUIRED = ("started_at", "finished_at", "duration_seconds")

LLM_USAGE_REQUIRED = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "n_requests",
    "cost_usd",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
)

HARNESS_REQUIRED = ("max_in_flight", "suite_seed", "dataset_subset_seed")

PIPELINE_CONFIG_REQUIRED = (
    "max_steps",
    "explain",
    "use_termination_checks",
    "use_final_termination_check",
)

EXAMPLE_REQUIRED = (
    "example_id",
    "problem",
    "ground_truth",
    "validator",
    "obtained",
    "success",
    "output_summary",
    "output",
)

TERMINATION_REQUIRED = (
    "reason", 
    "route", 
    "termination_rule_source", 
    "steps_count"
)

FAILURE_REQUIRED = (
    "example_id",
    "failure_id",
    "failure_category",
    "failure_note",
    "component_context",
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def validate_run_dir(run_dir: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not run_dir.is_dir():
        return False, [f"Not a directory: {run_dir}"]

    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        errors.append("Missing run_meta.json")
        return False, errors

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in run_meta.json: {e}"]

    for key in RUN_META_REQUIRED:
        if key not in meta:
            errors.append(f"run_meta.json missing key: {key!r}")
    acc = meta.get("overall_accuracy")
    if acc is not None and not isinstance(acc, (int, float)):
        errors.append("run_meta.json key 'overall_accuracy' must be a number")

    run_timing = meta.get("run_timing")
    if not isinstance(run_timing, dict):
        errors.append("run_meta.json key 'run_timing' must be an object")
    else:
        for k in RUN_TIMING_REQUIRED:
            if k not in run_timing:
                errors.append(f"run_meta.json.run_timing missing key: {k!r}")
            elif k == "duration_seconds" and run_timing[k] is not None and not isinstance(
                run_timing[k], (int, float)
            ):
                errors.append(f"run_meta.json.run_timing.{k} must be a number or null")

    llm_usage = meta.get("llm_usage")
    if not isinstance(llm_usage, dict):
        errors.append("run_meta.json key 'llm_usage' must be an object")
    else:
        for k in LLM_USAGE_REQUIRED:
            if k not in llm_usage:
                errors.append(f"run_meta.json.llm_usage missing key: {k!r}")
            elif k == "n_requests" and not isinstance(llm_usage[k], int):
                errors.append(f"run_meta.json.llm_usage.{k} must be an integer")
            elif k in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "reasoning_tokens",
                "cached_tokens",
                "cache_write_tokens",
            ) and not isinstance(llm_usage[k], int):
                errors.append(f"run_meta.json.llm_usage.{k} must be an integer")
            elif k == "cost_usd" and not isinstance(llm_usage[k], (int, float)):
                errors.append("run_meta.json.llm_usage.cost_usd must be a number")

    fcounts = meta.get("failure_counts_by_category")
    if not isinstance(fcounts, dict):
        errors.append("run_meta.json key 'failure_counts_by_category' must be an object")
    else:
        for cat, cnt in fcounts.items():
            if not isinstance(cat, str):
                errors.append("run_meta.json.failure_counts_by_category keys must be strings")
                break
            if not isinstance(cnt, int):
                errors.append(f"run_meta.json.failure_counts_by_category[{cat!r}] must be an integer")
                break

    harness = meta.get("harness")
    if not isinstance(harness, dict):
        errors.append("run_meta.json key 'harness' must be an object")
    else:
        for k in HARNESS_REQUIRED:
            if k not in harness:
                errors.append(f"run_meta.json.harness missing key: {k!r}")
        mi = harness.get("max_in_flight")
        if mi is not None and not isinstance(mi, int):
            errors.append("run_meta.json.harness.max_in_flight must be an integer or null")

    # Optional: system prompt registry usage + hashes
    sp_hashes = meta.get("system_prompts_hashes_by_canonical_name")
    if sp_hashes is not None:
        if not isinstance(sp_hashes, dict):
            errors.append("run_meta.json key 'system_prompts_hashes_by_canonical_name' must be an object")
        else:
            for h, name in sp_hashes.items():
                if not isinstance(h, str):
                    errors.append(
                        "run_meta.json.system_prompts_hashes_by_canonical_name keys must be strings"
                    )
                    break
                if not isinstance(name, str):
                    errors.append(
                        f"run_meta.json.system_prompts_hashes_by_canonical_name[{h!r}] must be a string"
                    )
                    break
                if len(h) != 64:
                    errors.append(
                        f"run_meta.json.system_prompts_hashes_by_canonical_name[{h!r}] must look like sha256 hex"
                    )
                    break

    sp_texts = meta.get("system_prompts_used_content_by_hash")
    if sp_texts is not None:
        if not isinstance(sp_texts, dict):
            errors.append("run_meta.json key 'system_prompts_used_content_by_hash' must be an object")
        else:
            for h, text in sp_texts.items():
                if not isinstance(h, str):
                    errors.append(
                        "run_meta.json.system_prompts_used_content_by_hash keys must be strings"
                    )
                    break
                if len(h) != 64:
                    errors.append(
                        f"run_meta.json.system_prompts_used_content_by_hash[{h!r}] must look like sha256 hex"
                    )
                    break
                if not isinstance(text, str):
                    errors.append(
                        f"run_meta.json.system_prompts_used_content_by_hash[{h!r}] must be a string"
                    )
                    break

    sp_used = meta.get("system_prompts_used_by_role")
    if sp_used is not None:
        if not isinstance(sp_used, dict):
            errors.append("run_meta.json key 'system_prompts_used_by_role' must be an object")
        else:
            for role, entries in sp_used.items():
                if not isinstance(role, str):
                    errors.append("run_meta.json.system_prompts_used_by_role keys must be strings")
                    break
                if not isinstance(entries, list):
                    errors.append(f"run_meta.json.system_prompts_used_by_role[{role!r}] must be a list")
                    break
                for i, ent in enumerate(entries):
                    if not isinstance(ent, dict):
                        errors.append(
                            f"run_meta.json.system_prompts_used_by_role[{role!r}][{i}] must be an object"
                        )
                        continue
                    for k in ("component", "prompt_name", "prompt_hash"):
                        if k not in ent:
                            errors.append(
                                f"run_meta.json.system_prompts_used_by_role[{role!r}][{i}] missing key {k!r}"
                            )
                    ph = ent.get("prompt_hash")
                    if isinstance(ph, str) and len(ph) != 64:
                        errors.append(
                            f"run_meta.json.system_prompts_used_by_role[{role!r}][{i}].prompt_hash must look like sha256 hex"
                        )

    ds = meta.get("dataset")
    if isinstance(ds, dict):
        spec = ds.get("subset_spec")
        if spec is not None:
            if not isinstance(spec, dict):
                errors.append("run_meta.json dataset.subset_spec must be an object when present")
            elif "example_ids" not in spec:
                errors.append("run_meta.json dataset.subset_spec missing key: 'example_ids'")
            elif not isinstance(spec.get("example_ids"), list):
                errors.append("run_meta.json dataset.subset_spec.example_ids must be a list")
    pipeline_config = meta.get("pipeline_config")
    if not isinstance(pipeline_config, dict):
        errors.append("run_meta.json key 'pipeline_config' must be an object")
    else:
        for key in PIPELINE_CONFIG_REQUIRED:
            if key not in pipeline_config:
                errors.append(f"run_meta.json.pipeline_config missing key: {key!r}")

    ex_path = run_dir / "examples.jsonl"
    if not ex_path.is_file():
        errors.append("Missing examples.jsonl")
    else:
        try:
            examples = _load_jsonl(ex_path)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSONL in examples.jsonl: {e}")
        else:
            for i, row in enumerate(examples):
                for key in EXAMPLE_REQUIRED:
                    if key not in row:
                        errors.append(f"examples.jsonl row {i} missing key: {key!r}")
                osum = row.get("output_summary")
                if osum is not None and not isinstance(osum, dict):
                    errors.append(f"examples.jsonl row {i}: output_summary must be an object")
                out = row.get("output")
                if out is not None and not isinstance(out, dict):
                    errors.append(f"examples.jsonl row {i}: output must be an object or null")
                if isinstance(out, dict):
                    rt = out.get("result_type")
                    if not isinstance(rt, str) or not rt:
                        errors.append(f"examples.jsonl row {i}: output.result_type must be a non-empty string")
                    elif rt == "CoTResult":
                        if "answer_text" not in out:
                            errors.append(f"examples.jsonl row {i}: output.CoTResult missing 'answer_text'")
                        if "reasoning" not in out:
                            errors.append(f"examples.jsonl row {i}: output.CoTResult missing 'reasoning'")
                    elif rt == "PipelineResult":
                        for k in ("success", "steps", "answer_spec", "final_premises"):
                            if k not in out:
                                errors.append(f"examples.jsonl row {i}: output.PipelineResult missing key: {k!r}")
                        if "steps" in out and not isinstance(out.get("steps"), list):
                            errors.append(f"examples.jsonl row {i}: output.PipelineResult.steps must be a list")
                        if "final_premises" in out and not isinstance(out.get("final_premises"), list):
                            errors.append(f"examples.jsonl row {i}: output.PipelineResult.final_premises must be a list")
                if isinstance(osum, dict) and row.get("validator", "").startswith("gsm8k_main_validator:symbolic_hybrid"):
                    termination = osum.get("termination")
                    if termination is not None and not isinstance(termination, dict):
                        errors.append(f"examples.jsonl row {i}: output_summary.termination must be an object")
                    if isinstance(termination, dict):
                        for tkey in TERMINATION_REQUIRED:
                            if tkey not in termination:
                                errors.append(
                                    f"examples.jsonl row {i}: output_summary.termination missing key: {tkey!r}"
                                )

    fail_path = run_dir / "failures.jsonl"
    if fail_path.is_file():
        try:
            fails = _load_jsonl(fail_path)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSONL in failures.jsonl: {e}")
        else:
            for i, row in enumerate(fails):
                for key in FAILURE_REQUIRED:
                    if key not in row:
                        errors.append(f"failures.jsonl row {i} missing key: {key!r}")
                ctx = row.get("component_context")
                if ctx is not None and not isinstance(ctx, dict):
                    errors.append(f"failures.jsonl row {i}: component_context must be an object")

    return len(errors) == 0, errors


def _latest_run_dir(artifacts_root: Path) -> Path | None:
    if not artifacts_root.is_dir():
        return None
    candidates = sorted(
        (p for p in artifacts_root.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate evaluation artifact directories.")
    p.add_argument("path", type=Path, help="run_<id> directory or artifacts root")
    p.add_argument("--latest", action="store_true", help="If path is artifacts root, validate latest run_*")
    args = p.parse_args(argv)

    target = args.path.resolve()
    if args.latest:
        latest = _latest_run_dir(target)
        if latest is None:
            print(f"No run_* directory under {target}", file=sys.stderr)
            return 2
        target = latest
        print(f"Validating {target}")

    ok, errs = validate_run_dir(target)
    if ok:
        print("OK: artifact schema checks passed.")
        return 0
    for e in errs:
        print(e, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
