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


RUN_META_REQUIRED = (
    "run_id",
    "pipeline_mode",
    "dataset",
    "pipeline_config",
    "seed",
    "model_specs_by_role",
    "ablation",
    "code_version",
)

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
