"""
Where-it-breaks experiment scaffolding (step 3e).

This module encodes variant generation for the staircase protocol and documents
what downstream analysis should read from persisted artifacts.

Logged / persisted per run (see eval.artifact):
- run_meta.json: run_id, pipeline_mode, dataset (subset_spec incl. example_ids), pipeline_config,
  seed, model_specs_by_role, ablation, code_version, suite_name, overall_accuracy,
  run_timing, llm_usage, failure_counts_by_category, harness (max_in_flight, suite_seed,
  dataset_subset_seed)
- examples.jsonl: example_id, problem, ground_truth, validator, obtained,
  success, reason, output_summary (mode-specific)
- failures.jsonl: example_id, failure_id, failure_category, failure_note,
  component_context (roles, step_index, used_premise_ids), debug_snapshot

Planned derived outputs (to be produced by analysis scripts):
- ablation_variants.json — list of variant specs before runs
- model_rankings.json — accuracy ladder from Phase 1
- failure_mode_profiles.json — aggregates from failures.jsonl
- representative_examples.json — ids for qualitative follow-up

Phase 2 has two entry points:
- ``run_where_it_breaks_phase_2`` — full pipeline per example; sweeps ``nl_to_symbol`` and ``selector``.
- ``run_where_it_breaks_phase_2_reuse_initial_symbolization`` — initial clauses from a prior run's
  ``examples.jsonl``; selector loop only (``run_symbolic_hybrid_after_nl_async``;
  ``EvaluationSuite.symbolic_hybrid_initial_by_example_id``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from eval.artifact.analyze_artifacts import _get_model_family_and_name
from eval.artifact.artifact_persist import new_run_id, persist_evaluation_run
from eval.artifact.validate_artifacts import validate_run_dir
from eval.eval_suite import EvaluationSuite, LLMRole, ModelMapping, ModelSpec, PipelineMode, SimpleEvalTask
from llm_prolog.pipeline import PipelineConfig
from llm_prolog.symbolic.types import (
    AnswerSpec,
    PipelineResult,
    Premise,
    initial_premises_for_hybrid_reuse_from_stored_result,
)


def _default_where_it_breaks_root(project_root: Path) -> Path:
    return project_root / "artifacts" / "where-it-breaks"


def _dataset_meta_from_suite(suite: EvaluationSuite) -> Dict[str, Any]:
    """
    Produce the `dataset` dict consumed by `persist_evaluation_run`.

    This follows the pattern in `eval/evaluate_symbolic_hybrid.py`.
    """
    # Best-effort: pull from the first task.
    task = suite.tasks[0] if suite.tasks else None
    task_id = getattr(task, "task_id", "unknown_task")
    task_id_str = str(task_id)
    split = "train" if ":train" in task_id_str else "test" if ":test" in task_id_str else "unknown"
    return {
        "name": "gsm8k" if task_id_str.startswith("gsm8k") else "unknown",
        "split": split,
        "subset_spec": {"task_id": task_id_str},
    }


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run_and_persist(
    *,
    suite: EvaluationSuite,
    artifacts_root: Path,
    variant_id: str,
    dataset_meta: Mapping[str, Any],
    ablation_overrides: Mapping[str, Any] | None = None,
    write_failures: bool = True,
    max_in_flight: int = 15,
) -> Tuple[Any, Path]:
    report = asyncio.run(
        suite.run_async(
            max_in_flight=max_in_flight,
            print_progress=True,
            show_openrouter_balance=True,
        )
    )

    _roles = suite.pipeline_mode.get_required_roles()
    _model = suite.model_by_role[_roles[0]].model if _roles else ""
    run_id = new_run_id(_model)
    run_dir = persist_evaluation_run(
        artifacts_root=artifacts_root.resolve(),
        run_id=run_id,
        suite=suite,
        suite_report=report,
        dataset=dataset_meta,
        ablation={
            "variant_id": variant_id,
            "component_overrides": dict(ablation_overrides or {}),
        },
        write_failures=write_failures,
    )
    print(f"Artifacts written to: {run_dir}")

    ok, errs = validate_run_dir(run_dir)
    if not ok:
        for err in errs:
            print(f"[artifact validation] {err}")
        raise RuntimeError(f"Artifact validation failed for run directory: {run_dir}")
    print("Artifact validation passed.")
    return report, run_dir


def _spec_to_name(spec: ModelSpec) -> str:
    return spec.model.split("/")[-1]


def _suite_with_all_roles_set_to(
    *,
    base_suite: EvaluationSuite,
    spec: ModelSpec,
) -> EvaluationSuite:
    return EvaluationSuite(
        name=base_suite.name,
        tasks=base_suite.tasks,
        pipeline_mode=base_suite.pipeline_mode,
        model_by_role=ModelMapping.set_spec_to_all_roles(spec, base_suite.pipeline_mode),
        prompt_overrides=base_suite.prompt_overrides,
        pipeline_cfg=base_suite.pipeline_cfg,
        keep_all_outcomes=base_suite.keep_all_outcomes,
        keep_random_k=base_suite.keep_random_k,
        seed=base_suite.seed,
    )


def _suite_with_role_overrides(
    *,
    base_suite: EvaluationSuite,
    default_spec: ModelSpec,
    overrides: Mapping[str, ModelSpec],
) -> EvaluationSuite:
    """
    Clone suite, setting all required roles to default_spec, then overriding named roles.
    """
    mapping = ModelMapping.set_spec_to_all_roles(default_spec, base_suite.pipeline_mode)
    for role_name, spec in overrides.items():
        role = LLMRole(role_name)
        mapping.mapping[role] = spec
    return EvaluationSuite(
        name=base_suite.name,
        tasks=base_suite.tasks,
        pipeline_mode=base_suite.pipeline_mode,
        model_by_role=mapping,
        prompt_overrides=base_suite.prompt_overrides,
        pipeline_cfg=base_suite.pipeline_cfg,
        keep_all_outcomes=base_suite.keep_all_outcomes,
        keep_random_k=base_suite.keep_random_k,
        seed=base_suite.seed,
    )


def model_spec_nl_to_symbol_from_run_meta(run_meta: Mapping[str, Any]) -> ModelSpec:
    d = ((run_meta.get("model_specs_by_role") or {}).get("nl_to_symbol")) or {}
    model = d.get("model")
    if not model:
        raise ValueError("run_meta.json missing model_specs_by_role.nl_to_symbol.model")
    return ModelSpec(
        model=str(model),
        temperature=d.get("temperature"),
        max_tokens=d.get("max_tokens"),
    )


def load_symbolic_hybrid_initial_state_from_run_dir(
    run_dir: str | Path,
) -> Tuple[Dict[str, Tuple[List[Premise], AnswerSpec]], Dict[str, Any]]:
    """
    Read a persisted evaluation run directory and build per-example (initial premises,
    answer_spec) for symbolic hybrid reuse.

    Requires full PipelineResult objects in examples.jsonl ``output`` fields.
    Premises are NL→symbol clauses plus, when that row ``success`` is true and the stored
    pipeline ``success`` is true, any ``final_termination_check`` linking rules from
    ``final_premises`` (see ``initial_premises_for_hybrid_reuse_from_stored_result``).
    """
    p = Path(run_dir).resolve()
    meta_path = p / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"run_meta.json not found under {p}")
    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if run_meta.get("pipeline_mode") != "symbolic_hybrid":
        raise ValueError(
            f"Expected pipeline_mode 'symbolic_hybrid', got {run_meta.get('pipeline_mode')!r}"
        )
    ex_path = p / "examples.jsonl"
    if not ex_path.is_file():
        raise FileNotFoundError(f"examples.jsonl not found under {p}")

    out: Dict[str, Tuple[List[Premise], AnswerSpec]] = {}
    for line in ex_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        eid = str(row.get("example_id", ""))
        payload = row.get("output")
        if not isinstance(payload, dict) or payload.get("result_type") != "PipelineResult":
            raise ValueError(
                f"example_id={eid!r}: need output.result_type=='PipelineResult' to reuse premises"
            )
        pr = PipelineResult.from_json_dict(payload)
        example_ok = bool(row.get("success"))
        initials = initial_premises_for_hybrid_reuse_from_stored_result(
            pr,
            example_task_success=example_ok,
        )
        if not initials:
            raise ValueError(f"example_id={eid!r}: no initial NL premises in stored final_premises")
        out[eid] = (initials, pr.answer_spec)
    if not out:
        raise ValueError(f"No examples loaded from {ex_path}")
    return out, run_meta


def _validate_seeds_cover_suite_tasks(
    suite: EvaluationSuite,
    seeds: Mapping[str, Any],
) -> None:
    for task in suite.tasks:
        for i, ex in enumerate(task.load_examples()):
            eid = str(getattr(ex, "id", i))
            if eid not in seeds:
                raise KeyError(
                    f"Initial-state map missing example_id={eid!r} (task {task.task_id!r})"
                )


def _suite_phase2_reuse_nl_from_artifact(
    *,
    base_suite: EvaluationSuite,
    best_spec: ModelSpec,
    nl_spec_from_artifact: ModelSpec,
    selector_spec: ModelSpec,
    seeds: Dict[str, Tuple[List[Premise], AnswerSpec]],
) -> EvaluationSuite:
    mapping = ModelMapping.set_spec_to_all_roles(best_spec, base_suite.pipeline_mode)
    mapping.mapping[LLMRole.NL_TO_SYMBOL] = nl_spec_from_artifact
    mapping.mapping[LLMRole.SELECTOR] = selector_spec
    return EvaluationSuite(
        name=base_suite.name,
        tasks=base_suite.tasks,
        pipeline_mode=base_suite.pipeline_mode,
        model_by_role=mapping,
        prompt_overrides=base_suite.prompt_overrides,
        pipeline_cfg=base_suite.pipeline_cfg,
        keep_all_outcomes=base_suite.keep_all_outcomes,
        keep_random_k=base_suite.keep_random_k,
        seed=base_suite.seed,
        symbolic_hybrid_initial_by_example_id=seeds,
    )


def run_where_it_breaks_phase_1(
    *,
    base_suite: EvaluationSuite,
    ladder_specs: Sequence[ModelSpec],
    artifacts_root: Path,
    write_failures: bool = True,
    max_in_flight: int = 15,
) -> List[Path]:
    """
    Phase 1: run the same EvaluationSuite across a list of ModelSpecs (applied to all required roles).
    """
    run_dirs: List[Path] = []
    dataset_meta = _dataset_meta_from_suite(base_suite)
    for spec in ladder_specs:
        suite = _suite_with_all_roles_set_to(base_suite=base_suite, spec=spec)
        suite.name = f"[where-it-breaks phase-1] {_spec_to_name(spec)} :: {base_suite.name}"
        _, run_dir = _run_and_persist(
            suite=suite,
            artifacts_root=artifacts_root,
            variant_id="where_it_breaks_phase_1",
            dataset_meta=dataset_meta,
            ablation_overrides={"all_roles": spec.model},
            write_failures=write_failures,
            max_in_flight=max_in_flight,
        )
        run_dirs.append(run_dir)
    return run_dirs


def run_where_it_breaks_phase_2(
    *,
    base_suite: EvaluationSuite,
    best_spec: ModelSpec,
    ladder_specs: Sequence[ModelSpec],
    artifacts_root: Path,
    write_failures: bool = True,
    max_in_flight: int = 15,
) -> List[Path]:
    """
    Phase 2 (full pipeline): staircase analysis.

    For each role in ``nl_to_symbol`` and ``selector``:
    - fix all roles to ``best_spec``
    - sweep the target role over ``ladder_specs`` (typically ordered worst retained → best fixed)
    """
    run_dirs: List[Path] = []
    dataset_meta = _dataset_meta_from_suite(base_suite)
    for role_name in ("nl_to_symbol", "selector"):
        for k, spec in enumerate(ladder_specs):
            suite = _suite_with_role_overrides(
                base_suite=base_suite,
                default_spec=best_spec,
                overrides={role_name: spec},
            )
            suite.name = (
                f"[where-it-breaks phase-2] sweep={role_name} k={k} "
                f"{_spec_to_name(spec)} (best={_spec_to_name(best_spec)}) :: {base_suite.name}"
            )
            _, run_dir = _run_and_persist(
                suite=suite,
                artifacts_root=artifacts_root,
                variant_id="where_it_breaks_phase_2",
                dataset_meta=dataset_meta,
                ablation_overrides={
                    "fixed_best_model": best_spec.model,
                    "swept_role": role_name,
                    "swept_model": spec.model,
                    "swept_index": k,
                },
                write_failures=write_failures,
                max_in_flight=max_in_flight,
            )
            run_dirs.append(run_dir)
    return run_dirs


def run_where_it_breaks_phase_2_reuse_initial_symbolization(
    *,
    base_suite: EvaluationSuite,
    best_spec: ModelSpec,
    ladder_specs: Sequence[ModelSpec],
    artifacts_root: Path,
    source_run_dir: str | Path,
    write_failures: bool = True,
    max_in_flight: int = 15,
) -> List[Path]:
    """
    Phase 2 with reused initial premises: load ``run_meta.json`` + ``examples.jsonl`` from
    ``source_run_dir``, take NL→symbol clauses from each stored ``PipelineResult``, and sweep
    only the **selector** role over ``ladder_specs``. The nl_to_symbol model on the suite matches
    ``run_meta.json`` (no NL calls in this mode).
    """
    run_dirs: List[Path] = []
    dataset_meta = _dataset_meta_from_suite(base_suite)
    reuse_path = str(Path(source_run_dir).resolve())
    seeds, run_meta = load_symbolic_hybrid_initial_state_from_run_dir(reuse_path)
    nl_spec_from_artifact = model_spec_nl_to_symbol_from_run_meta(run_meta)
    source_run_id = str(run_meta.get("run_id", ""))
    _validate_seeds_cover_suite_tasks(base_suite, seeds)

    role_name = "selector"
    for k, spec in enumerate(ladder_specs):
        suite = _suite_phase2_reuse_nl_from_artifact(
            base_suite=base_suite,
            best_spec=best_spec,
            nl_spec_from_artifact=nl_spec_from_artifact,
            selector_spec=spec,
            seeds=seeds,
        )
        suite.name = (
            f"[where-it-breaks phase-2] reuse_nl sweep={role_name} k={k} "
            f"{_spec_to_name(spec)} (best={_spec_to_name(best_spec)}) :: {base_suite.name}"
        )
        _, run_dir = _run_and_persist(
            suite=suite,
            artifacts_root=artifacts_root,
            variant_id="where_it_breaks_phase_2",
            dataset_meta=dataset_meta,
            ablation_overrides={
                "fixed_best_model": best_spec.model,
                "swept_role": role_name,
                "swept_model": spec.model,
                "swept_index": k,
                "reuse_initial_symbolization_run_dir": reuse_path,
                "source_run_id": source_run_id,
                "nl_to_symbol_model_from_artifact": nl_spec_from_artifact.model,
            },
            write_failures=write_failures,
            max_in_flight=max_in_flight,
        )
        run_dirs.append(run_dir)
    return run_dirs


def _parse_model_specs_csv(text: str, *, temperature: float | None, max_tokens: int | None) -> List[ModelSpec]:
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    return [ModelSpec(model=p, temperature=temperature, max_tokens=max_tokens) for p in parts]


def _load_run_metas_from_dir(root: Path) -> List[Dict[str, Any]]:
    run_metas: List[Dict[str, Any]] = []
    if not root.is_dir():
        return run_metas
    for meta_path in sorted(root.rglob("run_meta.json")):
        try:
            run_metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[WARN] Failed to read {meta_path}: {e}")
    return run_metas


def get_phase1_models_ordered_by_accuracy(phase1_dir: str | Path) -> List[str]:
    """
    Given a directory containing where-it-breaks Phase 1 artifacts (i.e., containing run dirs with
    `run_meta.json`), return model ids ordered by accuracy (highest to lowest).

    This is intended for automatically constructing Phase 2's ladder ordering from Phase 1 results.
    """
    phase1_root = Path(phase1_dir).resolve()
    phase1_metas = _load_run_metas_from_dir(phase1_root)
    if not phase1_metas:
        raise ValueError(f"No run_meta.json files found under {phase1_root}")

    # Collect per-model accuracy (latest/last-seen value wins if duplicates exist).
    model_to_acc: Dict[str, float] = {}
    seen_order: List[str] = []
    for rm in phase1_metas:
        model = (
            ((rm.get("model_specs_by_role") or {}).get("nl_to_symbol") or {}).get("model")
            or ((rm.get("ablation") or {}).get("component_overrides") or {}).get("all_roles")
            or "unknown"
        )
        model_str = str(model)
        if model_str not in seen_order:
            seen_order.append(model_str)

        acc = rm.get("overall_accuracy")
        try:
            model_to_acc[model_str] = float(acc) if acc is not None else 0.0
        except (TypeError, ValueError):
            model_to_acc[model_str] = 0.0

    # Sort by accuracy desc; tie-break by first-seen order for stability.
    order_index = {m: i for i, m in enumerate(seen_order)}
    ranked_models = sorted(
        model_to_acc.keys(),
        key=lambda m: (-model_to_acc[m], order_index.get(m, 10**9), m),
    )
    return ranked_models


def analyze_where_it_breaks_folder(
    parent_folder: str | Path,
    *,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Analyze a where-it-breaks artifact parent folder and generate:
    - Phase 1 bar chart: one bar per model, y=accuracy.
    - Phase 2 scatter chart: x=role-to-sweep, dots=phase-2 run accuracies colored by swept model,
      plus reference accuracies from phase 1.

    `parent_folder` should contain `phase-1/` and `phase-2/`.
    """
    import matplotlib.pyplot as plt

    parent = Path(parent_folder).resolve()
    phase1_root = parent / "phase-1"
    phase2_root = parent / "phase-2"
    phase1_metas = _load_run_metas_from_dir(phase1_root)
    phase2_metas = _load_run_metas_from_dir(phase2_root)

    if not phase1_metas:
        raise ValueError(f"No run_meta.json files found under {phase1_root}")
    if not phase2_metas:
        raise ValueError(f"No run_meta.json files found under {phase2_root}")

    # ----- Phase 1 data (model -> accuracy) -----
    phase1_model_acc: Dict[str, float] = {}
    for rm in phase1_metas:
        model = (
            ((rm.get("model_specs_by_role") or {}).get("nl_to_symbol") or {}).get("model")
            or ((rm.get("ablation") or {}).get("component_overrides") or {}).get("all_roles")
            or "unknown"
        )
        acc = rm.get("overall_accuracy")
        try:
            phase1_model_acc[str(model)] = float(acc) if acc is not None else 0.0
        except (TypeError, ValueError):
            phase1_model_acc[str(model)] = 0.0

    # Preserve first-seen order from phase 1 metas where possible.
    seen_order: List[str] = []
    for rm in phase1_metas:
        model = (
            ((rm.get("model_specs_by_role") or {}).get("nl_to_symbol") or {}).get("model")
            or ((rm.get("ablation") or {}).get("component_overrides") or {}).get("all_roles")
            or "unknown"
        )
        model_str = str(model)
        if model_str not in seen_order:
            seen_order.append(model_str)
    phase1_models = [m for m in seen_order if m in phase1_model_acc]
    phase1_accs = [phase1_model_acc[m] for m in phase1_models]
    model_families      = [_get_model_family_and_name(m)[0] for m in phase1_models]
    short_phase1_models = [_get_model_family_and_name(m)[1] for m in phase1_models]

    # ----- Plot 1: phase 1 bar graph -----
    # Assign each family a unique color using a qualitative colormap (e.g. tab20)
    unique_families = list(dict.fromkeys(model_families))  # preserves order
    cmap = plt.get_cmap("tab20")
    family_to_color = {fam: cmap(i % 20) for i, fam in enumerate(unique_families)}
    bar_colors = [family_to_color[fam] for fam in model_families]

    fig1 = plt.figure(figsize=(max(7, 0.9 * len(phase1_models)), 4.5))
    ax1 = fig1.add_subplot(111)
    bars = ax1.bar(range(len(short_phase1_models)), phase1_accs, color=bar_colors)
    ax1.set_xticks(range(len(short_phase1_models)))
    ax1.set_xticklabels(short_phase1_models, rotation=35, ha="right")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Where-it-breaks Phase 1: model performance")
    ax1.set_ylim(0.0, 1.0)

    # Make a legend mapping family to color
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=family_to_color[fam], label=fam) for fam in unique_families]
    ax1.legend(handles=handles, title="Model family", loc="best")

    fig1.tight_layout()

    # ----- Phase 2 data (role, model, accuracy) -----
    phase2_points: List[Tuple[str, str, float]] = []
    for rm in phase2_metas:
        overrides = ((rm.get("ablation") or {}).get("component_overrides") or {})
        role = overrides.get("swept_role")
        model = overrides.get("swept_model")
        if role is None:
            role = "unknown"
        if model is None:
            # Fallback: infer from role's assigned model.
            role_models = rm.get("model_specs_by_role") or {}
            model = (role_models.get(str(role)) or {}).get("model", "unknown")
        acc = rm.get("overall_accuracy")
        try:
            acc_val = float(acc) if acc is not None else 0.0
        except (TypeError, ValueError):
            acc_val = 0.0
        phase2_points.append((str(role), str(model), acc_val))

    role_order = ["nl_to_symbol", "selector"]
    all_roles = sorted({r for r, _m, _a in phase2_points if r not in role_order})
    role_order.extend(all_roles)
    role_to_x = {r: i for i, r in enumerate(role_order)}

    models_for_colors = sorted({m for _r, m, _a in phase2_points} | set(phase1_models))
    cmap = plt.get_cmap("tab20")
    model_to_color = {m: cmap(i % 20) for i, m in enumerate(models_for_colors)}

    # ----- Plot 2: phase 2 role sweep scatter + phase 1 reference -----
    fig2 = plt.figure(figsize=(max(7, 1.8 * len(role_order)), 4.8))
    ax2 = fig2.add_subplot(111)

    # Plot phase 2 dots.
    for role, model, acc in phase2_points:
        x = role_to_x.get(role, 0)
        ax2.scatter(
            x,
            acc,
            color=model_to_color.get(model, "gray"),
            marker="o",
            s=65,
            alpha=0.9,
        )

    # Overlay phase 1 reference points (same model's all-roles baseline) at each role.
    for model, ref_acc in phase1_model_acc.items():
        for role in role_order:
            x = role_to_x[role]
            ax2.scatter(
                x,
                ref_acc,
                color=model_to_color.get(model, "gray"),
                marker="x",
                s=60,
                alpha=0.8,
            )

    ax2.set_xticks(range(len(role_order)))
    ax2.set_xticklabels(role_order)
    ax2.set_xlabel("Role-to-sweep")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Where-it-breaks Phase 2: staircase performance (+ phase 1 references)")
    ax2.set_xlim(-1.0, len(role_order))
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis="y", alpha=0.2)

    # Legend: color per model + marker meaning.
    from matplotlib.lines import Line2D

    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=model_to_color[m], markersize=7, label=m.split("/")[-1])
        for m in models_for_colors
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="phase 2 run"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="phase 1 reference"),
    ]
    ax2.legend(handles=model_handles + marker_handles, loc="best", fontsize=8, ncol=2)
    fig2.tight_layout()

    if save_dir is not None:
        out = Path(save_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out / "where_it_breaks_phase1_performance.png", dpi=200, bbox_inches="tight")
        fig2.savefig(out / "where_it_breaks_phase2_staircase.png", dpi=200, bbox_inches="tight")
        print(f"[where-it-breaks] Saved plots to {out}")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    return {
        "parent_folder": str(parent),
        "phase1_model_acc": phase1_model_acc,
        "phase2_points": [{"role": r, "model": m, "accuracy": a} for (r, m, a) in phase2_points],
    }

def where_it_breaks_50exs():
    # Internal constants for a fixed reproducible experiment.
    gsm8k_size = 50
    gsm8k_seed = 99
    from_train_split = True  # matches __main__ default behavior
    temperature = 0.5
    max_tokens = None
    max_in_flight = 15
    write_failures = True

    # Use all models provided by eval/__init__.py.
    from eval import get_all_model_names_filtered

    all_model_names = get_all_model_names_filtered(
        include_llama405b=False,
        include_llama_family=True,
        include_gpt_family=True,
        include_qwen_family=True
    )

    project_root = Path(__file__).resolve().parents[1]
    where_root = project_root / "artifacts" / "where-it-breaks-50EXS"

    # Import here to avoid forcing dataset deps on module import.
    from eval.per_dataset.eval_gsm8k import (
        gsm8k_main_validator,
        gsm8k_success_measure,
        load_gsm8k_examples,
    )
    from eval.eval_suite import default_expected_repr, default_problem_str

    task = SimpleEvalTask(
        task_id=(
            f"gsm8k:{gsm8k_size}exs:{'train' if from_train_split else 'test'}:seed={gsm8k_seed}"
        ),
        examples=load_gsm8k_examples(
            size=gsm8k_size, seed=gsm8k_seed, from_train_split=from_train_split
        ),
        validator_fn=gsm8k_main_validator,
        success_measure_fn=gsm8k_success_measure,
        problem_fn=default_problem_str,
        expected_fn=default_expected_repr,
    )

    ladder_models_csv = ",".join(all_model_names)
    ladder_specs = _parse_model_specs_csv(
        ladder_models_csv,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not ladder_specs:
        raise SystemExit("No models parsed from ALL_MODEL_NAMES")

    mode = PipelineMode.SYMBOLIC_HYBRID
    pipeline_cfg = PipelineConfig(
        max_steps=20, 
        explain=False,
        use_termination_checks=False,
        use_final_termination_check=True,
        selector_num_candidates=3
        )
    base_suite = EvaluationSuite(
        name=f"Symbolic Hybrid On GSM8K ({gsm8k_size}EX)",
        tasks=[task],
        pipeline_mode=mode,
        model_by_role=ModelMapping.set_spec_to_all_roles(ladder_specs[0], mode),
        prompt_overrides=None,
        pipeline_cfg=pipeline_cfg,
        keep_all_outcomes=True,
        keep_random_k=0,
        seed=0,
    )

    phase1_root = _ensure_dir(where_root / "phase-1")
    phase2_root = _ensure_dir(where_root / "phase-2")
    (where_root / "README.json").write_text(
        json.dumps(
            {
                "analysis": "where-it-breaks",
                "layout": {
                    "phase-1": str(phase1_root),
                    "phase-2": str(phase2_root),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Phase 1: run across all models to measure accuracy ladder.
    run_where_it_breaks_phase_1(
        base_suite=base_suite,
        ladder_specs=ladder_specs,
        artifacts_root=phase1_root,
        write_failures=write_failures,
        max_in_flight=max_in_flight,
    )

    # Phase 2: run staircase sweep using phase-1 accuracy ordering (best -> worst).
    ranked_models = get_phase1_models_ordered_by_accuracy(phase1_root)
    spec_by_model = {s.model: s for s in ladder_specs}
    ranked_specs: List[ModelSpec] = [
        spec_by_model.get(
            m, ModelSpec(model=m, temperature=temperature, max_tokens=max_tokens)
        )
        for m in ranked_models
    ]
    if ranked_specs:
        phase2_best_spec = ranked_specs[0]
        phase2_ladder_specs = ranked_specs[1:]
    else:
        phase2_best_spec = ladder_specs[0]
        phase2_ladder_specs = ladder_specs[1:]

    run_where_it_breaks_phase_2(
        base_suite=base_suite,
        best_spec=phase2_best_spec,
        ladder_specs=phase2_ladder_specs,
        artifacts_root=phase2_root,
        write_failures=write_failures,
        max_in_flight=max_in_flight,
    )

    # Produce the two required figures (phase 1 + phase 2).
    return analyze_where_it_breaks_folder(where_root, show=False, save_dir=where_root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the where-it-breaks Phase 1 + Phase 2 analyses.")
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "Root directory to write artifacts under. "
            "If omitted, defaults to <project_root>/artifacts/where-it-breaks."
        ),
    )
    p.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Which phase(s) to run.",
    )
    p.add_argument(
        "--phase2-reuse-run-dir",
        type=Path,
        default=None,
        help=(
            "For phase 2 only: evaluation run directory (run_meta.json + examples.jsonl) "
            "to reuse stored initial NL symbolizations; sweeps selector only."
        ),
    )
    p.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated ordered model ids (best first). Used for Phase 1 ranking and Phase 2 ladder.",
    )
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--no-failures", action="store_true", help="Do not write failures.jsonl.")
    p.add_argument("--max-in-flight", type=int, default=15)
    p.add_argument("--gsm8k-size", type=int, required=True, help="How many GSM8K examples to evaluate.")
    p.add_argument("--gsm8k-seed", type=int, default=42)
    p.add_argument("--gsm8k-train", action="store_true", help="Use GSM8K train split (default).")
    p.add_argument("--gsm8k-test", action="store_true", help="Use GSM8K test split.")
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing where-it-breaks artifacts under --artifacts-dir.",
    )
    p.add_argument(
        "--analysis-save-dir",
        type=Path,
        default=None,
        help="If provided, save generated phase-1/phase-2 plots to this directory.",
    )
    p.add_argument(
        "--no-show-plots",
        action="store_true",
        help="Generate plots without opening an interactive window.",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    where_root = args.artifacts_dir.resolve() if args.artifacts_dir else _default_where_it_breaks_root(project_root)

    if args.analyze_only:
        analyze_where_it_breaks_folder(
            where_root,
            show=not args.no_show_plots,
            save_dir=args.analysis_save_dir,
        )
        raise SystemExit(0)

    # Import here to avoid forcing dataset deps on module import.
    from eval.per_dataset.eval_gsm8k import (
        gsm8k_main_validator,
        gsm8k_success_measure,
        load_gsm8k_examples,
    )
    from eval.eval_suite import default_expected_repr, default_problem_str

    from_train_split = True
    if args.gsm8k_test:
        from_train_split = False
    if args.gsm8k_train:
        from_train_split = True

    # Build the "exact same" base EvaluationSuite (tasks/pipeline_cfg fixed; only models vary per phase).
    task = SimpleEvalTask(
        task_id=f"gsm8k:{args.gsm8k_size}exs:{'train' if from_train_split else 'test'}:seed={args.gsm8k_seed}",
        examples=load_gsm8k_examples(size=args.gsm8k_size, seed=args.gsm8k_seed, from_train_split=from_train_split),
        validator_fn=gsm8k_main_validator,
        success_measure_fn=gsm8k_success_measure,
        problem_fn=default_problem_str,
        expected_fn=default_expected_repr,
    )

    ladder_specs = _parse_model_specs_csv(
        args.models,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if not ladder_specs:
        raise SystemExit("No models parsed from --models")

    mode = PipelineMode.SYMBOLIC_HYBRID
    pipeline_cfg = PipelineConfig(
        max_steps=20, 
        explain=False,
        use_termination_checks=False,
        use_final_termination_check=True,
        selector_num_candidates=3
        )
    base_suite = EvaluationSuite(
        name=f"Symbolic Hybrid On GSM8K ({args.gsm8k_size}EX)",
        tasks=[task],
        pipeline_mode=mode,
        model_by_role=ModelMapping.set_spec_to_all_roles(ladder_specs[0], mode),
        prompt_overrides=None,
        pipeline_cfg=pipeline_cfg,
        keep_all_outcomes=True,
        keep_random_k=0,
        seed=0,
    )

    phase1_root = _ensure_dir(where_root / "phase-1")
    phase2_root = _ensure_dir(where_root / "phase-2")
    (where_root / "README.json").write_text(
        json.dumps(
            {
                "analysis": "where-it-breaks",
                "layout": {
                    "phase-1": str(phase1_root),
                    "phase-2": str(phase2_root),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_failures = not args.no_failures
    if args.phase in ("1", "all"):
        run_where_it_breaks_phase_1(
            base_suite=base_suite,
            ladder_specs=ladder_specs,
            artifacts_root=phase1_root,
            write_failures=write_failures,
            max_in_flight=args.max_in_flight,
        )
    if args.phase in ("2", "all"):
        # If phase 2 runs after phase 1 in the same invocation, prefer the empirically-derived
        # phase-1 accuracy ordering (best -> worst) instead of trusting the user-provided order.
        phase2_best_spec = ladder_specs[0]
        phase2_ladder_specs = list(ladder_specs[1:])
        if args.phase == "all":
            ranked_models = get_phase1_models_ordered_by_accuracy(phase1_root)
            spec_by_model = {s.model: s for s in ladder_specs}
            ranked_specs: List[ModelSpec] = [
                spec_by_model.get(
                    m,
                    ModelSpec(model=m, temperature=args.temperature, max_tokens=args.max_tokens),
                )
                for m in ranked_models
            ]
            if ranked_specs:
                phase2_best_spec = ranked_specs[0]
                phase2_ladder_specs = ranked_specs[1:]
        if args.phase2_reuse_run_dir is not None:
            run_where_it_breaks_phase_2_reuse_initial_symbolization(
                base_suite=base_suite,
                best_spec=phase2_best_spec,
                ladder_specs=phase2_ladder_specs,
                artifacts_root=phase2_root,
                source_run_dir=args.phase2_reuse_run_dir,
                write_failures=write_failures,
                max_in_flight=args.max_in_flight,
            )
        else:
            run_where_it_breaks_phase_2(
                base_suite=base_suite,
                best_spec=phase2_best_spec,
                ladder_specs=phase2_ladder_specs,
                artifacts_root=phase2_root,
                write_failures=write_failures,
                max_in_flight=args.max_in_flight,
            )
