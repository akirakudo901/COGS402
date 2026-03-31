import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _format_dataset_info(run_meta: Dict[str, Any]) -> str:
    dataset = _safe_get(run_meta, ["dataset"], {}) or {}
    name = _safe_get(dataset, ["name"], "unknown")
    split = _safe_get(dataset, ["split"], None)

    subset_spec = _safe_get(dataset, ["subset_spec"], {}) or {}
    subset_size = _safe_get(subset_spec, ["size"], None)
    subset_seed = _safe_get(subset_spec, ["seed"], None)
    random_sample = _safe_get(subset_spec, ["random_sample"], None)

    parts: List[str] = [str(name)]
    if split is not None:
        parts.append(f"split={split}")

    spec_parts: List[str] = []
    if subset_size is not None:
        spec_parts.append(f"n={subset_size}")
    if subset_seed is not None:
        spec_parts.append(f"seed={subset_seed}")
    if random_sample is not None:
        spec_parts.append(f"random_sample={random_sample}")

    if spec_parts:
        parts.append(f"({', '.join(spec_parts)})")
    return " ".join(parts)


def _format_model_info(run_meta: Dict[str, Any]) -> str:
    # Example: {"cot_solver": {"model": "...", "temperature": 0.5, ...}}
    model_specs_by_role = _safe_get(run_meta, ["model_specs_by_role"], {}) or {}
    if not isinstance(model_specs_by_role, dict) or not model_specs_by_role:
        return "unknown"

    role_summaries: List[str] = []
    for role, spec in model_specs_by_role.items():
        if not isinstance(spec, dict):
            continue
        model_name = spec.get("model")
        if model_name is None:
            continue
        # Keep output short but include key generation setting if present.
        temperature = spec.get("temperature")
        if temperature is None:
            role_summaries.append(f"{role}={model_name}")
        else:
            role_summaries.append(f"{role}={model_name}, temp={temperature}")

    return "; ".join(role_summaries) if role_summaries else "unknown"


def _format_task_type(run_meta: Dict[str, Any]) -> str:
    pipeline_mode = run_meta.get("pipeline_mode")
    ablation_variant_id: Optional[str] = None
    ablation = run_meta.get("ablation")
    if isinstance(ablation, dict):
        ablation_variant_id = ablation.get("variant_id")

    if pipeline_mode and ablation_variant_id:
        return f"{pipeline_mode} (ablation={ablation_variant_id})"
    if pipeline_mode:
        return str(pipeline_mode)
    if ablation_variant_id:
        return f"ablation={ablation_variant_id}"
    suite_name = run_meta.get("suite_name")
    return str(suite_name) if suite_name is not None else "unknown"


def _extract_dataset_title_components(run_meta: Dict[str, Any]) -> Tuple[str, Optional[int], str]:
    """
    Return (dataset_name, subset_size, task_mode) for use in plot titles.
    """
    dataset = _safe_get(run_meta, ["dataset"], {}) or {}
    name = str(_safe_get(dataset, ["name"], "unknown"))
    subset_spec = _safe_get(dataset, ["subset_spec"], {}) or {}
    subset_size = _safe_get(subset_spec, ["size"], None)
    pipeline_mode = str(run_meta.get("pipeline_mode", "unknown"))
    return name, subset_size, pipeline_mode


def _camel_case_from_snake_or_kebab(text: str) -> str:
    # Replace underscores with spaces, split, then title-case and join.
    text = text.replace("_", " ").replace("-", " ")
    parts = [p for p in text.split(" ") if p]
    return " ".join(p.capitalize() for p in parts) if parts else text


def _get_model_family_and_name(full_model: str) -> Tuple[str, str]:
    """
    Convert model like "meta-llama/llama-3.3-70b-instruct" to "meta-llama" and "llama-3.3-70b-ins".
    """
    if not full_model:
        return "unknown"
    split = full_model.split("/")
    if len(split) == 1:
        family, suffix = "unknown", split[0]
    elif len(split) == 2:
        family, suffix = split
    else:
        family, suffix = split[0], split[-1]
    suffix = suffix.replace("instruct", "ins")
    return family, suffix

def _infer_model_size_key(model_name: str) -> float:
    """
    Sort key for "model size" (descending):
    - Prefer the parameter-count pattern "<NUM>b" (e.g. "70b", "235b", "30b")
    - Otherwise use the first number found (e.g. "4.1" in "gpt-4.1-mini")
    """
    if not model_name:
        return float("-inf")

    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model_name, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    m = re.search(r"(\d+(?:\.\d+)?)", model_name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    return float("-inf")


def analyze_artifacts_folder(
    folder_path: str,
    *,
    plot: bool = False,
    plot_path: Optional[str] = None,
    show_plot: bool = False,
) -> List[Dict[str, Any]]:
    """
    Recursively scan `folder_path` for `run_meta.json` files and print, for each run:
    dataset info, model info, task type, and accuracy.
    """
    run_metas: List[Dict[str, Any]] = []
    raw_run_meta_objects: List[Dict[str, Any]] = []

    for root, _dirs, files in os.walk(folder_path):
        if "run_meta.json" not in files:
            continue
        run_meta_path = os.path.join(root, "run_meta.json")
        try:
            with open(run_meta_path, "r", encoding="utf-8") as f:
                run_meta = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {run_meta_path}: {e}")
            continue

        dataset_info = _format_dataset_info(run_meta)
        model_info = _format_model_info(run_meta)
        task_type = _format_task_type(run_meta)
        accuracy = run_meta.get("overall_accuracy", None)
        run_id = run_meta.get("run_id", None)

        # Print a compact single-line summary.
        run_label = f"run_id={run_id}" if run_id is not None else "run_meta"
        print("-"*20)
        print(
            f"{run_meta_path} | {run_label} | \n"
            f" - dataset={dataset_info} | task={task_type}\n"
            f" - ** model={model_info} **\n"
            f" - ~~ accuracy={accuracy} ~~\n"
        )

        run_metas.append(
            {
                "run_meta_path": run_meta_path,
                "run_id": run_id,
                "dataset": dataset_info,
                "task_type": task_type,
                "model": model_info,
                "accuracy": accuracy,
            }
        )
        raw_run_meta_objects.append(run_meta)

    if plot and run_metas:
        try:
            import matplotlib.pyplot as plt

            # Assume all runs share the same dataset/mode; if not, we still use the first
            ds_name, subset_size, task_mode = _extract_dataset_title_components(raw_run_meta_objects[0])
            ds_title = _camel_case_from_snake_or_kebab(ds_name)
            task_title = _camel_case_from_snake_or_kebab(task_mode)

            if subset_size is not None:
                title = f"{ds_title} (n={subset_size}) - {task_title}"
            else:
                title = f"{ds_title} - {task_title}"

            # Build plotting entries, then group by family and sort by inferred size.
            entries: List[Dict[str, Any]] = []
            for rm, raw in zip(run_metas, raw_run_meta_objects):
                model_specs = _safe_get(raw, ["model_specs_by_role"], {}) or {}
                primary_model: Optional[str] = None
                if isinstance(model_specs, dict) and model_specs:
                    if "cot_solver" in model_specs and isinstance(model_specs["cot_solver"], dict):
                        primary_model = model_specs["cot_solver"].get("model")
                    else:
                        for spec in model_specs.values():
                            if isinstance(spec, dict) and "model" in spec:
                                primary_model = spec["model"]
                                break

                full_model = str(primary_model or "")
                family, short_name = (_get_model_family_and_name(full_model) if full_model else 
                                      _get_model_family_and_name(str(rm.get("model", ""))))
                size_key = _infer_model_size_key(short_name)

                acc = rm.get("accuracy")
                try:
                    acc_val = float(acc) if acc is not None else 0.0
                except (TypeError, ValueError):
                    acc_val = 0.0

                entries.append(
                    {
                        "label": short_name,
                        "family": family,
                        "size_key": size_key,
                        "accuracy": acc_val,
                    }
                )

            # Group by family, sort within family by decreasing size, and keep families together.
            families = sorted({e["family"] for e in entries})
            entries_sorted: List[Dict[str, Any]] = []
            for fam in families:
                fam_entries = [e for e in entries if e["family"] == fam]
                fam_entries.sort(key=lambda e: e["size_key"], reverse=True)
                entries_sorted.extend(fam_entries)

            x_labels = [e["label"] for e in entries_sorted]
            y_values = [e["accuracy"] for e in entries_sorted]

            # Assign one color per family.
            cmap = plt.get_cmap("tab20")
            family_to_color = {fam: cmap(i % 20) for i, fam in enumerate(families)}
            bar_colors = [family_to_color[e["family"]] for e in entries_sorted]

            plt.figure(figsize=(max(6, len(x_labels) * 0.8), 4))
            plt.bar(range(len(x_labels)), y_values, color=bar_colors)
            plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
            plt.ylabel("Accuracy")
            plt.title(title)
            # Legend: one entry per family.
            handles = [plt.Rectangle((0, 0), 1, 1, color=family_to_color[f]) for f in families]
            plt.legend(handles, families, title="Model family", loc="best")
            plt.tight_layout()

            if plot_path is None:
                plot_path = os.path.join(folder_path, "accuracy_by_model.png")

            if show_plot:
                plt.show()
            else:
                plt.savefig(plot_path, dpi=200)
                print(f"[INFO] Saved plot to {plot_path}")
        except Exception as e:
            print(f"[WARN] Failed to create plot: {e}")

    return run_metas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze run_meta.json artifacts.")
    parser.add_argument("folder", help="Folder to recursively search for run_meta.json files.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create a bar graph of model accuracy for the discovered runs.",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Where to save the plot PNG (default: <folder>/accuracy_by_model.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window (otherwise saves to PNG).",
    )
    args = parser.parse_args()

    analyze_artifacts_folder(args.folder, plot=args.plot, plot_path=args.plot_path, show_plot=args.show)
