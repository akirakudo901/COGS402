"""Artifact persistence and validation for evaluation runs."""

from __future__ import annotations

from typing import Any

__all__ = ["new_run_id", "persist_evaluation_run"]


def __getattr__(name: str) -> Any:
    if name == "new_run_id":
        from eval.artifact.artifact_persist import new_run_id as _new_run_id

        return _new_run_id
    if name == "persist_evaluation_run":
        from eval.artifact.artifact_persist import persist_evaluation_run as _persist_evaluation_run

        return _persist_evaluation_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
