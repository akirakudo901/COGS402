"""
Per-dataset evaluation task registry.

Each dataset module should expose either:
- TASKS: dict[str, EvalTask]
- get_tasks(): list[EvalTask]
"""

from __future__ import annotations

from typing import Dict

from eval.eval_suite import EvalTask

from .eval_entailmentbank import TASKS as _ENTAILMENTBANK_TASKS
from .eval_gsm8k import TASKS as _GSM8K_TASKS


TASK_REGISTRY: Dict[str, EvalTask] = {}
TASK_REGISTRY.update(_GSM8K_TASKS)
TASK_REGISTRY.update(_ENTAILMENTBANK_TASKS)

