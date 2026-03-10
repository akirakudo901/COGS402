"""
Chain-of-Thought (CoT) baseline utilities.

This module defines:
- CoTResult: a simple result container for CoT runs
- run_cot_baseline: a helper to run a CoT-style solve with an LLMClient
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .llm_client.llm_client import LLMClient


@dataclass(frozen=True)
class CoTResult:
    answer_text: str
    reasoning: Optional[str] = None
    model: Optional[str] = None


def run_cot_baseline(
    problem: str,
    *,
    llm: LLMClient,
    model_spec: Any | None = None,
    system_prompt_override: str | None = None,
) -> CoTResult:
    """
    Chain-of-Thought baseline.

    Returns a CoTResult containing the raw response and a best-effort extracted final answer.
    Dataset-specific validators should interpret CoTResult.answer_text appropriately.
    """
    system_prompt = system_prompt_override or (
        "You are a careful problem solver. Solve the user's problem.\n"
        "Return your final answer on a line starting with 'FINAL:' followed by the answer."
    )
    user_content = problem.strip()
    raw = llm.generate(
        system_prompt,
        user_content,
        model=getattr(model_spec, "model", None) if model_spec else None,
        temperature=getattr(model_spec, "temperature", None) if model_spec else None,
        max_tokens=getattr(model_spec, "max_tokens", None) if model_spec else None,
    )

    answer_text = raw.strip()
    for line in raw.splitlines()[::-1]:
        if line.strip().upper().startswith("FINAL:"):
            answer_text = line.split(":", 1)[1].strip()
            break
    return CoTResult(
        answer_text=answer_text,
        reasoning=raw,
        model=getattr(model_spec, "model", None) if model_spec else None,
    )

