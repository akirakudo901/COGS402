"""
Chain-of-Thought (CoT) baseline utilities.

This module defines:
- CoTResult: a simple result container for CoT runs
- run_cot_baseline: a helper to run a CoT-style solve with an LLMClient
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

from .llm_client.llm_client import LLMClient
from .llm_executor import LLMExecutor


@dataclass(frozen=True)
class CoTResult:
    answer_text: str
    reasoning: Optional[str] = None
    model: Optional[str] = None


_COGS402_WEI_TABLE20_MATH_WORD_PROBLEM_PROMPT_PREFIX = (
    "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
    "After they are done, there will be 21 trees. How many trees did the grove workers plant today? \n"
    "A: There are 15 trees originally. Then there were 21 trees after some more were planted. "
    "So there must have been 21 - 15 = 6. The answer is 6. \n"
    "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? \n"
    "A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. \n"
    "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? \n"
    "A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
    "After eating 35, they had 74 - 35 = 39. The answer is 39. \n"
    "Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
    "How many lollipops did Jason give to Denny? \n"
    "A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
    "So he gave Denny 20 - 12 = 8. The answer is 8. \n"
    "Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
    "How many toys does he have now? \n"
    "A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
    "5 + 4 = 9. The answer is 9. \n"
    "Q: There were nine computers in the server room. Five more computers were installed each day, "
    "from monday to thursday. How many computers are now in the server room? \n"
    "A: There were originally 9 computers. For each of 4 days, 5 more computers were added. "
    "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. \n"
    "Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
    "How many golf balls did he have at the end of wednesday? \n"
    "A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
    "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. \n"
    "Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? \n"
    "A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
    "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8. "
)


def _build_wei_table20_math_word_problem_prompt(problem: str) -> str:
    # Wei et al.'s Appendix G (Table 20) uses few-shot exemplars with the format:
    # Q: <problem>\nA: <chain-of-thought and final statement>.
    problem_clean = problem.strip()
    return f"{_COGS402_WEI_TABLE20_MATH_WORD_PROBLEM_PROMPT_PREFIX}\nQ: {problem_clean}\nA: "


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
    # Use Wei et al. (2022) Chain-of-Thought prompting exemplars (Appendix G, Table 20).
    # The paper's prompt is entirely within the user message; we leave the system prompt
    # empty by default to avoid interfering with that exact formatting.
    system_prompt = system_prompt_override or ""
    user_content = _build_wei_table20_math_word_problem_prompt(problem)
    raw = llm.generate(
        system_prompt,
        user_content,
        model=getattr(model_spec, "model", None) if model_spec else None,
        temperature=getattr(model_spec, "temperature", None) if model_spec else None,
        max_tokens=getattr(model_spec, "max_tokens", None) if model_spec else None,
    )
    return _cot_result_from_raw(raw, model_spec)


def _cot_result_from_raw(raw: str, model_spec: Any | None) -> CoTResult:
    answer_text = raw.strip()

    # Backwards compatible parsing for older prompts.
    for line in raw.splitlines()[::-1]:
        if line.strip().upper().startswith("FINAL:"):
            answer_text = line.split(":", 1)[1].strip()
            break

    if answer_text == raw.strip():
        # Match Wei et al.-style math word problem solutions, e.g.:
        # "The answer is 6."
        candidates: list[str] = []
        patterns = [
            r"(?:The answer is|So the answer is)\s*(.+?)(?:\s*[\.\!\?]\s*|$)",
            r"####\s*([^\n]+)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, raw, flags=re.IGNORECASE | re.MULTILINE):
                part = m.group(1).strip()
                # Strip trailing punctuation without eating decimal points.
                part = part.rstrip(" \t\r\n").rstrip(".")
                if part:
                    candidates.append(part)
        if candidates:
            answer_text = candidates[-1]

    return CoTResult(
        answer_text=answer_text,
        reasoning=raw,
        model=getattr(model_spec, "model", None) if model_spec else None,
    )


async def run_cot_baseline_async(
    problem: str,
    *,
    llm_exec: LLMExecutor,
    model_spec: Any | None = None,
    system_prompt_override: str | None = None,
) -> CoTResult:
    """
    Async Chain-of-Thought baseline via LLMExecutor.
    """
    system_prompt = system_prompt_override or ""
    user_content = _build_wei_table20_math_word_problem_prompt(problem)
    raw = await llm_exec.generate(
        system_prompt,
        user_content,
        model=getattr(model_spec, "model", None) if model_spec else None,
        temperature=getattr(model_spec, "temperature", None) if model_spec else None,
        max_tokens=getattr(model_spec, "max_tokens", None) if model_spec else None,
    )
    return _cot_result_from_raw(raw, model_spec)

