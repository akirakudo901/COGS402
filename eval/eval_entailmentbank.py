"""
Evaluation for the EntailmentBank dataset.

Supports two problem types:
- Gold-only: problem description includes only the gold (relevant) premises.
- Gold + distractor: problem description includes both gold and distractor premises.

Uses the generic evaluation harness from eval_common with a custom validator
and success measure. Success is defined as the pipeline deriving a conclusion
that matches the answer spec (result.success); optional string matching
against the hypothesis can be enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from llm_prolog.symbolic.types import PipelineResult

from eval.eval_common import evaluate_examples, run_single_example


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntailmentBankExample:
    """
    Single EntailmentBank instance: question, hypothesis, and premise sentences.

    - question: The question to which the hypothesis is an answer.
    - hypothesis: The conclusion to derive to answer the question.
    - gold_premises: Sentences that are leaves of the gold entailment tree (relevant only).
    - distractor_premises: Irrelevant sentences (for task 2).
    """

    id: str
    question: str
    hypothesis: str
    gold_premises: tuple[str, ...]
    distractor_premises: tuple[str, ...] = ()


@dataclass
class EntailmentBankEvalCase:
    """
    Evaluation case with a concrete .problem string for the pipeline.

    Used so the generic harness (which expects .problem) can run on
    EntailmentBank. Built from EntailmentBankExample + mode (gold_only vs gold_plus_distractor).

    problem (str) : the full problem string provided to the LLM.
    question (str) : the question defining the main problem's inquiry.
    hypothesis (str) : the hypothesis to deduce, which is an answer to the question and provided
    """

    problem: str
    question: str
    hypothesis: str
    id: str
    gold_only: bool = True

    @property
    def expected(self) -> str:
        return self.hypothesis

    @property
    def ground_truth(self) -> str:
        return self.hypothesis


@dataclass
class EntailmentBankObtained:
    """Value extracted from the pipeline result for EntailmentBank."""

    success: bool
    derived_answer: Optional[str] = None


# ---------------------------------------------------------------------------
# Problem building (gold-only vs gold + distractors)
# ---------------------------------------------------------------------------


def build_problem(example: EntailmentBankExample, gold_only: bool) -> str:
    """
    Build the problem string passed to the pipeline.

    - gold_only=True: hypothesis + gold premises only (task 1).
    - gold_only=False: hypothesis + gold premises + distractor premises (task 2).
    """
    lines = [
        "Here is a science question:",
        f"  '{example.question.strip()}'",
        "The following hypothesis is its answer which we aim to derive:",
        f"  '{example.hypothesis.strip()}'",
        "",
        "To derive the hypothesis, use the following premises (which might contain irrelevant premises):",
    ]
    import random

    if gold_only:
        for i, p in enumerate(example.gold_premises, start=1):
            lines.append(f"  {i}. {p.strip()}")
    else:
        # Scramble the gold and distractor premises together
        combined_premises = [(p, 'gold') for p in example.gold_premises]
        if example.distractor_premises:
            combined_premises += [(p, 'distractor') for p in example.distractor_premises]
        random.shuffle(combined_premises)
        for i, (p, _) in enumerate(combined_premises, start=1):
            lines.append(f"  {i}. {p.strip()}")
    return "\n".join(lines)


def to_eval_cases(
    examples: Iterable[EntailmentBankExample],
    gold_only: bool,
) -> List[EntailmentBankEvalCase]:
    """Convert raw examples to eval cases with .problem set for the given mode."""
    return [
        EntailmentBankEvalCase(
            problem=build_problem(ex, gold_only=gold_only),
            question=ex.question,
            hypothesis=ex.hypothesis,
            id=ex.id,
            gold_only=gold_only,
        )
        for ex in examples
    ]


# ---------------------------------------------------------------------------
# Validator and success measure
# ---------------------------------------------------------------------------


def entailmentbank_validator(result: PipelineResult) -> EntailmentBankObtained:
    """
    Extract from the pipeline result: whether it succeeded and the derived answer string.
    """
    derived: Optional[str] = None
    if result.answer_premise is not None:
        derived = str(result.answer_premise.clause)
    return EntailmentBankObtained(
        success=result.success,
        derived_answer=derived,
    )


def entailmentbank_success_measure(
    example: EntailmentBankEvalCase,
    obtained: Optional[EntailmentBankObtained],
) -> bool:
    """
    Success = pipeline derived a conclusion (result.success).
    """
    if obtained is None:
        return False
    return obtained.success


# ---------------------------------------------------------------------------
# Data loading (optional: from JSON/JSONL)
# ---------------------------------------------------------------------------


def load_entailmentbank_jsonl(path: Path) -> List[EntailmentBankExample]:
    """
    Load EntailmentBank JSONL.

    Supports the official EntailmentBank v3 task files (e.g.
    `task_2_example1.jsonl`), where each line is a JSON object like:
    - "id": str
    - "context": str  (space-separated `sentN: ...` encoding)
    - "question": str
    - "answer": str
    - "hypothesis": str
    - "proof": str
    - "meta": {
        "triples": { "sent1": "...", ... }   # canonical sentence text map
        "distractors": ["sent7", ...]        # task 2: distractor sentence IDs
      }

    Premise extraction:
    - Sentence map prefers `meta.triples` when present; otherwise parses `context`.
    - If `meta.distractors` exists: treat those as distractors and everything else
      in the sentence map as gold premises (task 2).
    - Otherwise: try to treat sentence IDs referenced in `proof` as gold (task 1
      fallback). If that fails, treat all sentences as gold.
    """
    import json
    import re

    def _sent_id_sort_key(sid: str) -> int:
        m = re.match(r"^sent(\d+)$", sid)
        return int(m.group(1)) if m else 10**9

    def _parse_context_string(context: str) -> dict[str, str]:
        """
        Parse the official `context` string format:
          "sent1: ... sent2: ... sent10: ...".
        """
        ctx = (context or "").strip()
        if not ctx:
            return {}
        matches = list(re.finditer(r"\b(sent\d+)\s*:\s*", ctx))
        if not matches:
            return {}

        out: dict[str, str] = {}
        for i, m in enumerate(matches):
            sid = m.group(1)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(ctx)
            text = ctx[start:end].strip()
            if text:
                out[sid] = text
        return out

    examples: List[EntailmentBankExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ex_id = str(obj.get("id", str(len(examples))))
            meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}

            question = str(obj.get("question") or meta.get("question_text") or "")
            hypothesis = str(obj.get("hypothesis") or "")
            if not hypothesis:
                # Fallback: build something usable if hypothesis is missing.
                answer_text = str(obj.get("answer") or meta.get("answer_text") or "")
                hypothesis = answer_text.strip()

            # Prefer canonical mapping from meta.triples; fall back to parsing context.
            triples = meta.get("triples")
            sent_map: dict[str, str]
            if isinstance(triples, dict):
                sent_map = {str(k): str(v) for k, v in triples.items()}
            else:
                sent_map = _parse_context_string(str(obj.get("context") or ""))

            # Task 2: distractor IDs provided explicitly.
            distractor_ids_raw = meta.get("distractors", obj.get("distractors"))
            distractor_ids: List[str] = []
            if isinstance(distractor_ids_raw, list):
                distractor_ids = [str(x) for x in distractor_ids_raw]

            # Determine gold IDs.
            if distractor_ids:
                dset = set(distractor_ids)
                gold_ids = [sid for sid in sent_map.keys() if sid not in dset]
            else:
                # Task 1 (or unknown): try extracting referenced sent ids from proof.
                gold_ids = []
                proof = obj.get("proof")
                if isinstance(proof, str) and proof:
                    seen: set[str] = set()
                    for sid in re.findall(r"\b(sent\d+)\b", proof):
                        if sid in sent_map and sid not in seen:
                            gold_ids.append(sid)
                            seen.add(sid)
                if not gold_ids:
                    gold_ids = list(sent_map.keys())

            gold_premises = [
                sent_map[sid]
                for sid in sorted(gold_ids, key=_sent_id_sort_key)
                if sid in sent_map
            ]
            distractor_premises = [sent_map[sid] for sid in distractor_ids if sid in sent_map]

            examples.append(
                EntailmentBankExample(
                    id=ex_id,
                    question=question,
                    hypothesis=hypothesis,
                    gold_premises=tuple(gold_premises),
                    distractor_premises=tuple(distractor_premises),
                )
            )
    return examples


# ---------------------------------------------------------------------------
# Public evaluation API
# ---------------------------------------------------------------------------


def run_single_entailmentbank(
    example: EntailmentBankEvalCase,
    *,
    temperature: float = 0.5,
    max_steps: int = 5,
    explain: bool = True,
) -> PipelineResult:
    """Run the pipeline on a single EntailmentBank eval case and print results."""
    return run_single_example(
        example,
        entailmentbank_validator,
        entailmentbank_success_measure,
        temperature=temperature,
        max_steps=max_steps,
        explain=explain,
        show_derived_label="Derived conclusion",
        show_expected_label="Hypothesis",
    )


def evaluate_entailmentbank(
    examples: Iterable[EntailmentBankEvalCase],
    *,
    max_steps: int = 8,
    explain: bool = False,
) -> None:
    """Run the pipeline over EntailmentBank eval cases and print accuracy summary."""
    evaluate_examples(
        examples,
        entailmentbank_validator,
        entailmentbank_success_measure,
        max_steps=max_steps,
        explain=explain,
        show_derived_label="Derived conclusion",
        show_expected_label="Hypothesis",
    )


def evaluate_entailmentbank_gold_only(
    examples: Iterable[EntailmentBankExample],
    *,
    max_steps: int = 8,
    explain: bool = False,
) -> None:
    """Evaluate on EntailmentBank with only gold premises (task 1)."""
    cases = to_eval_cases(examples, gold_only=True)
    evaluate_entailmentbank(cases, max_steps=max_steps, explain=explain)


def evaluate_entailmentbank_with_distractors(
    examples: Iterable[EntailmentBankExample],
    *,
    max_steps: int = 8,
    explain: bool = False,
) -> None:
    """Evaluate on EntailmentBank with gold + distractor premises (task 2)."""
    cases = to_eval_cases(examples, gold_only=False)
    evaluate_entailmentbank(cases, max_steps=max_steps, explain=explain)


# ---------------------------------------------------------------------------
# Example / demo data (no external file)
# ---------------------------------------------------------------------------

EXAMPLE_ENTAILMENTBANK_1 = EntailmentBankExample(
    id="demo_1",
    question="Why does metal feel colder than wood at the same room temperature?",
    hypothesis="Metal feels colder than wood at the same temperature because it conducts heat better.",
    gold_premises=(
        "Metal is a good conductor of heat.",
        "Wood is a poor conductor of heat.",
        "Heat flows from warmer objects to cooler objects.",
    ),
    distractor_premises=(
        "Plants convert sunlight into energy through photosynthesis.",
        "Birds can fly using their wings."
    ),
)
"""
possible derivation:
question="Why does metal feel colder than wood at the same room temperature?",
hypothesis="Because it conducts heat better.",
gold_premises=(
    "Metal is a good conductor of heat.",
    "Wood is a poor conductor of heat.",
    "Heat flows from warmer objects to cooler objects.",
),
distractor_premises=(
    "Plants convert sunlight into energy through photosynthesis.",
    "Birds can fly using their wings."
),
"""


EXAMPLE_ENTAILMENTBANK_2 = EntailmentBankExample(
    id="demo_2",
    question="What do plants need to grow?",
    hypothesis="Plants need sunlight to grow.",
    gold_premises=(
        "Plants perform photosynthesis using light.",
        "Photosynthesis requires sunlight.",
    ),
    distractor_premises=(
        "The Earth rotates around the sun.",
        "Mammals are warm-blooded.",
    ),
)

if __name__ == "__main__":
    import sys

    demo_examples = [EXAMPLE_ENTAILMENTBANK_1, EXAMPLE_ENTAILMENTBANK_2]

    if len(sys.argv) > 1 and sys.argv[1] == "gold_only":
        print("=== EntailmentBank Task 1: Gold premises only ===\n")
        evaluate_entailmentbank_gold_only(demo_examples, max_steps=10)
    elif len(sys.argv) > 1 and sys.argv[1] == "with_distractors":
        print("=== EntailmentBank Task 2: Gold + distractor premises ===\n")
        evaluate_entailmentbank_with_distractors(demo_examples, max_steps=10)
    elif len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        path = Path(sys.argv[1])
        mode = sys.argv[2] if len(sys.argv) > 2 else "gold_only"
        loaded = load_entailmentbank_jsonl(path)
        print(f"Loaded {len(loaded)} examples from {path}; mode={mode}\n")
        if mode == "with_distractors":
            evaluate_entailmentbank_with_distractors(loaded, max_steps=10)
        else:
            evaluate_entailmentbank_gold_only(loaded, max_steps=10)
    else:
        print("Usage:")
        print("  python -m test.eval_entailmentbank gold_only")
        print("  python -m test.eval_entailmentbank with_distractors")
        print("  python -m test.eval_entailmentbank <path-to.jsonl> [gold_only|with_distractors]")
        print("\nRunning demo: gold_only with 2 built-in examples.\n")
        evaluate_entailmentbank_gold_only(demo_examples, max_steps=10)
