# COGS402

Repository for **COGS402** (2025 Winter Term 2). It contains an **LLM–Prolog** pipeline that combines large language models with symbolic reasoning in a Horn-clause style to solve math word problems.

## Overview

The pipeline takes a **natural-language problem and query** (e.g. a GSM8K-style math word problem), converts them into **Prolog-style symbolic premises**, then **iteratively deduces new facts** by combining premises until an answer is derived. Prolog is used here as a syntax for Horn clauses, not as a full Prolog engine.

Evaluation is done by comparing the derived answer with a known ground truth (e.g. numeric answers for GSM8K).

## Repository structure

```
COGS402/
├── README.md
├── .gitignore
└── llm-prolog/          # LLM–Prolog pipeline implementation
    ├── config.py        # OpenRouter API configuration
    ├── llm_client.py    # OpenRouter chat completions client
    ├── types.py         # Horn-clause types (Term, Predicate, Fact, Rule, Premise, etc.)
    ├── nl_symbol_converter.py   # NL → symbolic premises (LLM)
    ├── selector.py      # Chooses which premises to combine next (LLM)
    ├── inference.py     # Symbolic inference (unification, rule+fact → new fact)
    ├── symbol_nl_converter.py   # Symbolic premises → NL explanations (LLM)
    ├── pipeline.py      # Orchestrates the full pipeline
    ├── eval_gsm8k.py    # GSM8K-style evaluation harness
    ├── llm_prolog.py    # Module-level description and notes
    └── llm_prolog_plan.md       # Detailed pipeline design document
```

## Pipeline flow

1. **NL–Symbol converter**  
   An LLM turns the problem and query into:
   - A set of initial **premises** (facts and rules in Prolog-like syntax)
   - An **answer spec**: the target predicate we want to prove (e.g. `answer(8)`).

2. **Iteration** (until answer found or max steps):
   - **Selector** (LLM): Decides which two premises to combine next, can propose extra background premises, and can signal stop.
   - **Inference** (symbolic): Unifies and applies rules (e.g. rule + fact → new fact). No LLM.
   - **Termination check**: If the new premise matches the answer head, the pipeline succeeds.
   - **Symbol–NL converter** (optional): Adds natural-language explanations for the derived premises.

3. **Evaluation**  
   For GSM8K-style tasks, the final derived fact is compared to the ground-truth answer (e.g. numeric).

Design details are in `llm-prolog/llm_prolog_plan.md`.

## Requirements
- **Python 3**.
- **OpenRouter API key**: set `OPENROUTER_API_KEY` in your environment.
- **Dependencies**: the code uses `requests` for the OpenRouter API.

## Configuration

- **OpenRouter**: API key and optional overrides (e.g. `OPENROUTER_MODEL`) are handled in `llm-prolog/config.py`. Default model is `openai/gpt-4.1-mini`.
- **Pipeline**: `PipelineConfig` in `pipeline.py` supports `max_steps` and an `explain` flag for NL explanations.

## Usage

**Run a single GSM8K-style example** (hard-coded in `eval_gsm8k.py`):

```bash
cd llm-prolog
export OPENROUTER_API_KEY="your-key"
python -m eval_gsm8k
```

**Use the pipeline from code**:

```python
from llm_prolog.llm_client import LLMClient
from llm_prolog.pipeline import run_pipeline, PipelineConfig

result = run_pipeline(
    problem="Alice has 3 apples. She buys 5 more. How many apples does Alice have now?",
    query="How many apples does Alice have now?",
    config=PipelineConfig(max_steps=8, explain=True),
)
print(result.success, result.answer_premise, result.reason)
```

**Batch evaluation**: call `evaluate_examples()` from `eval_gsm8k` with an iterable of `GSM8KExample(problem=..., query=..., ground_truth=...)`.