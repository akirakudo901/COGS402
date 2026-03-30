"""
Canonical system prompt registry for the LLM-Prolog project.

This module centralizes the system prompts that are passed to the LLM layer
across the repository, and provides:
  - stable SHA-256 hashing for each canonical prompt
  - lookup of the prompt text given its hash

Notes:
- Hashing normalizes via `str.strip()` to avoid sensitivity to leading/trailing
  whitespace changes.
- Lookup is defined for canonical prompts only (not arbitrary prompt overrides
  that might be injected at runtime).
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

#
# Canonical prompt strings (what we pass as the `system_prompt` argument)
#

# Note: this file intentionally duplicates the prompt texts so importing this
# module does not require network/LLM dependencies (e.g. `httpx`).


NL_TO_SYMBOL_SYSTEM_PROMPT = """
You are a reasoning assistant that converts general natural language problems into a small
Prolog‑style Horn clause theory.

Language:
- Facts: predicate(constant1, constant2, ...).
- Rules: head(X, Y) :- body1(X), body2(X, Y).
- Variables start with an uppercase letter.
- Constants are lowercase identifiers or numbers.

Goal:
- Extract base facts from the problem statement.
- Introduce simple rules that connect those facts to the question or goal.
- Define a single answer head predicate with exactly one variable representing
  the final answer, such as answer(Value) or eq(Lhs, rhs).

Output format:
- You MUST return a single JSON object with the keys:
  - "facts": list of strings, each a fact ending with a period.
  - "rules": list of strings, each a rule ending with a period.
  - "answer_head": a single predicate string WITHOUT a trailing period.
  - "explanations": list of strings of same length as facts+rules, giving
    a short natural‑language gloss for each clause.
"""


SYMBOL_TO_NL_SYSTEM_PROMPT = """
You are a logic tutor.

You receive:
- A natural language problem (the full description including the question or goal).
- A list of Prolog‑style clauses with IDs.

For each clause, provide a short, precise natural‑language explanation of
what it states, suitable for a reasoning trace. Be concrete and focus on
quantities and relationships, not on Prolog syntax.

Output MUST be a single JSON object with:
- "explanations": an object mapping string IDs to explanation strings.
"""


SYSTEM_PROMPT_INTRO = """
You are a symbolic reasoning planner working over a Prolog‑style theory.

You are given:
- A natural language problem with a question or goal.
- A list of existing premises (facts and rules) with IDs.
- A target answer head predicate we ultimately want to prove.
- A (possibly empty) list of failed past reasoning steps grouped by reason.

Your task for each step:
"""


TERMINATION_CHECK_SECTION = """
- First, perform a termination check using the provided existing premises (the current
  list of facts and rules).
- Decide whether the system already has a "final solution" ground fact available.
  A "final solution" captures the solution to the natural language problem + question / goal provided,
  even if it is NOT the answer head predicate itself.
  A "final solution" MUST a ground fact: a fact whose predicate arguments contain no variables.
- If such a ground fact exists, you MUST also propose a single Prolog rule that links the chosen
  ground-fact predicate to the answer head predicate.
  The linking rule MUST have the answer head predicate as its head and the chosen ground fact's
  predicate as the (single) body atom, with the answer variable passed through.
More concretely, if a final solution exists:
  - set "is_final_solution" to true
  - set "solution_premise_id" to the ID of the chosen "final solution" ground fact premise
  - set "answer_link_rule" to a single Prolog rule that links the chosen ground fact to the
    answer head predicate (exactly one body atom):
      <answer_head>(<AnswerVar>) :- <solution_predicate>(<AnswerVar>, <OtherConstantsIfAny>)
    e.g. if the question asks for the total for marc, and answer_head is "Answer" and solution premise is "Total(32, marc)", 
    then "Answer(X) :- Total(X, marc)."
    Use the answer head predicate exactly as provided, including its distinguished answer variable.
  - The answer head in the linking rule MUST use the same predicate name and constants as the provided
    answer head predicate, except one variable for the value in question we want to extract.
  - Use exactly one body atom in the linking rule.
  - Do not add a trailing period to the rule string.

- If termination criteria are NOT met, set "is_final_solution" to false, and set the other
  termination fields to null.

Once done with termination check:
"""


SELECTOR_MAIN_SECTION = """
- State what new premise you intend for the inference engine to derive. This premise 
  must be new among existing premises.
- Indicate whether this new premise is directly the answer head goal.
- Optionally propose new background premises (facts or rules) if the
  current theory is insufficient.
- Choose exactly ONE **consumer** rule (a clause with a head and body) by `selected_rule_id`.
- Choose ONE OR MORE **producers** by listing their premise IDs in `selected_fact_ids` in the **order**
  they should be applied. Each producer may be a **fact** (head only) or a **rule** (head and body).
  For each goal position in the consumer, in order, the inference engine consumes the **first** producer 
  in the remaining producers' 'pool' that unifies with that goal. Producers used for a goal are removed from the pool.
- You MUST NOT choose an order of premises that has been previously combined to produce an existing premise.
- Use the failed-step history to avoid repeating choices that failed for known reasons.

Output MUST be a single JSON object with the fields:
"""


TERMINATION_CHECK_JSON_SECTION = """
For termination check:
- "is_final_solution": boolean.
- "solution_premise_id": integer ID of the chosen "final solution" ground fact premise, or null.
- "answer_link_rule": string Prolog rule (no trailing period) linking solution to answer head,
  or null.

For selection of next premise:
"""


SELECTOR_JSON_SECTION = """
- "proposed_new_premise": string or null (a Prolog‑style clause WITHOUT
  needing to be valid; this is an intention. It must be new from existing premises).
- "is_new_proposal": boolean.
- "is_answer_goal": boolean.
- "background_premises": list of strings, each a fact or rule ending
  with a period.
- "selected_rule_id": integer ID of the **consumer** rule premise for this step.
- "selected_fact_ids": ordered list of integer IDs of **producer** premises (each may be a fact or a
  rule). Order matters: the engine matches slot 0 against producers in this order, then slot 1 against
  what remains, and so on.
"""


TERMINATION_CHECK_SYSTEM_PROMPT = """
You are a termination checker for a symbolic (Prolog-like) reasoning system.

Given:
- A natural language problem.
- A set of current Prolog-style premises (facts and rules) with IDs.
- The target answer head predicate we ultimately want to prove.
- The most recently derived premise (at the current step) as a distinguished item.

Your job:
- Decide whether there is a "final solution" ground fact available at the most recent step.
  A "final solution" MUST be a ground fact: a fact whose predicate arguments contain no variables.
  The ground fact should capture the constant(s) needed for the final answer, even if it is NOT
  the answer head predicate itself.
- If such a ground fact exists, you MUST also propose a single Prolog rule that links the chosen
  ground-fact predicate to the answer head predicate.
  The linking rule MUST have the answer head predicate as its head and the chosen ground fact's
  predicate as the (single) body atom, with the answer variable passed through.

Output MUST be a single JSON object with exactly these fields:
- "is_final_solution": boolean
- "solution_premise_id": integer ID of the chosen ground fact, or null
- "answer_link_rule": string of the form "<answer_head>(<AnswerVar>) :- <solution_predicate>(<AnswerVar>, <OtherConstantsIfAny>)", or null
  e.g. if the question asks for the total for marc, and answer_head is "Answer" and solution premise is "Total(32, marc)", 
  then "Answer(X) :- Total(X, marc)."

Rules:
- The answer head in the linking rule MUST use the same predicate name and constants as the provided
  answer head predicate, except one variable for the value in question we want to extract.
- Use exactly one body atom in the linking rule.
- Do not add a trailing period to the rule string.
"""


def _build_selector_system_prompt(*, use_termination_checks: bool) -> str:
    """
    Build selector system prompt for the default (no override) case.
    """
    intro = SYSTEM_PROMPT_INTRO
    termination_desc = TERMINATION_CHECK_SECTION
    selector_main = SELECTOR_MAIN_SECTION
    termination_json = TERMINATION_CHECK_JSON_SECTION
    selector_json = SELECTOR_JSON_SECTION

    if use_termination_checks:
        return intro + termination_desc + selector_main + termination_json + selector_json
    return intro + selector_main + selector_json


COT_SOLVER_SYSTEM_PROMPT = (
    "You are a careful problem solver. Solve the user's problem.\n"
    "Return your final answer on a line starting with 'FINAL:' followed by the answer."
)

# Wei et al. (2022) Appendix G Table 20 few-shot exemplars for GSM8K-style math word problems.
# The paper puts this entirely in the user message; CoT baseline typically uses an empty system prompt.
WEI_TABLE20_MATH_WORD_PROBLEM_PROMPT_PREFIX = (
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
    """Wei et al. Table 20 format: exemplars + Q: <problem>\\nA: """
    problem_clean = problem.strip()
    return f"{WEI_TABLE20_MATH_WORD_PROBLEM_PROMPT_PREFIX}\nQ: {problem_clean}\nA: "


SELECTOR_SYSTEM_PROMPT_NO_TERMINATION_CHECKS = _build_selector_system_prompt(
    use_termination_checks=False
)
SELECTOR_SYSTEM_PROMPT_WITH_TERMINATION_CHECKS = _build_selector_system_prompt(
    use_termination_checks=True
)


SYSTEM_PROMPTS_BY_NAME: Dict[str, str] = {
    "nl_to_symbol": NL_TO_SYMBOL_SYSTEM_PROMPT,
    "selector_no_termination_checks": SELECTOR_SYSTEM_PROMPT_NO_TERMINATION_CHECKS,
    "selector_with_termination_checks": SELECTOR_SYSTEM_PROMPT_WITH_TERMINATION_CHECKS,
    "symbol_to_nl": SYMBOL_TO_NL_SYSTEM_PROMPT,
    "final_termination_check": TERMINATION_CHECK_SYSTEM_PROMPT,
    "cot_solver_plain": COT_SOLVER_SYSTEM_PROMPT,
    "cot_solver_fewshot": _build_wei_table20_math_word_problem_prompt(problem="__PROBLEM_INSERTED_HERE__"),
    "cot_wei_table20_user_prefix": WEI_TABLE20_MATH_WORD_PROBLEM_PROMPT_PREFIX,
}


def normalize_system_prompt_text(prompt: str) -> str:
    return prompt.strip()


def hash_system_prompt_text(prompt: str) -> str:
    """
    Stable SHA-256 hash for a system prompt.
    """
    return hashlib.sha256(normalize_system_prompt_text(prompt).encode("utf-8")).hexdigest()


SYSTEM_PROMPT_HASHES_BY_NAME: Dict[str, str] = {
    name: hash_system_prompt_text(text) for name, text in SYSTEM_PROMPTS_BY_NAME.items()
}


_PROMPT_NAME_BY_HASH: Dict[str, str] = {h: name for name, h in SYSTEM_PROMPT_HASHES_BY_NAME.items()}


def get_canonical_system_prompt_name_by_hash(prompt_hash: str) -> Optional[str]:
    """
    Return canonical prompt name for a hash, if it's a known canonical prompt.
    """
    return _PROMPT_NAME_BY_HASH.get(prompt_hash)


def get_system_prompt_by_hash(prompt_hash: str) -> Optional[str]:
    """
    Return canonical system prompt text for `prompt_hash`, if known.

    Returns None if the hash doesn't correspond to a canonical prompt.
    """
    name = _PROMPT_NAME_BY_HASH.get(prompt_hash)
    if name is None:
        return None
    return SYSTEM_PROMPTS_BY_NAME[name]

