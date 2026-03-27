import os
import unittest
from unittest import mock

from llm_prolog.symbolic import inference
from llm_prolog.symbolic.types import Fact, Predicate, Premise, Rule, Term



class PlaceholderRewriteTests(unittest.TestCase):
    def test_prefers_b_to_a_variable_mapping_for_shared_placeholder(self):
        a = Predicate(name="pred", args=(Term.variable("A"),))
        b = Predicate(name="pred", args=(Term.variable("B"),))
        resolved = {
            "A": Term.variable("_1342"),
            "B": Term.variable("_1342"),
        }

        rewritten = inference._rewrite_placeholder_var_var_bindings(resolved, a, b)

        self.assertEqual(rewritten["B"], Term.variable("A"))
        self.assertNotIn("A", rewritten)

    def test_keeps_non_placeholder_variable_bindings(self):
        a = Predicate(name="pred", args=(Term.variable("A"),))
        b = Predicate(name="pred", args=(Term.variable("B"),))
        resolved = {
            "A": Term.variable("X"),
            "B": Term.variable("X"),
        }

        rewritten = inference._rewrite_placeholder_var_var_bindings(resolved, a, b)

        self.assertEqual(rewritten, resolved)

    def test_many_vars_share_placeholder_master_is_first_from_a_order(self):
        a = Predicate(
            name="pred",
            args=(Term.variable("A1"), Term.variable("A2")),
        )
        b = Predicate(
            name="pred",
            args=(Term.variable("B1"), Term.variable("B2")),
        )
        resolved = {
            "A1": Term.variable("_p"),
            "A2": Term.variable("_p"),
            "B1": Term.variable("_p"),
            "B2": Term.variable("_p"),
        }

        rewritten = inference._rewrite_placeholder_var_var_bindings(resolved, a, b)

        self.assertEqual(rewritten["B1"], Term.variable("A1"))
        self.assertEqual(rewritten["B2"], Term.variable("A1"))
        self.assertEqual(rewritten["A2"], Term.variable("A1"))
        self.assertNotIn("A1", rewritten)


class InferenceBackendPolicyTests(unittest.TestCase):
    def setUp(self):
        inference._PROLOG_BACKEND = None
        inference._PROLOG_BACKEND_ERROR = None

    def test_strict_policy_raises_when_backend_unavailable(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "strict"}):
            with mock.patch.object(inference, "_get_prolog_backend", side_effect=RuntimeError("missing")):
                a = Predicate(name="bird", args=(Term.variable("X"),))
                b = Predicate(name="bird", args=(Term.constant("penguin"),))
                with self.assertRaises(RuntimeError):
                    inference.unify_predicates(a, b)

    def test_fallback_policy_uses_python_unifier_when_backend_unavailable(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "fallback"}):
            with mock.patch.object(inference, "_get_prolog_backend", side_effect=RuntimeError("missing")):
                a = Predicate(name="bird", args=(Term.variable("X"),))
                b = Predicate(name="bird", args=(Term.constant("penguin"),))
                subst = inference.unify_predicates(a, b)
                self.assertIsNotNone(subst)
                assert subst is not None
                self.assertEqual(subst["X"], Term.constant("penguin"))


class InferenceBackendIntegrationShapeTests(unittest.TestCase):
    def setUp(self):
        inference._PROLOG_BACKEND = None
        inference._PROLOG_BACKEND_ERROR = None

    def test_unify_predicates_returns_expected_substitution_shape(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "strict"}):
            a = Predicate(name="likes", args=(Term.variable("X"), Term.constant("music")))
            b = Predicate(name="likes", args=(Term.constant("alice"), Term.variable("Y")))
            subst = inference.unify_predicates(a, b)
            self.assertIsNotNone(subst)
            assert subst is not None
            self.assertIn("X", subst)
            self.assertEqual(subst["X"], Term.constant("alice"))

    def test_infer_new_premise_reduces_rule_fact_with_backend_unifier(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "strict"}):
            rule = Rule(
                head=Predicate(name="flightless", args=(Term.variable("B"),)),
                body=(Predicate(name="bird", args=(Term.variable("B"),)),),
            )
            fact = Fact(predicate=Predicate(name="bird", args=(Term.constant("penguin"),)))
            premises = [
                Premise(id=1, clause=rule),
                Premise(id=2, clause=fact),
            ]
            clause = inference.infer_new_premise(premises)
            self.assertIsInstance(clause, Fact)
            assert isinstance(clause, Fact)
            self.assertEqual(
                clause.predicate,
                Predicate(name="flightless", args=(Term.constant("penguin"),)),
            )

    def test_mathis_semantics_still_reduce_under_fallback(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "fallback"}):
            with mock.patch.object(inference, "_get_prolog_backend", side_effect=RuntimeError("missing")):
                rule = Rule(
                    head=Predicate(name="value", args=(Term.variable("Z"),)),
                    body=(Predicate(name="mathIs", args=(Term.variable("Z"), Term.constant("2+3"))),),
                )
                premises = [Premise(id=1, clause=rule)]
                clause = inference.infer_new_premise(premises)
                self.assertIsInstance(clause, Fact)
                assert isinstance(clause, Fact)
                self.assertEqual(
                    clause.predicate,
                    Predicate(name="value", args=(Term.constant("5"),)),
                )

    def test_mathis_relation_linear_single_var_under_fallback(self):
        """``X is 2*X+1`` as equality: X = -1 (not SWI ``is/2``)."""
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "fallback"}):
            with mock.patch.object(inference, "_get_prolog_backend", side_effect=RuntimeError("missing")):
                rule = Rule(
                    head=Predicate(name="value", args=(Term.variable("X"),)),
                    body=(
                        Predicate(
                            name="mathIs",
                            args=(Term.variable("X"), Term.constant("2*X+1")),
                        ),
                    ),
                )
                premises = [Premise(id=1, clause=rule)]
                clause = inference.infer_new_premise(premises)
                self.assertIsInstance(clause, Fact)
                assert isinstance(clause, Fact)
                self.assertEqual(
                    clause.predicate,
                    Predicate(name="value", args=(Term.constant("-1"),)),
                )

    def test_mathis_constant_folds_before_eval_under_fallback(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "fallback"}):
            with mock.patch.object(inference, "_get_prolog_backend", side_effect=RuntimeError("missing")):
                rule = Rule(
                    head=Predicate(name="value", args=(Term.variable("Z"),)),
                    body=(
                        Predicate(
                            name="mathIs",
                            args=(Term.variable("Z"), Term.constant("2+3+4")),
                        ),
                    ),
                )
                premises = [Premise(id=1, clause=rule)]
                clause = inference.infer_new_premise(premises)
                self.assertIsInstance(clause, Fact)
                assert isinstance(clause, Fact)
                self.assertEqual(
                    clause.predicate,
                    Predicate(name="value", args=(Term.constant("9"),)),
                )
    
    def test_mathis_constant_folds_before_eval_under_fallback_even_with_variable(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "fallback"}):
            with mock.patch.object(inference, "_get_prolog_backend", side_effect=RuntimeError("missing")):
                rule = Rule(
                    head=Predicate(name="value", args=(Term.variable("Z"),)),
                    body=(
                        Predicate(
                            name="mathIs",
                            args=(Term.variable("Z"), Term.constant("2+3+4+W")),
                        ),
                    ),
                )
                premises = [Premise(id=1, clause=rule)]
                clause = inference.infer_new_premise(premises)
                print(f"clause body: {clause.body!r}")
                self.assertIsInstance(clause, Rule)
                assert isinstance(clause, Rule)
                self.assertEqual(
                    clause.head,
                    Predicate(name="value", args=(Term.variable("Z"),)),
                )
                self.assertEqual(
                    clause.body,
                    (
                        Predicate(
                            name="mathIs",
                            args=(
                                Term.variable("Z"),
                                Predicate(
                                    name="+",
                                    args=(Term.constant("9"), Term.variable("W")),
                                ),
                            ),
                        ),
                    ),
                )

    def test_mathis_relation_solves_for_rhs_var_under_fallback(self):
        """``10 is 2*X`` binds ``X`` (LHS constant, RHS has single variable)."""
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "fallback"}):
            with mock.patch.object(inference, "_get_prolog_backend", side_effect=RuntimeError("missing")):
                rule = Rule(
                    head=Predicate(name="value", args=(Term.variable("X"),)),
                    body=(
                        Predicate(
                            name="mathIs",
                            args=(Term.constant("10"), Term.constant("2*X")),
                        ),
                    ),
                )
                premises = [Premise(id=1, clause=rule)]
                clause = inference.infer_new_premise(premises)
                self.assertIsInstance(clause, Fact)
                assert isinstance(clause, Fact)
                self.assertEqual(
                    clause.predicate,
                    Predicate(name="value", args=(Term.constant("5"),)),
                )

    def test_is_predicate_is_accepted_under_fallback(self):
        """`is/2` should reduce like `mathIs/2` even when SWI backend is missing."""
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "fallback"}):
            with mock.patch.object(inference, "_get_prolog_backend", side_effect=RuntimeError("missing")):
                rule = Rule(
                    head=Predicate(name="value", args=(Term.variable("Z"),)),
                    body=(
                        Predicate(
                            name="is",
                            args=(
                                Term.variable("Z"),
                                Predicate(
                                    name="+",
                                    args=(Term.constant("2"), Term.constant("3")),
                                ),
                            ),
                        ),
                    ),
                )
                premises = [Premise(id=1, clause=rule)]
                clause = inference.infer_new_premise(premises)
                self.assertIsInstance(clause, Fact)
                assert isinstance(clause, Fact)
                self.assertEqual(
                    clause.predicate,
                    Predicate(name="value", args=(Term.constant("5"),)),
                )

    def test_mathis_nested_rhs_constant_folding_under_fallback_even_with_variable(self):
        """RHS provided as nested operator Predicates should fold constants."""
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "fallback"}):
            with mock.patch.object(inference, "_get_prolog_backend", side_effect=RuntimeError("missing")):
                expr = Predicate(
                    name="+",
                    args=(
                        Predicate(
                            name="+",
                            args=(
                                Predicate(
                                    name="+",
                                    args=(Term.constant("2"), Term.constant("3")),
                                ),
                                Term.constant("4"),
                            ),
                        ),
                        Term.variable("W"),
                    ),
                )
                rule = Rule(
                    head=Predicate(name="value", args=(Term.variable("Z"),)),
                    body=(Predicate(name="mathIs", args=(Term.variable("Z"), expr)),),
                )
                premises = [Premise(id=1, clause=rule)]
                clause = inference.infer_new_premise(premises)
                self.assertIsInstance(clause, Rule)
                assert isinstance(clause, Rule)
                self.assertEqual(
                    clause.head,
                    Predicate(name="value", args=(Term.variable("Z"),)),
                )
                self.assertEqual(
                    clause.body,
                    (
                        Predicate(
                            name="mathIs",
                            args=(
                                Term.variable("Z"),
                                Predicate(
                                    name="+",
                                    args=(Term.constant("9"), Term.variable("W")),
                                ),
                            ),
                        ),
                    ),
                )

    def test_two_slots_pool_order_consumes_first_matching_producers(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "strict"}):
            consumer = Rule(
                head=Predicate(name="goal", args=()),
                body=(
                    Predicate(name="p", args=(Term.constant("a"),)),
                    Predicate(name="p", args=(Term.constant("b"),)),
                ),
            )
            fc = Fact(predicate=Predicate(name="p", args=(Term.constant("c"),)))
            fa = Fact(predicate=Predicate(name="p", args=(Term.constant("a"),)))
            fb = Fact(predicate=Predicate(name="p", args=(Term.constant("b"),)))
            premises = [
                Premise(id=1, clause=consumer),
                Premise(id=2, clause=fc),
                Premise(id=3, clause=fa),
                Premise(id=4, clause=fb),
            ]
            clause = inference.infer_new_premise(premises)
            self.assertIsInstance(clause, Fact)
            assert isinstance(clause, Fact)
            self.assertEqual(clause.predicate, Predicate(name="goal", args=()))

    def test_rule_producer_splices_body_at_slot(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "strict"}):
            consumer = Rule(
                head=Predicate(name="h", args=(Term.variable("B"),)),
                body=(Predicate(name="bird", args=(Term.variable("B"),)),),
            )
            prod = Rule(
                head=Predicate(name="bird", args=(Term.variable("X"),)),
                body=(Predicate(name="penguin", args=(Term.variable("X"),)),),
            )
            premises = [Premise(id=1, clause=consumer), Premise(id=2, clause=prod)]
            clause = inference.infer_new_premise(premises)
            self.assertIsInstance(clause, Rule)
            assert isinstance(clause, Rule)
            self.assertEqual(len(clause.body), 1)
            self.assertEqual(clause.body[0].name, "penguin")

    def test_infer_new_premise_preserves_head_variable_name_under_prolog_unify(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "strict"}):
            from llm_prolog.symbolic.types import parse_fact_or_rule

            p2 = Premise(
                id=2, nl="", parent_ids=[],
                clause=parse_fact_or_rule("total_cans(30)."),
            )
            p4 = Premise(
                id=4, nl="", parent_ids=[],
                clause=parse_fact_or_rule(
                    "paid_cans(Total, Paid) :- offer(buy_1_get_1_free), is(Paid, Total // 2)."
                ),
            )
            p5 = Premise(
                id=5, nl="", parent_ids=[],
                clause=parse_fact_or_rule(
                    "total_cost(Paid, Cost) :- normal_price(Price), is(Cost, Paid * Price)."
                ),
            )
            p6 = Premise(
                id=6, nl="", parent_ids=[],
                clause=parse_fact_or_rule(
                    "answer(Cost) :- total_cans(Total), paid_cans(Total, Paid), total_cost(Paid, Cost)."
                ),
            )

            clause = inference.infer_new_premise([p6, p2, p4, p5])
            self.assertIsInstance(clause, Rule)
            assert isinstance(clause, Rule)
            expected = parse_fact_or_rule(
                "answer(Cost) :- offer(buy_1_get_1_free), normal_price(Fresh2), is(Cost, *(15, Fresh2))."
            )
            # Check head predicate
            self.assertEqual(clause.head, expected.head)
            # Check body contents: length and predicates
            self.assertEqual(len(clause.body), len(expected.body))
            for actual_atom, expected_atom in zip(clause.body, expected.body):
                self.assertEqual(actual_atom, expected_atom)

    def test_infer_new_premise_requires_consumer_rule_when_multiple_premises(self):
        premises = [
            Premise(id=1, clause=Fact(predicate=Predicate(name="p", args=()))),
            Premise(id=2, clause=Fact(predicate=Predicate(name="q", args=()))),
        ]
        self.assertIsNone(inference.infer_new_premise(premises))

    def test_validate_inference_premise_selection_errors(self):
        bad = [
            Premise(id=1, clause=Fact(predicate=Predicate(name="p", args=()))),
            Premise(id=2, clause=Fact(predicate=Predicate(name="q", args=()))),
        ]
        msg = inference.validate_inference_premise_selection(bad)
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertIn("consumer", msg.lower())

        class _NotClause:
            pass

        bad2 = [
            Premise(id=1, clause=Rule(head=Predicate(name="h", args=()), body=())),
            Premise(id=2, clause=_NotClause()),  # type: ignore[arg-type]
        ]
        msg2 = inference.validate_inference_premise_selection(bad2)
        self.assertIsNotNone(msg2)
        assert msg2 is not None
        self.assertIn("producers", msg2.lower())

    def test_validate_inference_premise_selection_accepts_rule_and_producers(self):
        ok = [
            Premise(
                id=1,
                clause=Rule(
                    head=Predicate(name="g", args=()),
                    body=(Predicate(name="p", args=()),),
                ),
            ),
            Premise(id=2, clause=Fact(predicate=Predicate(name="p", args=()))),
        ]
        self.assertIsNone(inference.validate_inference_premise_selection(ok))


if __name__ == "__main__":
    unittest.main()
