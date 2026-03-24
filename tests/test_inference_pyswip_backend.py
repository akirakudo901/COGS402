import os
import unittest
from unittest import mock

from llm_prolog.symbolic import inference
from llm_prolog.symbolic.types import Fact, Predicate, Premise, Rule, Term


class _FakeBackend:
    def unify_predicates(self, a, b, subst=None):
        return inference._py_unify_predicates(a, b, subst)


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
            with mock.patch.object(inference, "_get_prolog_backend", return_value=_FakeBackend()):
                a = Predicate(name="likes", args=(Term.variable("X"), Term.constant("music")))
                b = Predicate(name="likes", args=(Term.constant("alice"), Term.variable("Y")))
                subst = inference.unify_predicates(a, b)
                self.assertIsNotNone(subst)
                assert subst is not None
                self.assertIn("X", subst)
                self.assertEqual(subst["X"], Term.constant("alice"))

    def test_infer_new_premise_reduces_rule_fact_with_backend_unifier(self):
        with mock.patch.dict(os.environ, {"LLM_PROLOG_INFERENCE_POLICY": "strict"}):
            with mock.patch.object(inference, "_get_prolog_backend", return_value=_FakeBackend()):
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


if __name__ == "__main__":
    unittest.main()
