import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from llm_prolog.cot_baseline import CoTResult
from llm_prolog.pipeline import PipelineConfig
from llm_prolog.symbolic.types import (
    AnswerSpec,
    Fact,
    PipelineResult,
    PipelineStep,
    Premise,
    Predicate,
    SelectorDecision,
    Term,
)
from eval.artifact.artifact_persist import persist_evaluation_run
from eval.artifact.validate_artifacts import validate_run_dir
from eval.eval_suite import (
    ExampleOutcome,
    EvaluationSuite,
    ModelMapping,
    ModelSpec,
    PipelineMode,
    SimpleEvalTask,
    SuiteReport,
    TaskReport,
)


class _Example:
    def __init__(self, *, id: str, problem: str, ground_truth):
        self.id = id
        self.problem = problem
        self.ground_truth = ground_truth


class SerializationRoundTripTests(unittest.TestCase):
    def test_cot_result_roundtrip_json(self):
        original = CoTResult(
            answer_text="5",
            reasoning="some reasoning...",
            model="test-model",
        )
        payload = original.to_json_dict()
        restored = CoTResult.from_json_dict(payload)
        self.assertEqual(restored, original)

    def test_symbolic_pipeline_result_roundtrip_json(self):
        # Minimal symbolic graph: two background facts -> one derived answer fact.
        p1 = Premise(
            id=1,
            clause=Fact(predicate=Predicate(name="bonus_one", args=())),
            nl="bonus one",
            source="selector_background",
            parent_ids=None,
        )
        p2 = Premise(
            id=2,
            clause=Fact(predicate=Predicate(name="bonus_two", args=())),
            nl="bonus two",
            source="selector_background",
            parent_ids=None,
        )

        answer_p = Premise(
            id=3,
            clause=Fact(
                predicate=Predicate(
                    name="answer",
                    args=(Term.constant("42"),),
                )
            ),
            nl="final answer",
            source="inference",
            parent_ids=[1, 2],
        )

        answer_spec = AnswerSpec(target=Predicate(name="answer", args=(Term.variable("X"),)))

        decision = SelectorDecision(
            selected_premise_ids=[1, 2],
            proposed_new_premise="answer(42)",
            background_premises=["bonus_one.", "bonus_two."],
            is_answer_goal=True,
            should_stop=False,
        )

        step = PipelineStep(
            step_index=0,
            used_premise_ids=[1, 2],
            new_premise=answer_p,
            decision=decision,
            success=True,
            note=None,
        )

        original = PipelineResult(
            success=True,
            answer_premise=answer_p,
            steps=[step],
            answer_spec=answer_spec,
            final_premises=[p1, p2, answer_p],
            reason="answer_head_matched",
        )
        payload = original.to_json_dict()
        restored = PipelineResult.from_json_dict(payload)
        self.assertEqual(restored, original)


class ArtifactPersistenceTests(unittest.TestCase):
    def test_persist_outputs_cot_includes_reasoning(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts_root = Path(td)

            ex = _Example(id="0", problem="problem", ground_truth=5)
            result = CoTResult(answer_text="5", reasoning="step 1 ...", model="m")

            task = SimpleEvalTask(
                task_id="cot_task",
                examples=[ex],
                validator_fn=lambda r, mode: 5.0,
                success_measure_fn=lambda _ex, _obt: True,
                problem_fn=lambda e: e.problem,
                expected_fn=lambda e: str(e.ground_truth),
            )

            spec = ModelSpec(model="m", temperature=0.0, max_tokens=None)
            suite = EvaluationSuite(
                name="cot_suite",
                tasks=[task],
                pipeline_mode=PipelineMode.COT_BASELINE,
                model_by_role=ModelMapping.set_spec_to_all_roles(spec, PipelineMode.COT_BASELINE),
                prompt_overrides={},
                pipeline_cfg=PipelineConfig(max_steps=2, explain=True),
                keep_all_outcomes=True,
                keep_random_k=0,
                seed=0,
            )

            outcome = ExampleOutcome(
                idx=0,
                example_id="0",
                problem=ex.problem,
                expected=str(ex.ground_truth),
                obtained="",
                result=result,
                correct=True,
                error=None,
            )
            trep = TaskReport(
                task_id=task.task_id,
                pipeline_mode=PipelineMode.COT_BASELINE,
                total=1,
                correct=1,
                accuracy=1.0,
                outcomes=(outcome,),
            )
            suite_report = SuiteReport(pipeline_mode=PipelineMode.COT_BASELINE, task_reports=(trep,))

            dataset_meta = {"name": "gsm8k", "subset_spec": {"size": 1, "seed": 0, "random_sample": True}}
            run_dir = persist_evaluation_run(
                artifacts_root=artifacts_root,
                run_id="test_run_cot",
                suite=suite,
                suite_report=suite_report,
                dataset=dataset_meta,
                ablation={"variant_id": "x", "component_overrides": {}},
                write_failures=False,
            )
            ok, errs = validate_run_dir(run_dir)
            self.assertTrue(ok, "\n".join(errs))

            # Spot-check the persisted output reasoning.
            examples_path = run_dir / "examples.jsonl"
            row = examples_path.read_text(encoding="utf-8").strip()
            self.assertTrue(row)
            import json

            obj = json.loads(row)
            self.assertEqual(obj["output"]["result_type"], "CoTResult")
            self.assertEqual(obj["output"]["reasoning"], "step 1 ...")

    def test_persist_outputs_symbolic_includes_pipeline_steps(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts_root = Path(td)
            ex = _Example(id="0", problem="problem", ground_truth=42)

            # Result graph.
            p1 = Premise(
                id=1,
                clause=Fact(predicate=Predicate(name="bonus_one", args=())),
                nl="bonus one",
                source="selector_background",
                parent_ids=None,
            )
            p2 = Premise(
                id=2,
                clause=Fact(predicate=Predicate(name="bonus_two", args=())),
                nl="bonus two",
                source="selector_background",
                parent_ids=None,
            )
            answer_p = Premise(
                id=3,
                clause=Fact(predicate=Predicate(name="answer", args=(Term.constant("42"),))),
                nl="final answer",
                source="inference",
                parent_ids=[1, 2],
            )
            answer_spec = AnswerSpec(target=Predicate(name="answer", args=(Term.variable("X"),)))
            decision = SelectorDecision(
                selected_premise_ids=[1, 2],
                proposed_new_premise="answer(42)",
                background_premises=["bonus_one.", "bonus_two."],
                is_answer_goal=True,
                should_stop=False,
            )
            step = PipelineStep(
                step_index=0,
                used_premise_ids=[1, 2],
                new_premise=answer_p,
                decision=decision,
                success=True,
                note=None,
            )
            result = PipelineResult(
                success=True,
                answer_premise=answer_p,
                steps=[step],
                answer_spec=answer_spec,
                final_premises=[p1, p2, answer_p],
                reason="answer_head_matched",
            )

            task = SimpleEvalTask(
                task_id="sym_task",
                examples=[ex],
                validator_fn=lambda r, mode: 42.0,
                success_measure_fn=lambda _ex, _obt: True,
                problem_fn=lambda e: e.problem,
                expected_fn=lambda e: str(e.ground_truth),
            )

            spec = ModelSpec(model="m", temperature=0.0, max_tokens=None)
            suite = EvaluationSuite(
                name="sym_suite",
                tasks=[task],
                pipeline_mode=PipelineMode.SYMBOLIC_HYBRID,
                model_by_role=ModelMapping.set_spec_to_all_roles(spec, PipelineMode.SYMBOLIC_HYBRID),
                prompt_overrides={},
                pipeline_cfg=PipelineConfig(max_steps=2, explain=True),
                keep_all_outcomes=True,
                keep_random_k=0,
                seed=0,
            )

            outcome = ExampleOutcome(
                idx=0,
                example_id="0",
                problem=ex.problem,
                expected=str(ex.ground_truth),
                obtained="",
                result=result,
                correct=True,
                error=None,
            )
            trep = TaskReport(
                task_id=task.task_id,
                pipeline_mode=PipelineMode.SYMBOLIC_HYBRID,
                total=1,
                correct=1,
                accuracy=1.0,
                outcomes=(outcome,),
            )
            suite_report = SuiteReport(
                pipeline_mode=PipelineMode.SYMBOLIC_HYBRID,
                task_reports=(trep,),
            )

            dataset_meta = {"name": "gsm8k", "subset_spec": {"size": 1, "seed": 0, "random_sample": True}}
            run_dir = persist_evaluation_run(
                artifacts_root=artifacts_root,
                run_id="test_run_sym",
                suite=suite,
                suite_report=suite_report,
                dataset=dataset_meta,
                ablation={"variant_id": "x", "component_overrides": {}},
                write_failures=False,
            )
            ok, errs = validate_run_dir(run_dir)
            self.assertTrue(ok, "\n".join(errs))

            examples_path = run_dir / "examples.jsonl"
            import json

            obj = json.loads(examples_path.read_text(encoding="utf-8").strip())
            self.assertEqual(obj["output"]["result_type"], "PipelineResult")
            self.assertEqual(len(obj["output"]["steps"]), 1)
            self.assertEqual(len(obj["output"]["final_premises"]), 3)


if __name__ == "__main__":
    unittest.main()

