import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.workflow import (
    WorkflowPaths,
    _build_question_results,
    _render_result_markdown,
    _run_stage_script,
)


class WorkflowOutputTests(unittest.TestCase):
    @patch("agent.workflow.subprocess.run")
    def test_stage_scripts_run_through_bash(self, run_mock):
        _run_stage_script("run_stage1.sh", ["--image", "page.png"])

        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], "bash")
        self.assertTrue(command[1].endswith("scripts/run_stage1.sh"))
        self.assertEqual(command[2:], ["--image", "page.png"])
        self.assertTrue(run_mock.call_args.kwargs["check"])

    def test_workflow_preserves_original_image_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = WorkflowPaths(Path(tmp))
            paths.ensure()
            visible = {path.name for path in Path(tmp).iterdir()}
            self.assertEqual(
                visible,
                {"image", "preprocessed", "api_markdown", "code_outputs", "agent_outputs"},
            )

    def test_questions_and_answers_are_paired_in_question_order(self):
        verified_markdown = "16. First question\n\n17. Second question"
        report = {
            "items": [
                {"kind": "q", "qno": 16, "status": "accept_strict", "v_after_repair": "Y"},
                {"kind": "q", "qno": 17, "status": "accept_strict", "v_after_repair": "Y"},
            ]
        }
        answers = [
            {"qno": 17, "text": "answer 17", "status": "ok", "verdict": "Y"},
            {"qno": 16, "text": "answer 16", "status": "ok", "verdict": "Y"},
        ]

        questions = _build_question_results(verified_markdown, report, answers)
        rendered = _render_result_markdown("page", questions)

        self.assertEqual([question["qno"] for question in questions], [16, 17])
        self.assertEqual(questions[0]["handwritten_answer"]["text"], "answer 16")
        self.assertLess(rendered.index("answer 16"), rendered.index("## 题目 17"))
        self.assertGreater(rendered.index("answer 17"), rendered.index("## 题目 17"))
        self.assertNotIn("题干校验", rendered)
        self.assertNotIn("答案证据", rendered)
        self.assertNotIn("证据说明", rendered)


if __name__ == "__main__":
    unittest.main()
