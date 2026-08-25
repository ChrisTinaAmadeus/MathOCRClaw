import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from agent.workflow import (
    BASELINE_PROMPT,
    BASELINE_PROMPT_PATH,
    WorkflowPaths,
    _build_question_context_manifest,
    _draft_questions_from_markdown,
    _normalize_final_questions,
    _page_work_root,
    _render_result_markdown,
    _run_stage_script,
    build_argparser,
    run_agent,
)
from proofread.vlm_client import VLMClient


class WorkflowOutputTests(unittest.TestCase):
    def test_api_timeouts_default_to_six_minutes(self):
        args = build_argparser().parse_args(["--image", "page.png"])

        self.assertEqual(args.baseline_timeout, 360)
        self.assertEqual(args.review_timeout, 360)

    def test_first_api_prompt_is_the_canonical_benchmark_prompt(self):
        self.assertEqual(
            BASELINE_PROMPT,
            BASELINE_PROMPT_PATH.read_text(encoding="utf-8").strip(),
        )
        self.assertIn("作废内容必须忽略", BASELINE_PROMPT)
        self.assertIn("禁止输出 `划去`", BASELINE_PROMPT)

    def test_page_work_root_contains_the_page_name_exactly_once(self):
        base = Path("/tmp/mathocr-workflow")

        self.assertEqual(_page_work_root(base, "page01"), base / "page01")
        self.assertEqual(_page_work_root(base / "page01", "page01"), base / "page01")

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

    def test_first_pass_markdown_is_parsed_locally_for_second_pass(self):
        questions = _draft_questions_from_markdown(
            "1. 求 $x$。\n\n### 手写答案\n\n$x=2$\n\n"
            "2. [无法识别]\n\n### 手写答案\n\n_未识别到手写答案。_"
        )

        self.assertEqual([question["qno"] for question in questions], ["1", "2"])
        self.assertEqual(questions[0]["student_answer"], "$x=2$")
        self.assertEqual(questions[1]["answer_status"], "no_answer")

    def test_section_heading_is_preserved_without_polluting_previous_answer(self):
        questions = _draft_questions_from_markdown(
            "8. 单选题\nA. 1\nB. 2\n\n### 手写答案\n\nA\n\n"
            "二、多项选择题：本题共 1 小题。\n\n"
            "9. 多选题\nA. 甲\nB. 乙\n\n### 手写答案\n\nAB"
        )

        self.assertEqual(questions[0]["student_answer"], "A")
        self.assertEqual(
            questions[1]["section_heading_before"],
            "二、多项选择题：本题共 1 小题。",
        )

    def test_second_pass_questions_preserve_parts_and_context_links(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {"qno": "016", "question_text": "16. Draft"},
            }
        ]
        review = {
            "questions": [
                {
                    "qno": "016",
                    "question_markdown": "16. Corrected",
                    "source_context_ids": ["C001"],
                    "handwritten_answer": {
                        "text": "(1) x=2\n(2) y=3",
                        "answer_parts": [
                            {
                                "label": "(1)",
                                "transcription": "x=2",
                                "final_answer": "x=2",
                                "status": "ok",
                            },
                            {
                                "label": "(2)",
                                "transcription": "y=3",
                                "final_answer": "y=3",
                                "status": "ok",
                            },
                        ],
                        "uncertain_fragments": [],
                        "status": "ok",
                        "evidence_note": "two visible lines",
                    },
                }
            ]
        }

        questions = _normalize_final_questions(review, contexts)

        self.assertEqual(questions[0]["qno"], 16)
        self.assertEqual(questions[0]["question_markdown"], "16. Corrected")
        self.assertEqual(len(questions[0]["handwritten_answer"]["answer_parts"]), 2)
        self.assertEqual(
            questions[0]["question_review"]["source_context_ids"],
            ["C001"],
        )

    def test_choice_and_fill_answers_drop_scratch_and_keep_only_final_answer(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "3",
                    "question_text": "3. 选择正确答案\nA. 1\nB. 2\nC. 3\nD. 4",
                    "student_answer": "A",
                    "question_type": "choice",
                },
            },
            {
                "context_id": "C002",
                "draft": {
                    "qno": "12",
                    "question_text": "12. 计算结果为 ______",
                    "student_answer": "$4$",
                    "question_type": "fill",
                },
            },
        ]
        review = {
            "questions": [
                {
                    "qno": "3",
                    "question_type": "choice",
                    "question_markdown": contexts[0]["draft"]["question_text"],
                    "handwritten_answer": {
                        "text": "$\\vec a\\cdot\\vec b=1$\nA",
                        "answer_parts": [
                            {
                                "label": "overall",
                                "transcription": "$\\vec a\\cdot\\vec b=1$\nA",
                                "final_answer": "A",
                                "status": "ok",
                            }
                        ],
                        "status": "ok",
                    },
                },
                {
                    "qno": "12",
                    "question_type": "fill",
                    "question_markdown": contexts[1]["draft"]["question_text"],
                    "handwritten_answer": {
                        "text": "$a^2=4$\n4",
                        "answer_parts": [
                            {
                                "label": "overall",
                                "transcription": "$a^2=4$\n4",
                                "final_answer": "4",
                                "status": "ok",
                            }
                        ],
                        "status": "ok",
                    },
                },
            ]
        }

        questions = _normalize_final_questions(review, contexts)

        self.assertEqual(questions[0]["handwritten_answer"]["text"], "A")
        self.assertEqual(questions[1]["handwritten_answer"]["text"], "$4$")
        self.assertIn(
            "removed_choice_scratch_work",
            questions[0]["question_review"]["lint_actions"],
        )
        self.assertIn(
            "removed_fill_scratch_work",
            questions[1]["question_review"]["lint_actions"],
        )

    def test_solution_parts_render_explicit_no_answer_markers(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "18",
                    "question_text": "18. (1) 证明；(2) 求值；(3) 求范围。",
                    "student_answer": "(2) $x=1$",
                    "question_type": "solution",
                },
            }
        ]
        review = {
            "questions": [
                {
                    "qno": "18",
                    "question_type": "solution",
                    "question_markdown": contexts[0]["draft"]["question_text"],
                    "handwritten_answer": {
                        "answer_parts": [
                            {"label": "(1)", "status": "no_answer"},
                            {
                                "label": "(2)",
                                "transcription": "$x=1$",
                                "final_answer": "$x=1$",
                                "status": "ok",
                            },
                            {"label": "(3)", "status": "no_answer"},
                        ],
                        "status": "partial",
                    },
                }
            ]
        }

        question = _normalize_final_questions(review, contexts)[0]
        rendered = _render_result_markdown("page", [question])

        self.assertIn("(1) _未识别到手写答案。_", rendered)
        self.assertIn("(2) $x=1$", rendered)
        self.assertIn("(3) _未识别到手写答案。_", rendered)

    def test_multiple_fill_parts_keep_every_final_value(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "12",
                    "question_text": "12. 填空：(1) _____；(2) _____。",
                    "student_answer": "(1) $2$\n(2) $3$",
                    "question_type": "fill",
                },
            }
        ]
        review = {
            "questions": [
                {
                    "qno": "12",
                    "question_type": "fill",
                    "question_markdown": contexts[0]["draft"]["question_text"],
                    "handwritten_answer": {
                        "text": "(1) $1+1=2$\n(2) $1+2=3$",
                        "answer_parts": [
                            {"label": "(1)", "final_answer": "2", "status": "ok"},
                            {"label": "(2)", "final_answer": "3", "status": "ok"},
                        ],
                        "status": "ok",
                    },
                }
            ]
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "(1) $2$\n(2) $3$")
        self.assertEqual(
            [part["final_answer"] for part in question["handwritten_answer"]["answer_parts"]],
            ["$2$", "$3$"],
        )

    def test_missing_second_pass_question_is_restored_by_qno(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "1",
                    "question_text": "1. 第一题",
                    "student_answer": "$x=1$",
                    "answer_status": "ok",
                    "question_type": "solution",
                },
            },
            {
                "context_id": "C002",
                "draft": {
                    "qno": "2",
                    "question_text": "2. 第二题",
                    "student_answer": "$y=2$",
                    "answer_status": "ok",
                    "question_type": "solution",
                },
            },
        ]
        review = {
            "questions": [
                {
                    "qno": "2",
                    "question_markdown": "2. 第二题（复核）",
                    "handwritten_answer": {"text": "$y=2$", "status": "ok"},
                }
            ]
        }

        questions = _normalize_final_questions(review, contexts)

        self.assertEqual([question["qno"] for question in questions], [1, 2])
        self.assertEqual(questions[0]["handwritten_answer"]["text"], "$x=1$")
        self.assertIn(
            "restored_question_from_first_pass",
            questions[0]["question_review"]["lint_actions"],
        )
        self.assertEqual(questions[1]["question_markdown"], "2. 第二题（复核）")

    def test_question_guard_restores_unreadable_stem_and_deleted_enumerator(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "16",
                    "question_text": "16. [无法识别]",
                    "student_answer": "$x=1$",
                    "question_type": "solution",
                },
            },
            {
                "context_id": "C002",
                "draft": {
                    "qno": "6",
                    "question_text": "6. 判断事件：①甲；②乙；③丙；④丁。\nA. 1\nB. 2",
                    "student_answer": "B",
                    "question_type": "choice",
                },
            },
        ]
        review = {
            "questions": [
                {
                    "qno": "16",
                    "question_markdown": "16.",
                    "handwritten_answer": {"text": "$x=1$", "status": "ok"},
                },
                {
                    "qno": "6",
                    "question_markdown": "6. 判断事件：①甲；②乙；③丙；丁。\nA. 1\nB. 2",
                    "handwritten_answer": {
                        "answer_parts": [
                            {"label": "overall", "final_answer": "B", "status": "ok"}
                        ],
                        "status": "ok",
                    },
                },
            ]
        }

        questions = _normalize_final_questions(review, contexts)

        self.assertEqual(questions[0]["question_markdown"], "16. [无法识别]")
        self.assertIn("④丁", questions[1]["question_markdown"])
        self.assertIn(
            "rolled_back_deleted_question_structure",
            questions[1]["question_review"]["lint_actions"],
        )

    def test_context_manifest_records_both_api_passes_and_final_answer(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {"qno": "16", "question_text": "16. Draft"},
                "alignment": {"read_index": 1},
                "visual_context": {
                    "views": [
                        {
                            "kind": "context",
                            "path": "context.png",
                            "sent_to_second_api": True,
                        }
                    ]
                },
            }
        ]
        final_questions = [
            {
                "qno": 16,
                "question_markdown": "16. Corrected",
                "question_review": {"source_context_ids": ["C001"]},
                "handwritten_answer": {"text": "x=2"},
            }
        ]

        manifest = _build_question_context_manifest(
            "page",
            Path("preprocessed/page.png"),
            Path("api_markdown/page.md"),
            contexts,
            final_questions,
        )

        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["api_strategy"]["call_1"], "whole-page draft OCR")
        self.assertTrue(manifest["contexts"][0]["consumed_by_second_api"])
        self.assertEqual(manifest["contexts"][0]["final_records"][0]["qno"], 16)

    def test_result_markdown_matches_gold_format_and_excludes_alternate_guesses(self):
        rendered = _render_result_markdown(
            "page",
            [
                {
                    "qno": 1,
                    "question_markdown": "1. Question",
                    "handwritten_answer": {
                        "text": "x=2",
                        "uncertain_fragments": [
                            {
                                "text": "y=?",
                                "alternatives": ["y=3", "y=5"],
                                "location": "lower right",
                            }
                        ],
                    },
                }
            ],
        )

        self.assertIn("x=2", rendered)
        self.assertTrue(rendered.startswith("1. Question\n\n### 手写答案\n\nx=2"))
        self.assertNotIn("## 题目", rendered)
        self.assertNotIn("不确定手写片段", rendered)
        self.assertNotIn("y=3", rendered)

    @patch.object(VLMClient, "invoke")
    def test_uncached_workflow_uses_exactly_two_api_calls(self, invoke_mock):
        first_pass = "1. Draft question\n\n### 手写答案\n\n$x=2$"
        second_pass = {
            "page_notes": "reviewed",
            "questions": [
                {
                    "qno": "1",
                    "question_markdown": "1. Corrected question",
                    "source_context_ids": ["C001"],
                    "handwritten_answer": {
                        "text": "x=2",
                        "answer_parts": [
                            {
                                "label": "overall",
                                "transcription": "x=2",
                                "final_answer": "x=2",
                                "status": "ok",
                            }
                        ],
                        "uncertain_fragments": [],
                        "status": "ok",
                        "evidence_note": "visible below the stem",
                    },
                }
            ],
        }
        invoke_mock.side_effect = [
            first_pass,
            json.dumps(second_pass, ensure_ascii=False),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "page.png"
            Image.new("RGB", (240, 200), "white").save(image_path)
            work_root = root / "workflow"
            page_dir = work_root / "page" / "code_outputs" / "match"
            page_dir.mkdir(parents=True)
            (page_dir / "match.json").write_text(
                json.dumps(
                    {
                        "image_stem": "page",
                        "width": 240,
                        "height": 200,
                        "questions": [
                            {
                                "det_index": 1,
                                "read_index": 1,
                                "class_name": "problem_solving_question",
                                "score": 0.95,
                                "bbox_xyxy_padded": [20, 20, 220, 80],
                                "crop_path": "questions/q0001_det001/question.png",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = build_argparser().parse_args(
                [
                    "--image",
                    str(image_path),
                    "--work-root",
                    str(work_root),
                    "--skip-layout",
                    "--no-cache",
                ]
            )

            final = run_agent(args)
            result = json.loads(
                Path(final["outputs"]["result_json"]).read_text(encoding="utf-8")
            )
            manifest = json.loads(
                Path(final["outputs"]["question_contexts"]).read_text(encoding="utf-8")
            )

        self.assertEqual(invoke_mock.call_count, 2)
        self.assertEqual(final["api_strategy"]["call_count"], 2)
        self.assertEqual(final["api_metrics"]["logical_call_count"], 2)
        self.assertEqual(
            Path(final["outputs"]["result_json"]),
            work_root / "page" / "agent_outputs" / "result.json",
        )
        self.assertFalse((work_root / "page" / "agent_outputs" / "page").exists())
        self.assertFalse((work_root / "page" / "code_outputs" / "match" / "page").exists())
        self.assertEqual(
            result["questions"][0]["question_markdown"],
            "1. Corrected question",
        )
        self.assertEqual(
            result["questions"][0]["handwritten_answer"]["text"],
            "$x=2$",
        )
        self.assertTrue(manifest["contexts"][0]["consumed_by_second_api"])
        first_messages = invoke_mock.call_args_list[0].args[0]
        self.assertEqual(first_messages[0], {"type": "text", "text": BASELINE_PROMPT})
        second_messages = invoke_mock.call_args_list[1].args[0]
        second_text = "\n".join(
            message.get("text", "")
            for message in second_messages
            if message.get("type") == "text"
        )
        self.assertIn("first-pass whole-page Markdown", second_text)
        self.assertIn("C001 / context", second_text)


if __name__ == "__main__":
    unittest.main()
