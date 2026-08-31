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
    _same_page_symbol_groups,
    _selected_review_views,
    build_argparser,
    run_agent,
)
from agent.recognition_profiles import build_recognition_profile
from proofread.vlm_client import VLMClient


class WorkflowOutputTests(unittest.TestCase):
    def test_choice_detail_budget_prefers_handwriting_ink_companion(self):
        context = {
            "recognition_profile": {"question_type": "choice"},
            "visual_context": {
                "views": [
                    {"kind": "context", "path": "context.png"},
                    {"kind": "answer_detail_01", "path": "detail.png"},
                    {"kind": "answer_ink", "path": "ink.png"},
                ]
            },
        }

        selected = _selected_review_views(context, 1)

        self.assertEqual(
            [view["kind"] for view in selected],
            ["context", "answer_ink"],
        )

    def test_solution_symbol_audit_sends_color_detail_and_ink_companion(self):
        profile = build_recognition_profile(
            "18. 证明直线平行。",
            "solution",
            r"$AB // CD$",
        )
        context = {
            "recognition_profile": profile,
            "visual_context": {
                "views": [
                    {"kind": "context", "path": "context.png"},
                    {"kind": "answer_detail_01", "path": "detail.png"},
                    {"kind": "answer_ink", "path": "ink.png"},
                ]
            },
        }

        selected = _selected_review_views(context, 1)

        self.assertEqual(
            [view["kind"] for view in selected],
            ["context", "answer_detail_01", "answer_ink"],
        )

    def test_same_page_symbol_groups_are_unlabelled_writer_style_routes(self):
        contexts = [
            {
                "context_id": "C001",
                "recognition_profile": build_recognition_profile("A. 1\nB. 2", "choice", "B"),
                "visual_context": {"views": [{"kind": "context"}]},
            },
            {
                "context_id": "C002",
                "recognition_profile": build_recognition_profile("A. 1\nB. 2", "choice", "D"),
                "visual_context": {"views": [{"kind": "context"}]},
            },
        ]

        groups = _same_page_symbol_groups(contexts)

        self.assertEqual(groups[0]["tag"], "CHOICE_LETTER")
        self.assertEqual(groups[0]["context_ids"], ["C001", "C002"])
        self.assertNotIn("B", str(groups))
        self.assertNotIn("D", str(groups))

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
        self.assertEqual(questions[0]["choice_mode"], "single")
        self.assertEqual(questions[1]["choice_mode"], "multiple")

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

    def test_fill_no_answer_marker_is_not_wrapped_as_math(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "14",
                    "question_text": "14. 结果为 ______",
                    "student_answer": "_未识别到手写答案。_",
                    "answer_status": "no_answer",
                    "question_type": "fill",
                },
            }
        ]
        review = {
            "questions": [
                {
                    "qno": "14",
                    "question_type": "fill",
                    "question_markdown": contexts[0]["draft"]["question_text"],
                    "handwritten_answer": {"status": "no_answer"},
                }
            ]
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "")
        self.assertEqual(question["handwritten_answer"]["status"], "no_answer")

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

    def test_solution_guard_rejects_editorial_truncation(self):
        draft_answer = "(1) $x=1$\n(2) $y=2$\n(3) $z=3$"
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "19",
                    "question_text": "19. (1) 求值；(2) 求值；(3) 求值。",
                    "student_answer": draft_answer,
                    "answer_status": "ok",
                    "question_type": "solution",
                },
            }
        ]
        review = {
            "questions": [
                {
                    "qno": "19",
                    "question_type": "solution",
                    "question_markdown": contexts[0]["draft"]["question_text"],
                    "handwritten_answer": {
                        "text": "(1) $x=1$\n(2) ... (see full text)",
                        "status": "ok",
                    },
                }
            ]
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], draft_answer)
        self.assertIn(
            "rolled_back_editorial_solution_truncation",
            question["question_review"]["lint_actions"],
        )

    def test_api2_atomic_patch_applies_high_confidence_local_edit_only(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "17",
                    "question_text": "17. 解不等式 $x>2$。",
                    "student_answer": "$x>2$",
                    "answer_status": "ok",
                    "question_type": "solution",
                    "section_heading_before": "",
                },
            }
        ]
        review = {
            "schema_version": "api2_patch_v2",
            "question_reviews": [
                {
                    "qno": "17",
                    "source_context_ids": ["C001"],
                    "question_type": "solution",
                    "stem": {
                        "action": "edit",
                        "edits": [
                            {
                                "old": "x>2",
                                "new": "x\ge 2",
                                "kind": "ocr_correction",
                                "confidence": "high",
                                "context_id": "C001",
                                "evidence": "the relation has a visible lower bar",
                            }
                        ],
                    },
                    "answer": {
                        "action": "edit_solution",
                        "confidence": "high",
                        "evidence": "the answer relation has a visible lower bar",
                        "edits": [
                            {
                                "old": "x>2",
                                "new": "x\ge 2",
                                "kind": "ocr_correction",
                                "confidence": "high",
                                "context_id": "C001",
                                "evidence": "the relation has a visible lower bar",
                            }
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertIn(r"$x\ge 2$", question["question_markdown"])
        self.assertEqual(question["handwritten_answer"]["text"], r"$x\ge 2$")
        self.assertIn(
            "applied_solution_edit_1_ocr_correction",
            question["question_review"]["lint_actions"],
        )

    def test_api2_patch_cannot_delete_supported_answer_as_no_answer(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "15",
                    "question_text": "15. 求值。",
                    "student_answer": "$x=2$",
                    "answer_status": "ok",
                    "question_type": "solution",
                },
            }
        ]
        review = {
            "schema_version": "api2_patch_v2",
            "question_reviews": [
                {
                    "qno": "15",
                    "source_context_ids": ["C001"],
                    "question_type": "solution",
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "set_no_answer",
                        "confidence": "high",
                        "evidence": "crop appears blank",
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "$x=2$")
        self.assertIn(
            "rejected_no_answer_over_supported_first_pass",
            question["question_review"]["lint_actions"],
        )

    def test_api2_answer_replacement_requires_an_explicit_valid_context(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "12",
                    "question_text": "12. 填空：\\_\\_\\_。",
                    "student_answer": "$2$",
                    "answer_status": "ok",
                    "question_type": "fill",
                },
            }
        ]
        review = {
            "schema_version": "api2_patch_v2",
            "question_reviews": [
                {
                    "qno": "12",
                    "source_context_ids": ["C999"],
                    "question_type": "fill",
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "replace_fill",
                        "confidence": "high",
                        "evidence": "the crop appears to show 3",
                        "final_answers": [
                            {"label": "overall", "value": "$3$", "status": "ok"}
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "$2$")
        self.assertIn(
            "rejected_answer_replacement_without_valid_context",
            question["question_review"]["lint_actions"],
        )

    def test_api2_v3_choice_replacement_requires_contrastive_symbol_observation(self):
        question_text = "1. 选择正确答案。\nA. 1\nB. 2\nC. 3\nD. 4"
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "1",
                    "question_text": question_text,
                    "student_answer": "B",
                    "answer_status": "ok",
                    "question_type": "choice",
                },
                "recognition_profile": build_recognition_profile(
                    question_text,
                    "choice",
                    "B",
                ),
                "visual_context": {"views": [{"kind": "context"}]},
            },
            {
                "context_id": "C002",
                "draft": {
                    "qno": "2",
                    "question_text": question_text,
                    "student_answer": "D",
                    "answer_status": "ok",
                    "question_type": "choice",
                },
                "recognition_profile": build_recognition_profile(
                    question_text,
                    "choice",
                    "D",
                ),
                "visual_context": {"views": [{"kind": "context"}]},
            },
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "1",
                    "source_context_ids": ["C001"],
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "replace_choice",
                        "confidence": "high",
                        "evidence": "the final mark has a single outer right bow",
                        "final_answers": [
                            {"label": "overall", "value": "D", "status": "ok"}
                        ],
                        "symbol_observations": [],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "B")
        self.assertIn(
            "rejected_answer_replacement_without_symbol_observation",
            question["question_review"]["lint_actions"],
        )

    def test_api2_v3_accepts_profile_tagged_choice_stroke_observation(self):
        question_text = "1. 选择正确答案。\nA. 1\nB. 2\nC. 3\nD. 4"
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "1",
                    "question_text": question_text,
                    "student_answer": "B",
                    "answer_status": "ok",
                    "question_type": "choice",
                },
                "recognition_profile": build_recognition_profile(
                    question_text,
                    "choice",
                    "B",
                ),
                "visual_context": {"views": [{"kind": "context"}]},
            },
            {
                "context_id": "C002",
                "draft": {
                    "qno": "2",
                    "question_text": question_text,
                    "student_answer": "D",
                    "answer_status": "ok",
                    "question_type": "choice",
                },
                "recognition_profile": build_recognition_profile(
                    question_text,
                    "choice",
                    "D",
                ),
                "visual_context": {"views": [{"kind": "context"}]},
            },
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "1",
                    "source_context_ids": ["C001"],
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "replace_choice",
                        "confidence": "high",
                        "evidence": "the final mark has a straight left stem and one outer bow",
                        "final_answers": [
                            {"label": "overall", "value": "D", "status": "ok"}
                        ],
                        "symbol_observations": [
                            {
                                "tag": "CHOICE_LETTER",
                                "location": "final mark inside the answer parentheses",
                                "candidates": ["B", "D"],
                                "selected": "D",
                                "observed_features": (
                                    "one continuous outer right bow rather than two separate lobes"
                                ),
                                "context_id": "C001",
                                "reference_context_ids": ["C002"],
                            }
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "D")
        self.assertEqual(
            question["handwritten_answer"]["symbol_observations"][0]["tag"],
            "CHOICE_LETTER",
        )
        self.assertIn(
            "accepted_symbol_observation_1",
            question["question_review"]["lint_actions"],
        )

    def test_api2_patch_keep_preserves_plain_fill_answer_byte_exact(self):
        question_text = "14. 实数 $m=$ ______"
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "14",
                    "question_text": question_text,
                    "student_answer": "1",
                    "answer_status": "ok",
                    "question_type": "fill",
                },
                "recognition_profile": build_recognition_profile(
                    question_text,
                    "fill",
                    "1",
                ),
            }
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "14",
                    "source_context_ids": ["C001"],
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "keep",
                        "confidence": "high",
                        "final_answers": [],
                        "edits": [],
                        "symbol_observations": [],
                        "evidence": "",
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "1")
        self.assertEqual(question["handwritten_answer"]["answer_parts"], [])
        self.assertIn(
            "kept_first_pass_answer_byte_exact",
            question["question_review"]["lint_actions"],
        )

    def test_single_choice_rejects_multi_letter_api2_answer(self):
        question_text = "7. 单选题\nA. 1\nB. 2\nC. 3\nD. 4"
        profile = build_recognition_profile(
            question_text,
            "choice",
            "A",
            choice_mode="single",
        )
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "7",
                    "question_text": question_text,
                    "student_answer": "A",
                    "answer_status": "ok",
                    "question_type": "choice",
                    "choice_mode": "single",
                },
                "recognition_profile": profile,
                "visual_context": {"views": [{"kind": "context"}]},
            },
            {
                "context_id": "C002",
                "draft": {
                    "qno": "10",
                    "question_text": question_text,
                    "student_answer": "AC",
                    "answer_status": "ok",
                    "question_type": "choice",
                    "choice_mode": "multiple",
                },
                "recognition_profile": profile,
                "visual_context": {"views": [{"kind": "context"}]},
            },
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "7",
                    "source_context_ids": ["C001"],
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "replace_choice",
                        "confidence": "high",
                        "evidence": "two marks are claimed",
                        "final_answers": [
                            {"label": "overall", "value": "AC", "status": "ok"}
                        ],
                        "symbol_observations": [
                            {
                                "tag": "CHOICE_LETTER",
                                "location": "first retained glyph",
                                "candidates": ["A", "B"],
                                "selected": "A",
                                "observed_features": "apex and crossbar",
                                "context_id": "C001",
                                "reference_context_ids": ["C002"],
                            },
                            {
                                "tag": "CHOICE_LETTER",
                                "location": "second claimed glyph",
                                "candidates": ["C", "D"],
                                "selected": "C",
                                "observed_features": "open curve",
                                "context_id": "C001",
                            },
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "A")
        self.assertEqual(question["choice_mode"], "single")
        self.assertIn(
            "rejected_choice_answer_cardinality_or_grammar",
            question["question_review"]["lint_actions"],
        )

    def test_multiple_choice_requires_one_valid_observation_per_final_letter(self):
        question_text = "9. 多选题\nA. 1\nB. 2\nC. 3\nD. 4"
        profile = build_recognition_profile(
            question_text,
            "choice",
            "AD",
            choice_mode="multiple",
        )
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "9",
                    "question_text": question_text,
                    "student_answer": "AD",
                    "answer_status": "ok",
                    "question_type": "choice",
                    "choice_mode": "multiple",
                },
                "recognition_profile": profile,
                "visual_context": {"views": [{"kind": "context"}]},
            },
            {
                "context_id": "C002",
                "draft": {
                    "qno": "10",
                    "question_text": question_text,
                    "student_answer": "AC",
                    "answer_status": "ok",
                    "question_type": "choice",
                    "choice_mode": "multiple",
                },
                "recognition_profile": profile,
                "visual_context": {"views": [{"kind": "context"}]},
            },
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "9",
                    "source_context_ids": ["C001"],
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "replace_choice",
                        "confidence": "high",
                        "evidence": "two separately localized retained glyphs",
                        "final_answers": [
                            {"label": "overall", "value": "AC", "status": "ok"}
                        ],
                        "symbol_observations": [
                            {
                                "tag": "CHOICE_LETTER",
                                "location": "first glyph",
                                "candidates": ["A", "B"],
                                "selected": "A",
                                "observed_features": "apex and crossbar",
                                "context_id": "C001",
                                "reference_context_ids": ["C002"],
                            },
                            {
                                "tag": "CHOICE_LETTER",
                                "location": "second glyph",
                                "candidates": ["C", "D"],
                                "selected": "C",
                                "observed_features": "open curve without a left stem",
                                "context_id": "C001",
                                "reference_context_ids": ["C002"],
                            },
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "AC")
        self.assertEqual(question["choice_mode"], "multiple")
        self.assertIn(
            "accepted_replace_choice",
            question["question_review"]["lint_actions"],
        )

        review["question_reviews"][0]["answer"]["symbol_observations"] = review[
            "question_reviews"
        ][0]["answer"]["symbol_observations"][:1]
        rejected = _normalize_final_questions(review, contexts)[0]
        self.assertEqual(rejected["handwritten_answer"]["text"], "AD")
        self.assertIn(
            "rejected_choice_answer_without_per_glyph_observations",
            rejected["question_review"]["lint_actions"],
        )

    def test_choice_observation_rejects_multi_letter_pseudo_glyph(self):
        question_text = "7. 单选题\nA. 1\nB. 2\nC. 3\nD. 4"
        profile = build_recognition_profile(question_text, "choice", "A")
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "7",
                    "question_text": question_text,
                    "student_answer": "A",
                    "answer_status": "ok",
                    "question_type": "choice",
                    "choice_mode": "single",
                },
                "recognition_profile": profile,
            }
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "7",
                    "source_context_ids": ["C001"],
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "replace_choice",
                        "confidence": "high",
                        "evidence": "claimed AC token",
                        "final_answers": [
                            {"label": "overall", "value": "AC", "status": "ok"}
                        ],
                        "symbol_observations": [
                            {
                                "tag": "CHOICE_LETTER",
                                "location": "left margin",
                                "candidates": ["A", "AC", "C"],
                                "selected": "AC",
                                "observed_features": "claimed two-letter token",
                                "context_id": "C001",
                            }
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "A")
        self.assertIn(
            "rejected_symbol_observation_1_outside_profile_symbols",
            question["question_review"]["lint_actions"],
        )

    def test_symbol_guided_solution_edit_requires_independent_reference(self):
        question_text = "18. 求函数值。"
        profile = build_recognition_profile(question_text, "solution", "$f(1)<f(2)$")
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "18",
                    "question_text": question_text,
                    "student_answer": "$f(1)<f(2)$",
                    "answer_status": "ok",
                    "question_type": "solution",
                },
                "recognition_profile": profile,
            }
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "18",
                    "source_context_ids": ["C001"],
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "edit_solution",
                        "confidence": "high",
                        "evidence": "the final digit is claimed to be 1",
                        "edits": [
                            {
                                "old": "f(2)",
                                "new": "f(1)",
                                "kind": "ocr_correction",
                                "confidence": "high",
                                "context_id": "C001",
                                "evidence": "single vertical stroke",
                            }
                        ],
                        "symbol_observations": [
                            {
                                "tag": "DIGIT",
                                "location": "right side of the inequality",
                                "candidates": ["1", "2"],
                                "selected": "1",
                                "observed_features": "single vertical stroke",
                                "context_id": "C001",
                                "reference_context_ids": [],
                            }
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "$f(1)<f(2)$")
        self.assertIn(
            "rejected_symbol_guided_solution_edit_without_independent_reference",
            question["question_review"]["lint_actions"],
        )

    def test_exact_geometry_audit_patch_does_not_need_cross_question_reference(self):
        question_text = "18. 证明空间直线平行。"
        draft_answer = r"$A_1C_1 // AC$"
        profile = build_recognition_profile(
            question_text,
            "solution",
            draft_answer,
        )
        target = profile["symbol_audit"]["targets"][0]
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "18",
                    "question_text": question_text,
                    "student_answer": draft_answer,
                    "answer_status": "ok",
                    "question_type": "solution",
                },
                "recognition_profile": profile,
                "visual_context": {"views": [{"kind": "answer_detail_01"}]},
            }
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "18",
                    "source_context_ids": ["C001"],
                    "answer": {
                        "action": "edit_solution",
                        "confidence": "high",
                        "evidence": "two retained parallel strokes without an underline",
                        "edits": [
                            {
                                "old": target["draft_fragment"],
                                "new": r"$A_1C_1 \parallel AC$",
                                "kind": "ocr_correction",
                                "confidence": "high",
                                "context_id": "C001",
                                "evidence": "two retained parallel strokes without an underline",
                            }
                        ],
                        "symbol_observations": [
                            {
                                "audit_id": target["audit_id"],
                                "tag": "GEOMETRY_MARK",
                                "location": "between A_1C_1 and AC",
                                "candidates": [
                                    r"\parallel",
                                    r"\mathrel{\underline{\parallel}}",
                                    "=",
                                ],
                                "selected": r"\parallel",
                                "observed_features": (
                                    "two parallel oblique strokes and no separate lower bar"
                                ),
                                "context_id": "C001",
                                "reference_context_ids": [],
                            }
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], r"$A_1C_1 \parallel AC$")
        self.assertIn(
            "applied_solution_edit_1_ocr_correction",
            question["question_review"]["lint_actions"],
        )
        self.assertEqual(question["question_review"]["symbol_audit_target_count"], 1)
        self.assertEqual(question["question_review"]["symbol_audit_observed_count"], 1)

    def test_geometry_audit_ignores_model_edit_and_materializes_selected_candidate(self):
        question_text = "18. 证明空间直线平行。"
        draft_answer = r"$A_1C_1 // AC$"
        profile = build_recognition_profile(question_text, "solution", draft_answer)
        target = profile["symbol_audit"]["targets"][0]
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "18",
                    "question_text": question_text,
                    "student_answer": draft_answer,
                    "answer_status": "ok",
                    "question_type": "solution",
                },
                "recognition_profile": profile,
                "visual_context": {"views": [{"kind": "answer_detail_01"}]},
            }
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "18",
                    "source_context_ids": ["C001"],
                    "answer": {
                        "action": "edit_solution",
                        "confidence": "high",
                        "evidence": "claimed visible relation",
                        "edits": [
                            {
                                "old": target["draft_fragment"],
                                "new": r"$A_1C_1 = AC$",
                                "kind": "ocr_correction",
                                "confidence": "high",
                                "context_id": "C001",
                                "evidence": "claimed visible relation",
                            }
                        ],
                        "symbol_observations": [
                            {
                                "audit_id": target["audit_id"],
                                "tag": "GEOMETRY_MARK",
                                "location": "between the two segments",
                                "candidates": [r"\parallel", "="],
                                "selected": r"\parallel",
                                "observed_features": "two parallel oblique strokes",
                                "context_id": "C001",
                            }
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(
            question["handwritten_answer"]["text"],
            r"$A_1C_1 \parallel AC$",
        )
        self.assertIn(
            "discarded_1_model_authored_symbol_edits",
            question["question_review"]["lint_actions"],
        )
        self.assertIn(
            "materialized_1_symbol_audit_edits",
            question["question_review"]["lint_actions"],
        )

    def test_vector_angle_audit_normalizes_expanded_model_selection(self):
        question_text = "18. 求两平面所成角。"
        draft_answer = r"$\cos<\vec{n},\overrightarrow{AB}>$"
        profile = build_recognition_profile(question_text, "solution", draft_answer)
        target = profile["symbol_audit"]["targets"][0]
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "18",
                    "question_text": question_text,
                    "student_answer": draft_answer,
                    "answer_status": "ok",
                    "question_type": "solution",
                },
                "recognition_profile": profile,
                "visual_context": {"views": [{"kind": "answer_detail_01"}]},
            }
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "18",
                    "source_context_ids": ["C001"],
                    "answer": {
                        "action": "edit_solution",
                        "confidence": "high",
                        "evidence": "paired enclosing angle strokes are visible",
                        "edits": [
                            {
                                "old": r"<\vec{n},\overrightarrow{AB}>",
                                "new": r"\langle\vec{n},\overrightarrow{AB}\rangle",
                                "kind": "ocr_correction",
                                "confidence": "high",
                                "context_id": "C001",
                                "evidence": "paired enclosing angle strokes are visible",
                            }
                        ],
                        "symbol_observations": [
                            {
                                "audit_id": target["audit_id"],
                                "tag": "GEOMETRY_MARK",
                                "location": "inside the cosine expression",
                                "candidates": [
                                    r"\langle\cdot,\cdot\rangle",
                                    r"<\cdot,\cdot>",
                                ],
                                "selected": r"\langle\vec{n},\overrightarrow{AB}\rangle",
                                "observed_features": "opening and closing angle strokes enclose both vectors",
                                "context_id": "C001",
                            }
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(
            question["handwritten_answer"]["text"],
            r"$\cos\langle\vec{n},\overrightarrow{AB}\rangle$",
        )
        self.assertEqual(
            question["handwritten_answer"]["symbol_observations"][0]["selected"],
            r"\langle\cdot,\cdot\rangle",
        )

    def test_solution_edit_cannot_insert_a_newline_inside_inline_math(self):
        question_text = "19. 求体积。"
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "19",
                    "question_text": question_text,
                    "student_answer": "$h=2$\n$V=8/3$",
                    "answer_status": "ok",
                    "question_type": "solution",
                    "section_heading_before": "三、解答题",
                },
                "recognition_profile": build_recognition_profile(
                    question_text, "solution", "$h=2$\n$V=8/3$"
                ),
                "visual_context": {"views": [{"kind": "answer_detail"}]},
            }
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "19",
                    "source_context_ids": ["C001"],
                    "answer": {
                        "action": "edit_solution",
                        "confidence": "high",
                        "evidence": "The missing area line is visible.",
                        "symbol_observations": [],
                        "edits": [
                            {
                                "old": "h=2",
                                "new": "S=4\nh=2",
                                "kind": "insert_missing",
                                "confidence": "high",
                                "context_id": "C001",
                                "evidence": "The missing area line is visible.",
                            }
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], "$h=2$\n$V=8/3$")
        self.assertIn(
            "rejected_solution_edit_1_newline_inside_inline_latex",
            question["question_review"]["lint_actions"],
        )

    def test_api2_v3_rejects_fill_observation_unrelated_to_proposed_answer(self):
        question_text = r"13. 解不等式并填空：\_\_\_。"
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "13",
                    "question_text": question_text,
                    "student_answer": r"$x<3$",
                    "answer_status": "ok",
                    "question_type": "fill",
                },
                "recognition_profile": build_recognition_profile(
                    question_text,
                    "fill",
                    r"$x<3$",
                ),
            }
        ]
        review = {
            "schema_version": "api2_patch_v3",
            "question_reviews": [
                {
                    "qno": "13",
                    "source_context_ids": ["C001"],
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "replace_fill",
                        "confidence": "high",
                        "evidence": "a lower relation bar is visible",
                        "final_answers": [
                            {"label": "overall", "value": r"$x\ge7$", "status": "ok"}
                        ],
                        "symbol_observations": [
                            {
                                "tag": "DIGIT",
                                "location": "right endpoint",
                                "candidates": ["3", "5"],
                                "selected": "3",
                                "observed_features": "two open right-facing curves",
                                "context_id": "C001",
                            }
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["handwritten_answer"]["text"], r"$x<3$")
        self.assertIn(
            "rejected_symbol_observation_answer_mismatch",
            question["question_review"]["lint_actions"],
        )

    def test_p0_freeze_rolls_back_stem_figure_removal(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "18",
                    "question_text": "18. (17分)\n<插图>",
                    "student_answer": "(1) 证明过程",
                    "answer_status": "ok",
                    "question_type": "solution",
                },
            }
        ]
        review = {
            "schema_version": "api2_patch_v2",
            "question_reviews": [
                {
                    "qno": "18",
                    "source_context_ids": ["C001"],
                    "question_type": "solution",
                    "stem": {
                        "action": "edit",
                        "edits": [
                            {
                                "old": "<插图>",
                                "new": "",
                                "kind": "ocr_correction",
                                "confidence": "high",
                                "context_id": "C001",
                                "evidence": "hand-drawn axes match the student's pen strokes",
                            }
                        ],
                    },
                    "answer": {"action": "keep", "edits": []},
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertIn("<插图>", question["question_markdown"])
        self.assertIn(
            "applied_stem_edit_1_ocr_correction",
            question["question_review"]["lint_actions"],
        )
        self.assertIn(
            "rolled_back_deleted_question_structure",
            question["question_review"]["lint_actions"],
        )
        self.assertIn(
            "rejected_stem_image_removal_p0",
            question["question_review"]["lint_actions"],
        )

    def test_solution_section_prevents_api2_choice_misclassification_from_algebra(self):
        contexts = [
            {
                "context_id": "C001",
                "draft": {
                    "qno": "15",
                    "question_text": (
                        r"15. 求：(1) $A\cap B$；"
                        "\n"
                        r"(2) $(\complement_U A)\cup B$。"
                    ),
                    "student_answer": r"(1) $A\cap B=\varnothing$",
                    "answer_status": "ok",
                    "question_type": "solution",
                    "section_heading_before": "四、解答题",
                },
            }
        ]
        review = {
            "schema_version": "api2_patch_v2",
            "question_reviews": [
                {
                    "qno": "15",
                    "source_context_ids": ["C001"],
                    "question_type": "choice",
                    "stem": {"action": "keep", "edits": []},
                    "answer": {
                        "action": "replace_choice",
                        "confidence": "high",
                        "evidence": "misleading algebraic A and B",
                        "final_answers": [
                            {"label": "overall", "value": "A", "status": "ok"}
                        ],
                    },
                }
            ],
        }

        question = _normalize_final_questions(review, contexts)[0]

        self.assertEqual(question["question_type"], "solution")
        self.assertEqual(
            question["handwritten_answer"]["text"],
            r"(1) $A\cap B=\varnothing$",
        )
        self.assertIn(
            "rejected_answer_action_type_mismatch",
            question["question_review"]["lint_actions"],
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

        self.assertEqual(manifest["schema_version"], 4)
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
        self.assertIn("authoritative API1 draft", second_text)
        self.assertIn("api2_patch_v3", second_text)
        self.assertIn("recognition_profile", second_text)
        self.assertIn("C001 / context", second_text)

    @patch.object(VLMClient, "invoke")
    def test_repeated_api2_reviews_reuse_one_api1_draft(self, invoke_mock):
        first_pass = "1. Draft question\n\n### 手写答案\n\n$x=2$"
        second_pass = json.dumps(
            {
                "page_notes": "reviewed",
                "questions": [
                    {
                        "qno": "1",
                        "question_markdown": "1. Draft question",
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
            },
            ensure_ascii=False,
        )
        invoke_mock.side_effect = [first_pass, second_pass, second_pass, second_pass]

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
                    "--review-runs",
                    "3",
                ]
            )

            final = run_agent(args)
            api2_results = [Path(path) for path in final["outputs"]["api2_results"]]
            api2_payloads = [
                json.loads(path.read_text(encoding="utf-8")) for path in api2_results
            ]

        self.assertEqual(invoke_mock.call_count, 4)
        self.assertEqual(final["api_strategy"]["call_count"], 4)
        self.assertEqual(final["api_metrics"]["logical_call_count"], 4)
        self.assertEqual(final["api_metrics"]["api2_run_count"], 3)
        self.assertEqual([payload["api2_run"] for payload in api2_payloads], [1, 2, 3])
        self.assertTrue(all(payload["questions"] for payload in api2_payloads))


if __name__ == "__main__":
    unittest.main()
