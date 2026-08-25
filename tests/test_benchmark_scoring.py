import json
import tempfile
import unittest
from pathlib import Path

from benchmark.scoring import (
    candidate_from_result,
    candidate_from_first_pass,
    parse_markdown,
    score_files,
    score_page,
)
from benchmark.run_benchmark import (
    _workflow_args,
    build_argparser as build_benchmark_argparser,
)


GOLD = """1. 已知 $x=1$，求 $x+1$。

### 手写答案

(1) $x+1=2$

2. 下列说法正确的是
A. 甲
B. 乙
C. 丙
D. 丁

### 手写答案

B
"""


class BenchmarkScoringTests(unittest.TestCase):
    def test_benchmark_passes_extended_timeouts_to_both_api_calls(self):
        cli = build_benchmark_argparser().parse_args(
            ["--baseline-timeout", "480", "--review-timeout", "600"]
        )

        workflow_args = _workflow_args(cli, Path("page.png"), Path("workflow"))

        self.assertEqual(workflow_args.baseline_timeout, 480)
        self.assertEqual(workflow_args.review_timeout, 600)

    def test_result_json_question_number_is_not_scored_as_stem_text(self):
        candidate = candidate_from_result(
            {
                "questions": [
                    {
                        "qno": 19,
                        "question_markdown": "19. 已知 $x=1$。",
                        "handwritten_answer": {"text": "$x=1$", "status": "ok"},
                    }
                ]
            }
        )

        self.assertEqual(candidate[0]["stem"], "已知 $x=1$。")

    def test_gold_scores_one_hundred_against_itself(self):
        questions = parse_markdown(GOLD)

        result = score_page(questions, questions)

        self.assertEqual(len(questions), 2)
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["matched_question_count"], 2)
        self.assertGreater(result["format_diagnostics"]["gold_latex_formula_spans"], 0)
        self.assertEqual(result["format_diagnostics"]["formula_span_count_ratio"], 1.0)

    def test_first_pass_prefers_actual_page_markdown(self):
        candidate = candidate_from_first_pass(
            {
                "page_markdown": "1. Markdown stem\n\n### 手写答案\n\n$x=2$",
                "questions": [
                    {
                        "qno": "9",
                        "question_text": "different structured stem",
                        "student_answer": "different answer",
                    }
                ],
            }
        )

        self.assertEqual(candidate[0]["qno"], "1")
        self.assertIn("Markdown stem", candidate[0]["stem"])
        self.assertEqual(candidate[0]["answer"], "$x=2$")

    def test_section_heading_does_not_pollute_previous_answer_or_question_type(self):
        parsed = parse_markdown(
            "8. 选择正确答案\nA. 1\nB. 2\nC. 3\nD. 4\n\n"
            "### 手写答案\n\nA\n\n"
            "二、多项选择题：本题共 1 小题。\n\n"
            "9. 选择正确答案\nA. 甲\nB. 乙\nC. 丙\nD. 丁\n\n"
            "### 手写答案\n\nAB"
        )

        self.assertEqual(parsed[0]["answer"], "A")
        result = score_page(parsed, parsed)
        self.assertEqual(result["details"][0]["answer_type"], "choice")
        self.assertEqual(result["score"], 100.0)

    def test_result_uses_structured_final_answers_instead_of_scratch(self):
        candidate = candidate_from_result(
            {
                "questions": [
                    {
                        "qno": 3,
                        "question_type": "choice",
                        "question_markdown": (
                            "3. 选择正确答案\nA. 1\nB. 2\nC. 3\nD. 4"
                        ),
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
                        "qno": 12,
                        "question_type": "fill",
                        "question_markdown": "12. 结果为 ______",
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
        )

        self.assertEqual(candidate[0]["answer"], "A")
        self.assertEqual(candidate[1]["answer"], "$4$")

    def test_choice_type_is_determined_from_stem_even_when_answer_has_work(self):
        gold = parse_markdown(
            "1. 选择正确答案\nA. 甲\nB. 乙\nC. 丙\nD. 丁\n\n"
            "### 手写答案\n\nB"
        )
        candidate = [
            {
                **gold[0],
                "answer": "B\n$B(n)=\\lg\\frac{n+1}{n}$",
            }
        ]

        result = score_page(gold, candidate)

        self.assertEqual(result["details"][0]["answer_type"], "choice")
        self.assertEqual(result["details"][0]["answer_score"], 1.0)

    def test_result_keeps_all_structured_fill_final_answers(self):
        candidate = candidate_from_result(
            {
                "questions": [
                    {
                        "qno": 12,
                        "question_type": "fill",
                        "question_markdown": "12. 填空：(1) _____；(2) _____。",
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
        )

        self.assertEqual(candidate[0]["answer"], "(1) $2$\n(2) $3$")

    def test_omission_and_hallucination_reduce_score(self):
        gold = parse_markdown(GOLD)
        candidate = [
            {
                "qno": "1",
                "order_index": 0,
                "stem": gold[0]["stem"],
                "answer": "",
                "status": "no_answer",
            },
            {
                "qno": "99",
                "order_index": 1,
                "stem": "额外题目",
                "answer": "额外答案",
                "status": "ok",
            },
        ]

        result = score_page(gold, candidate)

        self.assertGreaterEqual(result["omission_count"], 1)
        self.assertGreaterEqual(result["hallucination_count"], 1)
        self.assertLess(result["score"], 100.0)

    def test_score_files_reports_api1_api2_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold_path = root / "gold.md"
            first_path = root / "first.json"
            final_path = root / "final.json"
            gold_path.write_text(GOLD, encoding="utf-8")
            first_path.write_text(
                json.dumps(
                    {
                        "page_markdown": (
                            "1. 错误题干\n\n### 手写答案\n\n_未识别到手写答案。_"
                        )
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            parsed = parse_markdown(GOLD)
            final_path.write_text(
                json.dumps(
                    {
                        "questions": [
                            {
                                "qno": question["qno"],
                                "question_markdown": question["stem"],
                                "handwritten_answer": {
                                    "text": question["answer"],
                                    "status": question["status"],
                                },
                            }
                            for question in parsed
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            scored = score_files(gold_path, first_path, final_path)

        self.assertEqual(scored["workflow"]["score"], 100.0)
        self.assertGreater(scored["gain"], 0.0)
        self.assertEqual(scored["evaluator"]["judge_model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
