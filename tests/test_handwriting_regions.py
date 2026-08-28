import tempfile
import unittest
from pathlib import Path

from PIL import Image

from agent.handwriting_regions import (
    classify_question_type,
    draw_handwriting_overlay,
    save_handwriting_views,
    score_and_divide_question_frame,
)


class HandwritingRegionTests(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {
                "class_name": "problem_solving_question",
                "score": 0.92,
                "bbox_xyxy_padded": [100, 100, 700, 260],
            },
            {
                "class_name": "problem_solving_question",
                "score": 0.90,
                "bbox_xyxy_padded": [110, 600, 690, 720],
            },
            {
                "class_name": "problem_solving_question",
                "score": 0.94,
                "bbox_xyxy_padded": [900, 120, 1500, 300],
            },
            {
                "class_name": "problem_solving_question",
                "score": 0.91,
                "bbox_xyxy_padded": [910, 650, 1510, 800],
            },
        ]

    def test_frame_extends_to_next_question_without_crossing_columns(self):
        region = score_and_divide_question_frame((1600, 1200), self.questions, 0)
        x0, y0, x1, y1 = region["frame_bbox_xyxy"]
        self.assertEqual(y0, 260)
        self.assertGreater(y1, 260)
        self.assertLess(y1, 600)
        self.assertLessEqual(x1, 800)
        self.assertEqual(region["boundary_kind"], "next_question")
        self.assertGreater(region["score"], 0.85)
        self.assertFalse(region["contains_stem"])

    def test_short_answer_frame_does_not_overlap_stem_box(self):
        region = score_and_divide_question_frame((1600, 1200), self.questions, 0)
        stem = region["source_bbox_xyxy"]
        answer = region["frame_bbox_xyxy"]
        self.assertGreaterEqual(answer[1], stem[3])
        self.assertEqual(region["answer_start_rule"], "after_stem_bottom")

    def test_collapsed_short_answer_gap_uses_non_degenerate_stem_tail(self):
        questions = [
            {
                "class_name": "problem_solving_question",
                "score": 0.93,
                "bbox_xyxy_padded": [606, 221, 2675, 731],
            },
            {
                "class_name": "problem_solving_question",
                "score": 0.91,
                "bbox_xyxy_padded": [620, 742, 2660, 980],
            },
        ]

        region = score_and_divide_question_frame((2747, 3000), questions, 0)
        stem = region["source_bbox_xyxy"]
        answer = region["frame_bbox_xyxy"]

        self.assertEqual(answer[3] - answer[1], 75)
        self.assertLess(answer[1], stem[3])
        self.assertTrue(region["contains_stem"])
        self.assertEqual(
            region["answer_start_rule"],
            "fallback_include_stem_tail_when_answer_gap_collapses",
        )
        self.assertIn("collapsed_gap_fallback", region["boundary_kind"])

    def test_last_question_extends_to_page_bottom(self):
        region = score_and_divide_question_frame((1600, 1200), self.questions, 3)
        self.assertEqual(region["frame_bbox_xyxy"][3], 1200)
        self.assertEqual(region["boundary_kind"], "page_bottom")

    def test_spanning_question_does_not_join_neighboring_columns(self):
        questions = self.questions + [
            {
                "class_name": "problem_solving_question",
                "score": 0.88,
                "is_spanning": True,
                "bbox_xyxy_padded": [100, 20, 1500, 80],
            }
        ]
        region = score_and_divide_question_frame((1600, 1200), questions, 0)
        self.assertLessEqual(region["frame_bbox_xyxy"][2], 800)

    def test_detector_class_drives_question_type(self):
        choice = classify_question_type({"class_name": "multiple_choice_question"})
        fill = classify_question_type({"class_name": "fill_blank_question"})
        self.assertEqual(choice["type"], "choice")
        self.assertEqual(fill["type"], "fill_blank")
        self.assertEqual(choice["source"], "detector_class")

    def test_question_text_is_used_when_detector_type_is_unknown(self):
        choice = classify_question_type(
            {"class_name": "partial_question"},
            "1. Choose one: A. 1 B. 2 C. 3 D. 4",
        )
        fill = classify_question_type(
            {"class_name": "partial_question"},
            r"2. 结果为 \underline{\qquad}",
        )
        escaped_fill = classify_question_type(
            {"class_name": "partial_question"},
            r"3. 结果为 \_\_\_\_\_\_。",
        )
        self.assertEqual(choice["type"], "choice")
        self.assertEqual(fill["type"], "fill_blank")
        self.assertEqual(escaped_fill["type"], "fill_blank")

    def test_strong_text_structure_can_override_problem_solving_label(self):
        result = classify_question_type(
            {"class_name": "problem_solving_question"},
            "请选择： A. 1 B. 2 C. 3 D. 4",
        )
        self.assertEqual(result["type"], "choice")
        self.assertEqual(result["source"], "text_override_detector")

    def test_algebraic_a_b_parentheses_inside_latex_are_not_choice_options(self):
        result = classify_question_type(
            {},
            r"15. 求：(1) $A\cap B$；(2) $(\complement_U A)\cup B$。",
        )

        self.assertNotEqual(result["type"], "choice")

    def test_choice_frame_stays_near_options_instead_of_using_all_working_space(self):
        questions = [dict(question) for question in self.questions]
        questions[0]["class_name"] = "multiple_choice_question"
        region = score_and_divide_question_frame((1600, 1200), questions, 0)
        self.assertEqual(region["question_type"]["type"], "choice")
        self.assertEqual(region["boundary_kind"], "choice_adaptive_margin")
        self.assertLess(region["frame_bbox_xyxy"][3], 400)

    def test_long_choice_text_receives_larger_safe_margin(self):
        questions = [dict(question) for question in self.questions]
        questions[0]["class_name"] = "multiple_choice_question"
        short_region = score_and_divide_question_frame(
            (1600, 1200),
            questions,
            0,
            question_text="1. A. 1 B. 2 C. 3 D. 4",
        )
        long_region = score_and_divide_question_frame(
            (1600, 1200),
            questions,
            0,
            question_text=("A long multi-line question statement with corrections. " * 8)
            + "\nA. 1\nB. 2\nC. 3\nD. 4",
        )
        short_box = short_region["frame_bbox_xyxy"]
        long_box = long_region["frame_bbox_xyxy"]
        self.assertLessEqual(long_box[0], short_box[0])
        self.assertGreaterEqual(long_box[2], short_box[2])
        self.assertLessEqual(long_box[1], short_box[1])
        self.assertGreaterEqual(long_box[3], short_box[3])
        self.assertGreater(
            long_region["adaptive_details"]["vertical_margin_px"],
            short_region["adaptive_details"]["vertical_margin_px"],
        )

    def test_saved_views_link_stem_answer_and_details(self):
        region = score_and_divide_question_frame((1600, 1200), self.questions, 0)
        with tempfile.TemporaryDirectory() as tmp:
            views = save_handwriting_views(
                Image.new("RGB", (1600, 1200), "white"),
                region,
                Path(tmp),
            )
            kinds = [view["kind"] for view in views]
            self.assertEqual(views[0]["kind"], "context")
            self.assertIn("stem", kinds)
            self.assertIn("answer", kinds)
            self.assertIn("answer_ink", kinds)
            self.assertTrue(any(kind.startswith("answer_detail_") for kind in kinds))
            context_box = views[0]["bbox_xyxy"]
            stem_box = region["source_bbox_xyxy"]
            answer_box = region["frame_bbox_xyxy"]
            self.assertLessEqual(context_box[0], min(stem_box[0], answer_box[0]))
            self.assertLessEqual(context_box[1], min(stem_box[1], answer_box[1]))
            self.assertGreaterEqual(context_box[2], max(stem_box[2], answer_box[2]))
            self.assertGreaterEqual(context_box[3], max(stem_box[3], answer_box[3]))
            self.assertTrue(all(Path(view["path"]).is_file() for view in views))
            ink_view = next(view for view in views if view["kind"] == "answer_ink")
            self.assertIn("faint strokes", ink_view["purpose"])

    def test_overlay_contains_stem_and_handwriting_frames(self):
        region = score_and_divide_question_frame((1600, 1200), self.questions, 0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "overlay.png"
            draw_handwriting_overlay(
                Image.new("RGB", (1600, 1200), "white"),
                [{"qno": 1, "region": region}],
                output,
            )
            rendered = Image.open(output).convert("RGB")
            self.assertTrue(output.is_file())
            self.assertEqual(rendered.getpixel((100, 259)), (0, 220, 90))


if __name__ == "__main__":
    unittest.main()
