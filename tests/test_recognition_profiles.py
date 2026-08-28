import unittest

from agent.recognition_profiles import (
    build_recognition_profile,
    profile_symbols,
    profile_tags,
)


class RecognitionProfileTests(unittest.TestCase):
    def test_choice_profile_uses_closed_letter_alphabet_only(self):
        profile = build_recognition_profile(
            "1. 下列结论正确的是\nA. 1\nB. 2\nC. 3\nD. 4",
            "choice",
            "B",
        )

        self.assertEqual(
            profile["answer_grammar"]["allowed_final_tokens"],
            ["A", "B", "C", "D"],
        )
        self.assertEqual(profile_tags(profile), {"CHOICE_LETTER"})
        self.assertEqual(profile["version"], "symbol_profile_v2")
        self.assertEqual(profile["answer_grammar"]["choice_mode"], "single")
        self.assertEqual(profile["answer_grammar"]["max_selected"], 1)
        self.assertEqual(profile_symbols(profile, "CHOICE_LETTER"), {"A", "B", "C", "D"})
        checks = profile["symbol_families"][0]["confusion_checks"]
        self.assertTrue(any("B vs D" in check for check in checks))

    def test_multiple_choice_profile_allows_more_than_one_final_letter(self):
        profile = build_recognition_profile(
            "9. 多选题\nA. 1\nB. 2\nC. 3\nD. 4",
            "choice",
            "AC",
            choice_mode="multiple",
        )

        self.assertEqual(profile["answer_grammar"]["choice_mode"], "multiple")
        self.assertEqual(profile["answer_grammar"]["max_selected"], 4)
        self.assertNotIn(
            "single_choice_answer_cardinality_violation",
            profile["recognition_risk"]["reasons"],
        )

    def test_fill_profile_marks_relevant_symbol_families_without_giving_answer(self):
        profile = build_recognition_profile(
            r"13. 解不等式 $x^2-3x\le 0$，用区间表示结果：\_\_\_。",
            "fill",
            r"$[0,3)$",
        )

        self.assertTrue(
            {"DIGIT", "SIGN_RELATION", "SET_INTERVAL", "SCRIPT_POSITION"}
            .issubset(profile_tags(profile))
        )
        serialized = str(profile)
        self.assertNotIn("correct_answer", serialized)
        self.assertIn("never evidence or an answer key", profile["usage_rule"])

    def test_physics_and_chemistry_profiles_are_extensible(self):
        physics = build_recognition_profile(
            "物体的加速度为多少 m/s^2？",
            "fill",
            "",
        )
        chemistry = build_recognition_profile(
            "写出该化学反应方程式并标注气体。",
            "fill",
            "",
        )

        self.assertIn("PHYSICS_UNIT", profile_tags(physics))
        self.assertIn("CHEM_FORMULA", profile_tags(chemistry))
        self.assertEqual(physics["recognition_risk"]["priority"], "high")

    def test_solid_geometry_profile_marks_parallel_and_equal_symbol(self):
        profile = build_recognition_profile(
            "在四棱柱中证明两条棱平行，并判断对应线段是否相等。",
            "solution",
            r"$AB \\mathrel{\\underline{\\parallel}} CD$",
        )

        geometry = next(
            family
            for family in profile["symbol_families"]
            if family["tag"] == "GEOMETRY_MARK"
        )
        self.assertIn(
            r"\mathrel{\underline{\parallel}}",
            geometry["symbols"],
        )
        self.assertTrue(
            any("parallel and equal" in check for check in geometry["confusion_checks"])
        )


if __name__ == "__main__":
    unittest.main()
