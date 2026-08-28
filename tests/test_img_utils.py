import unittest

import numpy as np
from PIL import Image, ImageDraw

from proofread.img_utils import (
    STRONG_ENHANCE,
    enhance_for_vlm,
    enhance_handwriting_ink,
    scan_document_for_ocr,
)


class DocumentPreprocessTests(unittest.TestCase):
    def test_vlm_enhancement_preserves_nonzero_thin_crop_dimension(self):
        enhanced = enhance_for_vlm(
            Image.new("RGB", (2069, 1), "white"),
            STRONG_ENHANCE,
        )

        self.assertEqual(enhanced.size, (1400, 1))

    def test_handwriting_companion_upscales_and_increases_faint_stroke_contrast(self):
        image = Image.new("RGB", (180, 80), (238, 238, 238))
        draw = ImageDraw.Draw(image)
        draw.line((20, 40, 160, 40), fill=(170, 170, 170), width=2)

        enhanced = enhance_handwriting_ink(image, max_edge=360)
        output = np.asarray(enhanced.convert("L"))

        self.assertEqual(enhanced.size, (360, 160))
        background = float(output[20:40, 20:300].mean())
        stroke = float(output[78:83, 40:320].mean())
        self.assertGreater(background - stroke, 20.0)

    def test_removes_shadow_without_erasing_black_or_red_strokes(self):
        pixels = np.full((180, 320, 3), 238, dtype=np.uint8)
        pixels[:, :160] = 180
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.line((20, 40, 300, 40), fill=(15, 15, 15), width=3)
        draw.line((20, 90, 300, 90), fill=(220, 25, 25), width=5)

        cleaned, report = scan_document_for_ocr(image)
        output = np.asarray(cleaned)

        self.assertEqual(report["method"], "background_division")
        self.assertTrue(report["colored_ink_preserved"])
        red_pixel = output[90, 160].astype(np.int16)
        self.assertGreater(int(red_pixel[0] - max(red_pixel[1], red_pixel[2])), 80)
        self.assertLessEqual(int(output[40, 160].max()), 80)
        self.assertLessEqual(
            abs(float(output[140, 40].mean()) - float(output[140, 280].mean())),
            5.0,
        )

    def test_preserves_dark_and_faded_red_ink(self):
        image = Image.new("RGB", (360, 200), (225, 225, 220))
        draw = ImageDraw.Draw(image)
        draw.line((20, 40, 340, 40), fill=(20, 20, 20), width=4)
        draw.line((20, 90, 340, 90), fill=(105, 55, 58), width=5)
        draw.line((20, 140, 340, 140), fill=(185, 125, 128), width=4)

        cleaned, report = scan_document_for_ocr(image)
        output = np.asarray(cleaned)

        self.assertTrue(report["colored_ink_preserved"])
        self.assertLessEqual(int(output[40, 180].max()), 80)
        for y in (90, 140):
            pixel = output[y, 180].astype(np.int16)
            self.assertGreater(int(pixel[0] - max(pixel[1], pixel[2])), 30)
            self.assertLess(int(pixel.mean()), 220)


if __name__ == "__main__":
    unittest.main()
