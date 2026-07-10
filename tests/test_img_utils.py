import unittest

import numpy as np
from PIL import Image, ImageDraw

from proofread.img_utils import scan_document_for_ocr


class DocumentPreprocessTests(unittest.TestCase):
    def test_removes_shadow_and_red_ink_without_erasing_black_strokes(self):
        pixels = np.full((180, 320, 3), 238, dtype=np.uint8)
        pixels[:, :160] = 180
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.line((20, 40, 300, 40), fill=(15, 15, 15), width=3)
        draw.line((20, 90, 300, 90), fill=(220, 25, 25), width=5)

        cleaned, report = scan_document_for_ocr(image)
        output = np.asarray(cleaned)

        self.assertGreater(report["red_pixels_removed"], 0)
        self.assertGreaterEqual(int(output[90, 160].min()), 250)
        self.assertLessEqual(int(output[40, 160].max()), 80)
        self.assertLessEqual(
            abs(float(output[140, 40].mean()) - float(output[140, 280].mean())),
            5.0,
        )


if __name__ == "__main__":
    unittest.main()
