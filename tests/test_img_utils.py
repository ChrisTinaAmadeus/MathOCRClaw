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

    def test_removes_dark_and_faded_red_without_leaving_colored_edges(self):
        image = Image.new("RGB", (360, 200), (225, 225, 220))
        draw = ImageDraw.Draw(image)
        draw.line((20, 40, 340, 40), fill=(20, 20, 20), width=4)
        draw.line((20, 90, 340, 90), fill=(105, 55, 58), width=5)
        draw.line((20, 140, 340, 140), fill=(185, 125, 128), width=4)

        cleaned, report = scan_document_for_ocr(image)
        output = np.asarray(cleaned)

        self.assertGreater(report["red_core_pixels"], 0)
        self.assertGreater(report["red_edge_pixels_added"], 0)
        self.assertLessEqual(int(output[40, 180].max()), 80)
        for y in (90, 140):
            pixel = output[y, 180].astype(np.int16)
            self.assertGreaterEqual(int(pixel.mean()), 235)
            self.assertLessEqual(int(pixel.max() - pixel.min()), 8)


if __name__ == "__main__":
    unittest.main()
