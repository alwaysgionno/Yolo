"""Test delle funzioni geometriche di FaceProcessor (ritagli e riquadri), senza caricare i modelli."""

from __future__ import annotations

import unittest

import numpy as np

from app.face_processor import _Crop, _crop_with_padding, _iou, _is_cut_by_crop

FRAME_WIDTH, FRAME_HEIGHT = 640, 480


class CropTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    def test_padding_is_added_around_the_region(self) -> None:
        crop = _crop_with_padding(self.frame, (200, 100, 300, 200))
        self.assertEqual((crop.offset_x, crop.offset_y), (175, 75))
        self.assertEqual(crop.image.shape[:2], (150, 150))

    def test_padding_stops_at_the_frame_border(self) -> None:
        crop = _crop_with_padding(self.frame, (0, 0, 100, 100))
        self.assertEqual((crop.offset_x, crop.offset_y), (0, 0))
        self.assertEqual(crop.image.shape[:2], (125, 125))

    def test_tiny_region_is_discarded(self) -> None:
        self.assertIsNone(_crop_with_padding(self.frame, (10, 10, 20, 200)))


class CutFaceTest(unittest.TestCase):
    def crop_at(self, x: int, y: int, width: int = 200, height: int = 200) -> _Crop:
        return _Crop(np.zeros((height, width, 3), dtype=np.uint8), x, y)

    def is_cut(self, box: tuple[float, float, float, float], crop: _Crop) -> bool:
        return _is_cut_by_crop(*box, crop, FRAME_WIDTH, FRAME_HEIGHT)

    def test_face_inside_the_crop_is_kept(self) -> None:
        self.assertFalse(self.is_cut((50, 50, 150, 150), self.crop_at(100, 100)))

    def test_face_cut_by_an_inner_edge_is_discarded(self) -> None:
        crop = self.crop_at(100, 100)
        for label, box in {
            "sinistra": (-5, 50, 60, 150),
            "alto": (50, 0, 150, 90),
            "destra": (140, 50, 201, 150),
            "basso": (50, 120, 150, 199.5),
        }.items():
            with self.subTest(label):
                self.assertTrue(self.is_cut(box, crop))

    def test_face_cut_by_the_frame_border_is_kept(self) -> None:
        top_left = self.crop_at(0, 0)
        self.assertFalse(self.is_cut((-5, -3, 80, 90), top_left))
        bottom_right = self.crop_at(FRAME_WIDTH - 200, FRAME_HEIGHT - 200)
        self.assertFalse(self.is_cut((120, 110, 203, 204), bottom_right))


class IouTest(unittest.TestCase):
    def test_overlap_values(self) -> None:
        self.assertEqual(_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertEqual(_iou((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)  # si toccano soltanto
        self.assertAlmostEqual(_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)


if __name__ == "__main__":
    unittest.main()
