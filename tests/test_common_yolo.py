import unittest

import numpy as np
import torch

from common_yolo.geometry import scale_coords, xywh2xyxy, xyxy2xywh
from common_yolo.image import letterbox
from face.utils.datasets import letterbox as face_letterbox
from face.utils.general import scale_coords as face_scale_coords
from face.utils.general import xywh2xyxy as face_xywh2xyxy
from face.utils.general import xyxy2xywh as face_xyxy2xywh
from yolo.utils.datasets import letterbox as yolo_letterbox
from yolo.utils.utils import scale_coords as yolo_scale_coords
from yolo.utils.utils import xywh2xyxy as yolo_xywh2xyxy
from yolo.utils.utils import xyxy2xywh as yolo_xyxy2xywh


class CommonYoloTest(unittest.TestCase):
    def test_letterbox_matches_existing_implementations(self):
        image = np.arange(37 * 53 * 3, dtype=np.uint8).reshape(37, 53, 3)

        actual = letterbox(image, new_shape=(96, 128), auto=False)
        yolo_expected = yolo_letterbox(image, new_shape=(96, 128), auto=False)
        face_expected = face_letterbox(image, new_shape=(96, 128), auto=False)

        self.assertTrue(np.array_equal(actual[0], yolo_expected[0]))
        self.assertEqual(actual[1:], yolo_expected[1:])
        self.assertTrue(np.array_equal(actual[0], face_expected[0]))
        self.assertEqual(actual[1:], face_expected[1:])

    def test_box_conversion_matches_existing_implementations(self):
        xyxy = torch.tensor([[10.0, 20.0, 30.0, 60.0], [5.0, 8.0, 15.0, 18.0]])
        xywh = torch.tensor([[20.0, 40.0, 20.0, 40.0], [10.0, 13.0, 10.0, 10.0]])

        self.assertTrue(torch.equal(xyxy2xywh(xyxy), yolo_xyxy2xywh(xyxy)))
        self.assertTrue(torch.equal(xyxy2xywh(xyxy), face_xyxy2xywh(xyxy)))
        self.assertTrue(torch.equal(xywh2xyxy(xywh), yolo_xywh2xyxy(xywh)))
        self.assertTrue(torch.equal(xywh2xyxy(xywh), face_xywh2xyxy(xywh)))

    def test_scale_coords_matches_existing_implementations(self):
        coords = torch.tensor([[8.0, 10.0, 40.0, 50.0], [0.0, 5.0, 60.0, 64.0]])

        actual = scale_coords((64, 64), coords.clone(), (40, 50, 3))
        yolo_expected = yolo_scale_coords((64, 64), coords.clone(), (40, 50, 3))
        face_expected = face_scale_coords((64, 64), coords.clone(), (40, 50, 3))

        self.assertTrue(torch.equal(actual, yolo_expected))
        self.assertTrue(torch.equal(actual, face_expected))


if __name__ == "__main__":
    unittest.main()
