import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from face.recognition_dataset import BinaryFaceDataset
from convert_face_dataset import convert_data


class FaceDatasetConversionTest(unittest.TestCase):
    def test_convert_folder_images_to_binary_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "raw_faces"
            output = Path(tmp_dir) / "faces"
            self._write_image(root / "alice" / "1.jpg", 30)
            self._write_image(root / "alice" / "2.jpg", 60)
            self._write_image(root / "bob" / "1.jpg", 90)
            self._write_image(root / "bob" / "2.jpg", 120)

            result = convert_data(root, output, val_ratio=0.5, val_max_samples=10, seed=7)

            self.assertEqual(result.train_samples + result.val_samples, 4)
            self.assertEqual(result.train_classes + result.val_classes, 2)
            for stem in ["train_data", "val_data"]:
                for suffix in [".data", ".header", ".label"]:
                    self.assertTrue((output / f"{stem}{suffix}").exists())

            train_dataset = BinaryFaceDataset(str(output / "train_data"), is_train=False)
            val_dataset = BinaryFaceDataset(str(output / "val_data"), is_train=False)

            self.assertEqual(len(train_dataset) + len(val_dataset), 4)
            self.assertGreaterEqual(train_dataset.num_classes + val_dataset.num_classes, 2)
            image, label = train_dataset[0] if len(train_dataset) else val_dataset[0]
            self.assertEqual(image.shape, (3, 112, 112))
            self.assertIsInstance(label, int)

    @staticmethod
    def _write_image(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((16, 16, 3), value, dtype=np.uint8)
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise RuntimeError(f"写入测试图片失败: {path}")


if __name__ == "__main__":
    unittest.main()
