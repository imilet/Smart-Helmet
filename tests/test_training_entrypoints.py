import unittest


class TrainingEntrypointsTest(unittest.TestCase):
    def test_helmet_training_config_loads(self):
        from train_helmet import load_data_config

        data = load_data_config("configs/helmet_data.yaml")

        self.assertEqual(data["nc"], 3)
        self.assertEqual(data["names"], ["person", "head", "helmet"])

    def test_face_recognition_binary_dataset_can_be_constructed(self):
        from face.recognition_dataset import BinaryFaceDataset

        dataset = BinaryFaceDataset("datasets/faces/train_data", is_train=True)

        self.assertEqual(dataset.root_path, "datasets/faces/train_data")

    def test_training_models_can_be_initialized(self):
        from train_face_detector import build_model as build_face_detector_model
        from train_face_recognition import build_models as build_face_recognition_models
        from train_helmet import build_model as build_helmet_model

        helmet = build_helmet_model("configs/helmet_yolov5s.yaml", nc=3)
        face_detector = build_face_detector_model("face/models/yolov5n.yaml", nc=1)
        recognizer, metric = build_face_recognition_models(num_classes=2)

        self.assertEqual(helmet.nc, 3)
        self.assertEqual(face_detector.nc, 1)
        self.assertEqual(metric.class_dim, 2)
        self.assertTrue(hasattr(recognizer, "forward"))


if __name__ == "__main__":
    unittest.main()
