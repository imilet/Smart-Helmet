from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

from legacy_imports import use_legacy_package
from face.models.experimental import attempt_load
from face.utils.datasets import letterbox
from face.utils.general import non_max_suppression_face, scale_coords
from face.utils_face import load_image, norm_crop, scale_coords_landmarks
from schemas import FaceIdentity


class FaceRecognizerAdapter:
    def __init__(
        self,
        face_db_path: str = "face_db",
        yolov5_face_weights: str = "weights/face_yolov5n_0_5.pt",
        mobilefacenet_model_path: str = "weights/mobilefacenet.pth",
        threshold: float = 0.6,
        img_size: int = 640,
        conf_thres: float = 0.6,
        iou_thres: float = 0.5,
    ):
        self.face_db_path = Path(face_db_path)
        self.yolov5_face_weights = str(Path(yolov5_face_weights))
        self.mobilefacenet_model_path = str(Path(mobilefacenet_model_path))
        self.threshold = threshold
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_legacy_package("face")
        self.face_detector = attempt_load(self.yolov5_face_weights, map_location=self.device).eval()
        self.feature_model = torch.jit.load(self.mobilefacenet_model_path, map_location=self.device).eval()
        self.face_db = self._load_face_db()

    def recognize_image(self, image) -> List[FaceIdentity]:
        face_images, boxes = self._detect_faces(image)
        if not face_images:
            return []

        features = self._extract_features(face_images)
        matched = [self._match_name(feature) for feature in features]
        return [
            FaceIdentity(name=name, confidence=confidence, box=tuple(float(v) for v in box))
            for box, (name, confidence) in zip(boxes, matched)
        ]

    def _detect_faces(self, image):
        original = image.copy()
        img = letterbox(image, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.face_detector(img_tensor)[0]
            pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)

        faces = []
        boxes = []
        det = pred[0]
        if det is None or len(det) == 0:
            return faces, boxes

        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], original.shape).round()
        det[:, 5:15] = scale_coords_landmarks(img_tensor.shape[2:], det[:, 5:15], original.shape).round()

        for item in det:
            x1, y1, x2, y2 = item[:4].cpu().numpy().astype(int)
            landmarks = item[5:15].cpu().numpy().astype(float)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original.shape[1], x2), min(original.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            face = norm_crop(original, landmarks)
            if face is None or face.size == 0:
                face = original[y1:y2, x1:x2]
            face = cv2.resize(face, (112, 112))
            faces.append(face)
            boxes.append((x1, y1, x2, y2))
        return faces, boxes

    def _extract_features(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        batch = np.asarray(face_images, dtype=np.float32)
        batch = batch.transpose((0, 3, 1, 2))
        batch = (batch - 127.5) / 127.5
        face_tensor = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            features = self.feature_model(face_tensor).detach().cpu().numpy()
        return [feature for feature in features]

    def _load_face_db(self) -> Dict[str, np.ndarray]:
        face_db = {}
        if not self.face_db_path.exists():
            return face_db

        for image_path in sorted(self.face_db_path.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = load_image(str(image_path))
            if image is None:
                continue
            face_images, _ = self._detect_faces(image)
            if len(face_images) != 1:
                continue
            features = self._extract_features(face_images)
            if features:
                face_db[image_path.stem] = features[0]
        return face_db

    def _match_name(self, feature):
        if not self.face_db:
            return "unknown", 0.0

        similarities = {
            name: _cosine_similarity(feature, db_feature)
            for name, db_feature in self.face_db.items()
        }
        name, confidence = max(similarities.items(), key=lambda item: item[1])
        if confidence <= self.threshold:
            return "unknown", float(confidence)
        return name, float(confidence)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm < 1e-6 or right_norm < 1e-6:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
