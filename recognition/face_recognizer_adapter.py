import sys
from pathlib import Path
from typing import List

from schemas import FaceIdentity

MOBILEFACENET_ROOT = Path("/home/lowkeng/code/LVAN/Pytorch-MobileFaceNet")


class FaceRecognizerAdapter:
    def __init__(
        self,
        face_db_path: str = str(MOBILEFACENET_ROOT / "face_db"),
        yolov5_face_weights: str = "weights/face_yolov5n_0_5.pt",
        mobilefacenet_model_path: str = "weights/mobilefacenet.pth",
        threshold: float = 0.6,
        img_size: int = 640,
        conf_thres: float = 0.6,
        iou_thres: float = 0.5,
    ):
        self.face_db_path = face_db_path
        self.yolov5_face_weights = yolov5_face_weights
        self.mobilefacenet_model_path = mobilefacenet_model_path
        self.threshold = threshold
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.recognizer = None
        self._load_recognizer()

    def _load_recognizer(self) -> None:
        _switch_to_mobilefacenet_imports()
        from face_recognizer import FaceRecognizer

        self.recognizer = FaceRecognizer(
            yolov5_face_weights=self.yolov5_face_weights,
            mobilefacenet_model_path=self.mobilefacenet_model_path,
            face_db_path=self.face_db_path,
            threshold=self.threshold,
            img_size=self.img_size,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            align_face=True,
            use_gray=False,
        )

    def recognize_image(self, image) -> List[FaceIdentity]:
        face_imgs, boxes, _ = self.recognizer.yolov5_face_detect(image)
        if face_imgs is None or boxes is None or len(face_imgs) == 0:
            return []

        face_imgs_pre = self.recognizer.preprocess_face(face_imgs)
        features = self.recognizer.extract_face_features(face_imgs_pre)
        names = [self._match_name(feature) for feature in features]

        return [
            FaceIdentity(name=name, confidence=confidence, box=tuple(float(v) for v in box))
            for box, (name, confidence) in zip(boxes, names)
        ]

    def _match_name(self, feature):
        if not self.recognizer.face_db:
            return "unknown", 0.0

        similarities = {
            name: self.recognizer.calculate_cosine_similarity(feature, db_feature)
            for name, db_feature in self.recognizer.face_db.items()
        }
        name, confidence = max(similarities.items(), key=lambda item: item[1])
        if confidence <= self.threshold:
            return "unknown", float(confidence)
        return name, float(confidence)


def _switch_to_mobilefacenet_imports() -> None:
    if not MOBILEFACENET_ROOT.exists():
        raise FileNotFoundError(f"参考项目不存在: {MOBILEFACENET_ROOT}")

    # 两个参考项目都有 models/utils 包，加载人脸模块前清掉同名缓存，避免导入串包。
    for module_name in list(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]

    root = str(MOBILEFACENET_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
