import sys
from pathlib import Path
from typing import List

import torch

from schemas import Detection

SMART_CONSTRUCTION_ROOT = Path("/home/lowkeng/code/LVAN/Smart_Construction-master")


class HelmetDetector:
    def __init__(
        self,
        weights: str = "helmet_head_person_s.pt",
        img_size: int = 640,
        conf_thres: float = 0.4,
        iou_thres: float = 0.5,
        device: str = "",
    ):
        self.weights = str(Path(weights).resolve())
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device_name = device
        self.model = None
        self.device = None
        self.half = False
        self.names = None
        self._load_model()

    def _load_model(self) -> None:
        _ensure_reference_path(SMART_CONSTRUCTION_ROOT)
        from models.experimental import attempt_load
        from utils.utils import check_img_size
        from utils import torch_utils

        self.device = torch_utils.select_device(self.device_name)
        self.model = attempt_load(self.weights, map_location=self.device)
        self.img_size = check_img_size(self.img_size, s=self.model.stride.max())
        self.half = self.device.type != "cpu"
        if self.half:
            self.model.half()
        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names

    def detect(self, image) -> List[Detection]:
        from utils.datasets import letterbox
        from utils.utils import non_max_suppression, scale_coords

        img = letterbox(image, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = img.copy()
        img_tensor = torch.from_numpy(img).to(self.device)
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                classes=None,
                agnostic=False,
            )

        detections = []
        det = pred[0]
        if det is None or len(det) == 0:
            return detections

        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], image.shape).round()
        for *xyxy, conf, cls in det:
            class_name = self.names[int(cls)]
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(conf),
                    box=tuple(float(v) for v in xyxy),
                )
            )
        return detections


def _ensure_reference_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"参考项目不存在: {path}")
    for module_name in list(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
