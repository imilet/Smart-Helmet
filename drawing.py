from typing import Iterable

import cv2

from schemas import Detection, PersonHelmetResult


def draw_detections(image, detections: Iterable[Detection]) -> None:
    colors = {
        "person": (255, 180, 0),
        "head": (0, 180, 255),
        "helmet": (0, 200, 0),
    }
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box]
        color = colors.get(det.class_name, (180, 180, 180))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{det.class_name} {det.confidence:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_person_results(image, results: Iterable[PersonHelmetResult]) -> None:
    for result in results:
        x1, y1, x2, y2 = [int(v) for v in result.face_box]
        color = (0, 200, 0) if result.has_helmet else (0, 0, 255)
        status = "helmet" if result.has_helmet else "no_helmet"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{result.name} {status}",
            (x1, max(0, y1 - 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
