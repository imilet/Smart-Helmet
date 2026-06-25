from typing import Iterable, List, Optional

from schemas import Box, Detection, FaceIdentity, PersonHelmetResult


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return box_area((x1, y1, x2, y2))


def overlap_ratio(inner: Box, outer: Box) -> float:
    area = box_area(inner)
    if area == 0:
        return 0.0
    return intersection_area(inner, outer) / area


def expand_head_region(box: Box) -> Box:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    # 安全帽通常位于头/脸上半部分，向上扩张并略向左右放宽，减少漏关联。
    return (
        x1 - width * 0.25,
        y1 - height * 0.35,
        x2 + width * 0.25,
        y2 + height * 0.05,
    )


def find_matching_head(face: FaceIdentity, heads: Iterable[Detection]) -> Optional[Detection]:
    best_head = None
    best_overlap = 0.0
    for head in heads:
        ratio = overlap_ratio(face.box, head.box)
        if ratio > best_overlap:
            best_overlap = ratio
            best_head = head
    if best_overlap >= 0.35:
        return best_head
    return None


def has_nearby_helmet(region: Box, helmets: Iterable[Detection]) -> bool:
    search_region = expand_head_region(region)
    for helmet in helmets:
        if overlap_ratio(helmet.box, search_region) >= 0.2:
            return True
        if overlap_ratio(search_region, helmet.box) >= 0.2:
            return True
    return False


def associate_faces_with_helmets(
    faces: Iterable[FaceIdentity],
    detections: Iterable[Detection],
) -> List[PersonHelmetResult]:
    detections = list(detections)
    heads = [item for item in detections if item.class_name == "head"]
    helmets = [item for item in detections if item.class_name == "helmet"]

    results = []
    for face in faces:
        matched_head = find_matching_head(face, heads)
        helmet_region = matched_head.box if matched_head is not None else face.box
        results.append(
            PersonHelmetResult(
                name=face.name,
                confidence=face.confidence,
                face_box=face.box,
                has_helmet=has_nearby_helmet(helmet_region, helmets),
            )
        )
    return results
