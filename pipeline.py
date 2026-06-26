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


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


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
    best_score = 0.0
    for head in heads:
        score = score_head_match(face, head)
        if score > best_score:
            best_score = score
            best_head = head
    if best_score >= 0.45:
        return best_head
    return None


def score_head_match(face: FaceIdentity, head: Detection) -> float:
    face_overlap = overlap_ratio(face.box, head.box)
    face_cx, face_cy = box_center(face.box)
    head_cx, head_cy = box_center(head.box)
    head_width = max(1.0, head.box[2] - head.box[0])
    head_height = max(1.0, head.box[3] - head.box[1])
    center_distance = abs(face_cx - head_cx) / head_width + abs(face_cy - head_cy) / head_height
    center_score = clamp01(1.0 - center_distance / 1.2)
    return 0.65 * face_overlap + 0.25 * center_score + 0.10 * clamp01(head.confidence)


def has_nearby_helmet(region: Box, helmets: Iterable[Detection]) -> bool:
    best_score = 0.0
    for helmet in helmets:
        best_score = max(best_score, score_helmet_match(region, helmet))
    return best_score >= 0.60


def score_helmet_match(region: Box, helmet: Detection) -> float:
    search_region = expand_head_region(region)
    helmet_overlap = overlap_ratio(helmet.box, search_region)
    region_overlap = overlap_ratio(search_region, helmet.box)
    overlap_score = max(helmet_overlap, region_overlap)

    region_cx, region_cy = box_center(region)
    helmet_cx, helmet_cy = box_center(helmet.box)
    region_width = max(1.0, region[2] - region[0])
    region_height = max(1.0, region[3] - region[1])
    horizontal_score = clamp01(1.0 - abs(helmet_cx - region_cx) / (region_width * 0.9))
    vertical_offset = (region_cy - helmet_cy) / region_height
    upper_score = clamp01((vertical_offset + 0.2) / 0.8)

    return (
        0.45 * overlap_score
        + 0.25 * upper_score
        + 0.20 * horizontal_score
        + 0.10 * clamp01(helmet.confidence)
    )


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
