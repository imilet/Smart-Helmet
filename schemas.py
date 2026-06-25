from dataclasses import dataclass
from typing import Tuple

Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class FaceIdentity:
    name: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class PersonHelmetResult:
    name: str
    confidence: float
    face_box: Box
    has_helmet: bool
