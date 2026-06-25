import argparse
import os
from pathlib import Path

import cv2

from detectors.helmet_detector import HelmetDetector
from drawing import draw_detections, draw_person_results
from pipeline import associate_faces_with_helmets
from recognition.face_recognizer_adapter import FaceRecognizerAdapter

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}


def parse_args():
    parser = argparse.ArgumentParser(description="图片/视频安全帽检测 + 人脸识别")
    parser.add_argument("--source", required=True, help="输入图片或视频路径")
    parser.add_argument("--output", default="runs/output", help="输出目录")
    parser.add_argument("--helmet-weights", default="weights/helmet_head_person_s.pt", help="安全帽检测权重")
    parser.add_argument("--device", default="cpu", help="推理设备，例如 cpu 或 0")
    parser.add_argument("--face-db", default="face_db", help="人脸库目录")
    parser.add_argument("--face-threshold", type=float, default=0.6, help="人脸识别相似度阈值")
    return parser.parse_args()


def build_models(args):
    face_recognizer = FaceRecognizerAdapter(
        face_db_path=args.face_db,
        threshold=args.face_threshold,
    )
    helmet_detector = HelmetDetector(
        weights=args.helmet_weights,
        device=args.device,
    )
    return helmet_detector, face_recognizer


def process_image(source: Path, output_dir: Path, helmet_detector, face_recognizer) -> Path:
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"无法读取图片: {source}")

    detections = helmet_detector.detect(image)
    faces = face_recognizer.recognize_image(image)
    results = associate_faces_with_helmets(faces, detections)

    draw_detections(image, detections)
    draw_person_results(image, results)

    output_path = output_dir / source.name
    cv2.imwrite(str(output_path), image)
    return output_path


def process_video(source: Path, output_dir: Path, helmet_detector, face_recognizer) -> Path:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"无法读取视频: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = output_dir / f"{source.stem}_result.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        detections = helmet_detector.detect(frame)
        faces = face_recognizer.recognize_image(frame)
        results = associate_faces_with_helmets(faces, detections)
        draw_detections(frame, detections)
        draw_person_results(frame, results)
        writer.write(frame)

        frame_index += 1
        if frame_index % 10 == 0:
            print(f"已处理 {frame_index} 帧")

    capture.release()
    writer.release()
    return output_path


def main():
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"输入文件不存在: {source}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    helmet_detector, face_recognizer = build_models(args)
    suffix = source.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        output_path = process_image(source, output_dir, helmet_detector, face_recognizer)
    elif suffix in VIDEO_SUFFIXES:
        output_path = process_video(source, output_dir, helmet_detector, face_recognizer)
    else:
        raise ValueError(f"暂不支持的输入类型: {source.suffix}")

    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
