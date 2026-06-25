import argparse
import random
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


@dataclass
class ConversionResult:
    train_samples: int
    val_samples: int
    train_classes: int
    val_classes: int


class BinaryFaceDatasetWriter:
    def __init__(self, output_prefix):
        self.output_prefix = str(output_prefix)
        self.data_file = open(self.output_prefix + ".data", "wb")
        self.header_file = open(self.output_prefix + ".header", "wb")
        self.label_file = open(self.output_prefix + ".label", "wb")
        self.offset = 0

    def add_image(self, key, image_bytes):
        key_bytes = key.encode("ascii")
        self.data_file.write(struct.pack("I", len(key_bytes)))
        self.data_file.write(key_bytes)
        self.data_file.write(struct.pack("I", len(image_bytes)))
        self.data_file.write(image_bytes)
        self.offset += 4 + len(key_bytes) + 4
        self.header_file.write(f"{key}\t{self.offset}\t{len(image_bytes)}\n".encode("ascii"))
        self.offset += len(image_bytes)

    def add_label(self, key, label_id):
        self.label_file.write(f"{key}\t{label_id}\n".encode("ascii"))

    def close(self):
        self.data_file.close()
        self.header_file.close()
        self.label_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="将按身份分目录的人脸图片转换为 MobileFaceNet 二进制训练数据")
    parser.add_argument("--input-dir", default="datasets/faces/raw", help="原始人脸目录，每个子目录代表一个身份")
    parser.add_argument("--output-dir", default="datasets/faces", help="输出 .data/.header/.label 的目录")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="验证集目标比例，按身份切分")
    parser.add_argument("--val-max-samples", type=int, default=100000, help="验证集最大图片数，<=0 表示不限制")
    parser.add_argument("--seed", type=int, default=2024, help="身份切分随机种子")
    return parser.parse_args()


def collect_person_images(root_path):
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")

    person_images = []
    for person_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = sorted(
            path for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            person_images.append((person_dir.name, images))
        else:
            print(f"警告：文件夹 {person_dir.name} 中没有有效图片，已跳过")
    return person_images


def split_person_identities(person_images, val_ratio, val_max_samples, seed):
    total_samples = sum(len(images) for _, images in person_images)
    target_val = int(total_samples * val_ratio)
    if val_max_samples > 0:
        target_val = min(target_val, val_max_samples)

    rng = random.Random(seed)
    shuffled = list(person_images)
    rng.shuffle(shuffled)

    val_persons = []
    train_persons = []
    val_samples = 0
    for person_name, images in shuffled:
        if len(images) >= 2 and val_samples + len(images) <= target_val:
            val_persons.append((person_name, images))
            val_samples += len(images)
        else:
            train_persons.append((person_name, images))

    return _assign_labels(train_persons), _assign_labels(val_persons), len(train_persons), len(val_persons)


def _assign_labels(person_images):
    data = []
    for label_id, (_, images) in enumerate(sorted(person_images, key=lambda item: item[0])):
        for image_path in images:
            data.append((image_path, label_id))
    return data


def write_binary_dataset(data, output_prefix, desc):
    writer = BinaryFaceDatasetWriter(output_prefix)
    written = 0
    try:
        for image_path, label_id in tqdm(data, desc=desc):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"警告：跳过损坏或无法读取的图片 {image_path}")
                continue
            ok, encoded = cv2.imencode(".bmp", image)
            if not ok:
                print(f"警告：图片编码失败，已跳过 {image_path}")
                continue
            key = str(uuid.uuid1())
            writer.add_image(key, encoded.tobytes())
            writer.add_label(key, label_id)
            written += 1
    finally:
        writer.close()
    return written


def convert_data(root_path, output_dir, val_ratio=0.02, val_max_samples=100000, seed=2024):
    person_images = collect_person_images(root_path)
    total_samples = sum(len(images) for _, images in person_images)
    if total_samples == 0:
        raise ValueError("没有找到任何有效图片，请检查输入路径和图片格式")

    train_data, val_data, train_classes, val_classes = split_person_identities(
        person_images, val_ratio, val_max_samples, seed
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_count = write_binary_dataset(train_data, output_path / "train_data", "转换训练数据")
    val_count = write_binary_dataset(val_data, output_path / "val_data", "转换验证数据")
    return ConversionResult(
        train_samples=train_count,
        val_samples=val_count,
        train_classes=train_classes,
        val_classes=val_classes,
    )


def main():
    args = parse_args()
    result = convert_data(args.input_dir, args.output_dir, args.val_ratio, args.val_max_samples, args.seed)
    print(
        "数据转换完成！"
        f"训练集：{result.train_samples} 张/{result.train_classes} 类，"
        f"验证集：{result.val_samples} 张/{result.val_classes} 类"
    )


if __name__ == "__main__":
    main()
