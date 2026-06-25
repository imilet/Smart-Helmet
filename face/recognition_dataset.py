import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class BinaryFaceIndex:
    def __init__(self, root_path):
        self.root_path = str(root_path)
        self.offsets = {}
        self.labels = {}
        self.num_classes = 0
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        header_path = Path(self.root_path + ".header")
        label_path = Path(self.root_path + ".label")
        if not header_path.exists() or not label_path.exists():
            self._loaded = True
            return

        with header_path.open("rb") as file:
            for line in file:
                key, value_pos, value_len = line.rstrip(b"\n").split(b"\t")
                self.offsets[key] = (int(value_pos), int(value_len))

        persons = set()
        with label_path.open("rb") as file:
            for line in file:
                key, label = line.rstrip(b"\n").split(b"\t")
                label_id = int(label)
                self.labels[key] = label_id
                persons.add(label_id)
        self.num_classes = len(persons)
        self._loaded = True

    def read_image_bytes(self, key):
        value_pos, value_len = self.offsets[key]
        with open(self.root_path + ".data", "rb") as file:
            file.seek(value_pos)
            return file.read(value_len)


class BinaryFaceDataset(Dataset):
    def __init__(self, root_path, is_train=True, image_size=112):
        self.root_path = str(root_path)
        self.is_train = is_train
        self.image_size = image_size
        self.index = BinaryFaceIndex(self.root_path)
        self.keys = None

    def _ensure_loaded(self):
        if self.keys is not None:
            return
        self.index.load()
        self.keys = list(self.index.labels.keys())
        if self.is_train:
            np.random.shuffle(self.keys)

    @property
    def num_classes(self):
        self._ensure_loaded()
        return self.index.num_classes

    def __len__(self):
        self._ensure_loaded()
        return len(self.keys)

    def __getitem__(self, index):
        self._ensure_loaded()
        key = self.keys[index]
        image_bytes = self.index.read_image_bytes(key)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = preprocess_face_image(image, self.image_size, self.is_train)
        return image.astype(np.float32), self.index.labels[key]


def preprocess_face_image(image, image_size=112, is_train=False):
    image = cv2.resize(image, (image_size, image_size))
    if is_train and random.random() > 0.5:
        image = cv2.flip(image, 1)
    if is_train:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ops = [random_brightness, random_contrast, random_color]
        np.random.shuffle(ops)
        if random.random() > 0.5:
            image = ops[0](image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = image.transpose((2, 0, 1))
    return (image - 127.5) / 127.5


def random_brightness(image, lower=0.7, upper=1.3):
    return ImageEnhance.Brightness(image).enhance(np.random.uniform(lower, upper))


def random_contrast(image, lower=0.7, upper=1.3):
    return ImageEnhance.Contrast(image).enhance(np.random.uniform(lower, upper))


def random_color(image, lower=0.7, upper=1.3):
    return ImageEnhance.Color(image).enhance(np.random.uniform(lower, upper))
