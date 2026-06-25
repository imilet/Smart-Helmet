# Smart Helmet Detection

图片/视频文件安全帽检测 + 人脸识别。

## 环境

使用已有 conda 环境：

```bash
conda activate y8face_cpu
```

已验证该环境可导入 `torch`、`cv2`、`numpy`，并能加载安全帽检测和人脸识别模型。

## 模型文件

- 安全帽检测：`weights/helmet_head_person_s.pt`
- 人脸检测：`weights/face_yolov5n_0_5.pt`
- 人脸识别：`weights/mobilefacenet.pth`
- 默认人脸库：`face_db`

安全帽类别为：

```text
person, head, helmet
```

## 图片检测

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp python detect.py \
  --source /path/to/input.jpg \
  --output runs/output \
  --device cpu
```

## 视频检测

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp python detect.py \
  --source /path/to/input.mp4 \
  --output runs/output \
  --device cpu
```

## 训练

安全帽检测训练使用 YOLOv5 格式数据，默认数据配置在 `configs/helmet_data.yaml`：

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp python train_helmet.py \
  --data configs/helmet_data.yaml \
  --cfg configs/helmet_yolov5s.yaml \
  --device cpu
```

默认安全帽数据目录：

```text
datasets/helmet/images/train
datasets/helmet/images/val
datasets/helmet/labels/train
datasets/helmet/labels/val
```

人脸检测训练使用 YOLOv5-face 格式数据，默认数据配置在 `configs/face_detector_data.yaml`：

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp python train_face_detector.py \
  --data configs/face_detector_data.yaml \
  --cfg face/models/yolov5n.yaml \
  --device cpu
```

人脸识别训练使用二进制数据根路径，不带后缀传入；实际文件需要成组存在：

```text
datasets/faces/train_data.data
datasets/faces/train_data.label
datasets/faces/train_data.header
```

可先将按身份分目录的人脸图片转换为二进制格式：

```text
datasets/faces/raw
├── person_a
│   ├── 1.jpg
│   └── 2.jpg
└── person_b
    ├── 1.jpg
    └── 2.jpg
```

转换命令：

```bash
PYTHONPATH=. python convert_face_dataset.py \
  --input-dir datasets/faces/raw \
  --output-dir datasets/faces \
  --val-ratio 0.02
```

转换后会生成：

```text
datasets/faces/train_data.data
datasets/faces/train_data.header
datasets/faces/train_data.label
datasets/faces/val_data.data
datasets/faces/val_data.header
datasets/faces/val_data.label
```

训练命令：

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp python train_face_recognition.py \
  --train-root datasets/faces/train_data \
  --device cpu
```

## 测试

```bash
PYTHONPATH=. python tests/test_common_yolo.py
PYTHONPATH=. python tests/test_association.py
PYTHONPATH=. python tests/test_project_independence.py
PYTHONPATH=. python tests/test_face_dataset_conversion.py
PYTHONPATH=. MPLCONFIGDIR=/tmp python tests/test_training_entrypoints.py
```

## 当前规则

第一版采用简单空间关联规则：

- 先将人脸框匹配到重叠的 `head` 框；
- 如果匹配到头部，则在头部上方扩展区域查找 `helmet`；
- 如果没有匹配到头部，则直接在人脸上方扩展区域查找 `helmet`。

这能先跑通图片/视频流程，但复杂多人遮挡场景可能误判，后续可继续优化关联规则。


## 持续更新中...