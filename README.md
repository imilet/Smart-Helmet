# Smart Helmet Detection

图片/视频文件安全帽检测 + 人脸识别。

项目内已包含推理所需源码、模型和默认人脸库，不依赖外部参考项目源码。

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

## 测试

```bash
PYTHONPATH=. python tests/test_association.py
PYTHONPATH=. python tests/test_project_independence.py
```

## 当前规则

第一版采用简单空间关联规则：

- 先将人脸框匹配到重叠的 `head` 框；
- 如果匹配到头部，则在头部上方扩展区域查找 `helmet`；
- 如果没有匹配到头部，则直接在人脸上方扩展区域查找 `helmet`。

这能先跑通图片/视频流程，但复杂多人遮挡场景可能误判，后续可继续优化关联规则。
