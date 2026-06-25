import os
import cv2
import numpy as np
import torch
from typing import Tuple, Optional
from skimage import transform as trans

# ===================== 常量定义（提升可维护性） =====================
# 人脸关键点标准坐标（用于对齐）
STANDARD_FACE_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# 模型输入尺寸
FACE_ALIGN_SIZE = 112
FONT_DEFAULT_SIZE = 50
FONT_COLOR_RED = (255, 0, 0)
BOX_COLOR_BLUE = (0, 0, 255)
BOX_THICKNESS = 2

# ===================== 工具函数（解耦核心逻辑） =====================
def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    安全加载图片（兼容中文路径）
    :param image_path: 图片路径
    :return: BGR格式图片数组，加载失败返回None
    """
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"警告：无法读取图片 {image_path}（格式不支持或文件损坏）")
        return img
    except Exception as e:
        print(f"错误：读取图片 {image_path} 时出错 - {str(e)}")
        return None

def scale_coords_landmarks(img1_shape: Tuple[int, int], 
                           coords: torch.Tensor, 
                           img0_shape: Tuple[int, int], 
                           ratio_pad: Optional[Tuple] = None) -> torch.Tensor:
    """将人脸关键点坐标从缩放后的图片映射回原始图片（优化边界检查逻辑）"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # 批量处理关键点坐标（简化代码）
    coords[:, ::2] -= pad[0]  # 所有x坐标
    coords[:, 1::2] -= pad[1] # 所有y坐标
    coords[:, :10] /= gain
    
    # 统一边界检查（简化代码）
    coords[:, ::2] = coords[:, ::2].clamp(0, img0_shape[1])  # x坐标限制在图片宽度内
    coords[:, 1::2] = coords[:, 1::2].clamp(0, img0_shape[0]) # y坐标限制在图片高度内
    return coords

def estimate_norm(lmk: np.ndarray) -> np.ndarray:
    """计算人脸对齐所需的变换矩阵（优化输入校验）"""
    lmk = np.array(lmk, dtype=np.float32).reshape(5, 2)
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, STANDARD_FACE_LANDMARKS)
    return tform.params[0:2, :]

def norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = FACE_ALIGN_SIZE) -> np.ndarray:
    """人脸对齐（增加异常处理）"""
    try:
        M = estimate_norm(landmark)
        return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    except Exception as e:
        print(f"警告：人脸对齐失败 - {str(e)}，使用原始裁剪图")
        return img
    
