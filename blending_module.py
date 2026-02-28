"""
图像融合模块 (Blending Module)
================================

用于超高分辨率图像生成系统的图像融合算法实现。

功能特性:
    - 拉普拉斯金字塔融合（推荐方案）
    - 泊松融合（备选方案）
    - 加权平均融合（快速方案）
    - 接缝检测与修复
    - 色彩一致性校正

技术规格:
    - 金字塔层数: 6层（含原始图像）
    - 接缝检测窗口: 16x16
    - SSIM阈值: 0.95
    - 支持CUDA加速

作者: AI Assistant
版本: 1.0.0
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import warnings
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """融合方法枚举"""
    LAPLACIAN = "laplacian"           # 拉普拉斯金字塔融合
    POISSON = "poisson"               # 泊松融合
    WEIGHTED_AVERAGE = "weighted"     # 加权平均融合


class PoissonMode(Enum):
    """泊松融合模式"""
    NORMAL = cv2.NORMAL_CLONE           # 正常克隆
    MIXED = cv2.MIXED_CLONE             # 混合克隆
    MONOCHROME = cv2.MONOCHROME_TRANSFER  # 单色传输


class WeightType(Enum):
    """权重类型"""
    LINEAR = "linear"                   # 线性衰减
    COSINE = "cosine"                   # 余弦窗
    SIGMOID = "sigmoid"                 # S型过渡


@dataclass
class Seam:
    """
    接缝数据类
    
    用于存储检测到的接缝信息，包括位置、严重程度和修复建议。
    
    Attributes:
        x: 接缝左上角x坐标
        y: 接缝左上角y坐标
        width: 接缝区域宽度
        height: 接缝区域高度
        ssim_score: 局部SSIM分数（越低表示接缝越明显）
        severity: 严重程度（'low', 'medium', 'high'）
        suggested_fix: 建议修复方法
    """
    x: int
    y: int
    width: int
    height: int
    ssim_score: float
    severity: str = field(default="low")
    suggested_fix: str = field(default="")
    
    def __post_init__(self):
        """根据SSIM分数确定严重程度"""
        if self.ssim_score < 0.85:
            self.severity = "high"
            self.suggested_fix = "poisson_refinement"
        elif self.ssim_score < 0.92:
            self.severity = "medium"
            self.suggested_fix = "increase_blend_width"
        else:
            self.severity = "low"
            self.suggested_fix = "none"


@dataclass
class TileInfo:
    """
    瓦片信息数据类
    
    Attributes:
        image: 瓦片图像数据
        x: 瓦片在全局图像中的x坐标
        y: 瓦片在全局图像中的y坐标
        row: 瓦片行索引
        col: 瓦片列索引
    """
    image: np.ndarray
    x: int
    y: int
    row: int
    col: int


@dataclass
class OverlapRegion:
    """
    重叠区域数据类
    
    Attributes:
        tile1_idx: 第一个瓦片索引
        tile2_idx: 第二个瓦片索引
        x1_start, y1_start: 在tile1中的重叠区域起始坐标
        x2_start, y2_start: 在tile2中的重叠区域起始坐标
        width, height: 重叠区域尺寸
        direction: 重叠方向 ('horizontal' 或 'vertical')
    """
    tile1_idx: int
    tile2_idx: int
    x1_start: int
    y1_start: int
    x2_start: int
    y2_start: int
    width: int
    height: int
    direction: str


class BlendingModule:
    """
    图像融合模块
    
    实现多种图像融合算法，用于超高分辨率图像生成系统的瓦片融合。
    
    主要功能:
        1. 拉普拉斯金字塔融合 - 多分辨率无缝融合
        2. 泊松融合 - 梯度域融合
        3. 加权平均融合 - 快速线性融合
        4. 接缝检测 - 基于SSIM的局部质量评估
        5. 色彩校正 - 全局直方图匹配和局部引导滤波
    
    Attributes:
        method: 默认融合方法
        num_levels: 金字塔层数
        ssim_threshold: 接缝检测SSIM阈值
        use_cuda: 是否使用CUDA加速
    
    Example:
        >>> blender = BlendingModule(method='laplacian', num_levels=6)
        >>> result = blender.laplacian_fusion(tiles, overlap_map)
        >>> seams = blender.detect_seams(result, tiles)
    """
    
    def __init__(
        self,
        method: str = 'laplacian',
        num_levels: int = 6,
        ssim_threshold: float = 0.95,
        use_cuda: bool = False
    ):
        """
        初始化融合模块
        
        Args:
            method: 默认融合方法 ('laplacian', 'poisson', 'weighted')
            num_levels: 金字塔层数（拉普拉斯融合）
            ssim_threshold: 接缝检测SSIM阈值
            use_cuda: 是否尝试使用CUDA加速
        
        Raises:
            ValueError: 如果method参数无效
        """
        self.method = FusionMethod(method)
        self.num_levels = num_levels
        self.ssim_threshold = ssim_threshold
        self.use_cuda = use_cuda and self._check_cuda_available()
        
        # 初始化CUDA
        if self.use_cuda:
            self._init_cuda()
        
        logger.info(f"BlendingModule initialized: method={method}, "
                   f"levels={num_levels}, cuda={self.use_cuda}")
    
    def _check_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        try:
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            if cuda_available:
                logger.info(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
            return cuda_available
        except Exception as e:
            logger.warning(f"CUDA check failed: {e}")
            return False
    
    def _init_cuda(self):
        """初始化CUDA设备"""
        try:
            cv2.cuda.setDevice(0)
            logger.info("CUDA device initialized")
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}")
            self.use_cuda = False
    
    # ==================== 金字塔构建方法 ====================
    
    def build_gaussian_pyramid(
        self,
        image: np.ndarray,
        levels: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        构建高斯金字塔
        
        通过连续下采样构建多分辨率表示，每层尺寸为上一层的1/2。
        
        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            levels: 金字塔层数，默认使用初始化时设置的层数
        
        Returns:
            高斯金字塔列表，从原始分辨率到最低分辨率
        
        Example:
            >>> pyramid = blender.build_gaussian_pyramid(image, levels=6)
            >>> print([p.shape for p in pyramid])
        """
        if levels is None:
            levels = self.num_levels
        
        # 确保图像是float32类型
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        pyramid = [image.copy()]
        current = image.copy()
        
        for i in range(levels - 1):
            # 检查是否可以继续下采样
            if current.shape[0] < 2 or current.shape[1] < 2:
                logger.warning(f"Stopping pyramid at level {i+1}: image too small")
                break
            
            # 使用OpenCV的pyrDown进行下采样
            if self.use_cuda and len(current.shape) == 3:
                try:
                    gpu_current = cv2.cuda.GpuMat()
                    gpu_current.upload(current)
                    gpu_down = cv2.cuda.pyrDown(gpu_current)
                    current = gpu_down.download()
                except Exception as e:
                    logger.debug(f"CUDA pyrDown failed, using CPU: {e}")
                    current = cv2.pyrDown(current)
            else:
                current = cv2.pyrDown(current)
            
            pyramid.append(current)
        
        return pyramid
    
    def build_laplacian_pyramid(
        self,
        gaussian_pyramid: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        构建拉普拉斯金字塔
        
        拉普拉斯金字塔 = 高斯金字塔层 - 上采样(高斯金字塔下一层)
        表示图像的细节信息，用于多分辨率融合。
        
        Args:
            gaussian_pyramid: 高斯金字塔列表
        
        Returns:
            拉普拉斯金字塔列表
        
        Note:
            最后一层与原始高斯金字塔最后一层相同（最低分辨率）
        """
        laplacian_pyramid = []
        
        for i in range(len(gaussian_pyramid) - 1):
            current = gaussian_pyramid[i]
            next_level = gaussian_pyramid[i + 1]
            
            # 上采样下一层到当前尺寸
            if self.use_cuda and len(current.shape) == 3:
                try:
                    gpu_next = cv2.cuda.GpuMat()
                    gpu_next.upload(next_level)
                    # 获取目标尺寸
                    target_size = (current.shape[1], current.shape[0])
                    gpu_up = cv2.cuda.pyrUp(gpu_next, dstsize=target_size)
                    up_sampled = gpu_up.download()
                except Exception as e:
                    logger.debug(f"CUDA pyrUp failed, using CPU: {e}")
                    up_sampled = cv2.pyrUp(next_level, dstsize=(current.shape[1], current.shape[0]))
            else:
                up_sampled = cv2.pyrUp(next_level, dstsize=(current.shape[1], current.shape[0]))
            
            # 计算拉普拉斯层
            laplacian = current - up_sampled
            laplacian_pyramid.append(laplacian)
        
        # 最后一层使用原始高斯金字塔的最低分辨率层
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        return laplacian_pyramid
    
    def collapse_laplacian_pyramid(
        self,
        laplacian_pyramid: List[np.ndarray]
    ) -> np.ndarray:
        """
        重建图像从拉普拉斯金字塔
        
        通过逐层上采样和叠加重建原始图像。
        
        Args:
            laplacian_pyramid: 拉普拉斯金字塔列表
        
        Returns:
            重建后的图像
        
        Example:
            >>> lap_pyr = blender.build_laplacian_pyramid(gauss_pyr)
            >>> reconstructed = blender.collapse_laplacian_pyramid(lap_pyr)
        """
        # 从最低分辨率开始
        current = laplacian_pyramid[-1].copy()
        
        # 逐层重建
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            layer = laplacian_pyramid[i]
            
            # 上采样当前层
            if self.use_cuda and len(current.shape) == 3:
                try:
                    gpu_current = cv2.cuda.GpuMat()
                    gpu_current.upload(current)
                    target_size = (layer.shape[1], layer.shape[0])
                    gpu_up = cv2.cuda.pyrUp(gpu_current, dstsize=target_size)
                    current = gpu_up.download()
                except Exception as e:
                    logger.debug(f"CUDA pyrUp failed, using CPU: {e}")
                    current = cv2.pyrUp(current, dstsize=(layer.shape[1], layer.shape[0]))
            else:
                current = cv2.pyrUp(current, dstsize=(layer.shape[1], layer.shape[0]))
            
            # 叠加拉普拉斯细节
            current = current + layer
        
        return current



    # ==================== 融合方法 ====================
    
    def laplacian_fusion(
        self,
        tiles: List[Union[np.ndarray, TileInfo]],
        overlap_map: Optional[List[OverlapRegion]] = None,
        output_shape: Optional[Tuple[int, int]] = None,
        weight_type: WeightType = WeightType.COSINE
    ) -> np.ndarray:
        """
        拉普拉斯金字塔融合（推荐方案）
        
        多分辨率无缝融合算法，通过在不同尺度上应用距离衰减权重
        实现平滑过渡，避免接缝和鬼影。
        
        算法复杂度: O(N²·4/3)，其中N是图像边长
        
        Args:
            tiles: 瓦片列表，可以是图像数组或TileInfo对象
            overlap_map: 重叠区域映射（可选，自动计算）
            output_shape: 输出图像形状 (height, width)，可选
            weight_type: 权重类型（线性、余弦、S型）
        
        Returns:
            融合后的图像
        
        Example:
            >>> tiles = [tile1, tile2, tile3, tile4]  # 2x2网格
            >>> result = blender.laplacian_fusion(tiles, output_shape=(2048, 2048))
        
        Reference:
            Burt, P. J., & Adelson, E. H. (1983). A multiresolution spline
            with application to image mosaics. ACM TOG, 2(4), 217-236.
        """
        # 转换TileInfo为图像数组
        tile_images = []
        tile_positions = []
        
        for i, tile in enumerate(tiles):
            if isinstance(tile, TileInfo):
                tile_images.append(tile.image)
                tile_positions.append((tile.y, tile.x))
            else:
                tile_images.append(tile)
                # 假设网格布局
                if output_shape:
                    grid_size = int(np.ceil(np.sqrt(len(tiles))))
                    row, col = i // grid_size, i % grid_size
                    tile_h, tile_w = tile.shape[:2]
                    tile_positions.append((row * tile_h, col * tile_w))
        
        # 确定输出尺寸
        if output_shape is None:
            max_h = max(pos[0] + img.shape[0] for img, pos in zip(tile_images, tile_positions))
            max_w = max(pos[1] + img.shape[1] for img, pos in zip(tile_images, tile_positions))
            output_shape = (max_h, max_w)
        
        output_h, output_w = output_shape[:2]
        num_channels = tile_images[0].shape[2] if len(tile_images[0].shape) == 3 else 1
        
        # 初始化权重累加器和结果累加器
        if num_channels == 1:
            weight_accumulator = np.zeros((output_h, output_w), dtype=np.float32)
            result_accumulator = np.zeros((output_h, output_w), dtype=np.float32)
        else:
            weight_accumulator = np.zeros((output_h, output_w), dtype=np.float32)
            result_accumulator = np.zeros((output_h, output_w, num_channels), dtype=np.float32)
        
        # 为每个瓦片构建金字塔并融合
        for tile_img, (y, x) in zip(tile_images, tile_positions):
            tile_h, tile_w = tile_img.shape[:2]
            
            # 确保数据类型一致
            if tile_img.dtype != np.float32:
                tile_img = tile_img.astype(np.float32)
            
            # 构建金字塔
            gauss_pyr = self.build_gaussian_pyramid(tile_img, self.num_levels)
            lap_pyr = self.build_laplacian_pyramid(gauss_pyr)
            
            # 生成距离权重图
            weight_map = self._create_distance_weight_map(
                tile_h, tile_w, weight_type
            )
            
            # 构建权重金字塔
            weight_gauss_pyr = self.build_gaussian_pyramid(weight_map, self.num_levels)
            
            # 对每个金字塔层应用权重
            weighted_lap_pyr = []
            for lap_layer, weight_layer in zip(lap_pyr, weight_gauss_pyr):
                # 调整权重层尺寸以匹配拉普拉斯层
                if weight_layer.shape[:2] != lap_layer.shape[:2]:
                    weight_layer = cv2.resize(
                        weight_layer,
                        (lap_layer.shape[1], lap_layer.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # 扩展权重维度以匹配图像
                if len(lap_layer.shape) == 3 and len(weight_layer.shape) == 2:
                    weight_layer = np.expand_dims(weight_layer, axis=-1)
                
                weighted_layer = lap_layer * weight_layer
                weighted_lap_pyr.append(weighted_layer)
            
            # 重建加权图像
            weighted_tile = self.collapse_laplacian_pyramid(weighted_lap_pyr)
            
            # 将加权图像放置到输出位置
            y_end = min(y + tile_h, output_h)
            x_end = min(x + tile_w, output_w)
            tile_h_actual = y_end - y
            tile_w_actual = x_end - x
            
            # 裁剪到实际区域
            weighted_tile = weighted_tile[:tile_h_actual, :tile_w_actual]
            weight_map_cropped = weight_map[:tile_h_actual, :tile_w_actual]
            
            # 累加
            if num_channels == 1:
                result_accumulator[y:y_end, x:x_end] += weighted_tile
                weight_accumulator[y:y_end, x:x_end] += weight_map_cropped
            else:
                result_accumulator[y:y_end, x:x_end] += weighted_tile
                weight_accumulator[y:y_end, x:x_end] += weight_map_cropped
        
        # 归一化
        # 避免除零
        weight_accumulator = np.maximum(weight_accumulator, 1e-6)
        
        if num_channels == 1:
            result = result_accumulator / weight_accumulator
        else:
            result = result_accumulator / np.expand_dims(weight_accumulator, axis=-1)
        
        # 裁剪到有效范围
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
    
    def _create_distance_weight_map(
        self,
        height: int,
        width: int,
        weight_type: WeightType,
        feather_width: Optional[int] = None
    ) -> np.ndarray:
        """
        创建距离衰减权重图
        
        生成从边缘向中心递增的权重图，用于平滑融合。
        
        Args:
            height: 图像高度
            width: 图像宽度
            weight_type: 权重类型
            feather_width: 羽化宽度（边缘过渡区宽度）
        
        Returns:
            权重图 (H, W)，值域[0, 1]
        """
        if feather_width is None:
            feather_width = min(height, width) // 8
        
        # 创建坐标网格
        y = np.arange(height).reshape(-1, 1)
        x = np.arange(width).reshape(1, -1)
        
        # 计算到边缘的距离
        dist_top = y
        dist_bottom = height - 1 - y
        dist_left = x
        dist_right = width - 1 - x
        
        # 取到最近边缘的距离
        dist_to_edge = np.minimum(np.minimum(dist_top, dist_bottom), 
                                   np.minimum(dist_left, dist_right))
        
        # 归一化到[0, 1]
        normalized_dist = np.clip(dist_to_edge / feather_width, 0, 1)
        
        # 应用权重函数
        if weight_type == WeightType.LINEAR:
            weight = normalized_dist
        elif weight_type == WeightType.COSINE:
            # 余弦窗: 0.5 * (1 + cos(π * (1 - x))) = 0.5 * (1 - cos(π * x))
            weight = 0.5 * (1 - np.cos(np.pi * normalized_dist))
        elif weight_type == WeightType.SIGMOID:
            # S型过渡
            weight = 1 / (1 + np.exp(-10 * (normalized_dist - 0.5)))
        else:
            weight = normalized_dist
        
        return weight.astype(np.float32)
    
    def poisson_fusion(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        mask: Optional[np.ndarray] = None,
        center: Optional[Tuple[int, int]] = None,
        mode: PoissonMode = PoissonMode.NORMAL
    ) -> np.ndarray:
        """
        泊松融合（备选方案）
        
        使用OpenCV的seamlessClone实现梯度域融合，
        保持源图像的梯度信息同时匹配目标图像的边界。
        
        Args:
            src: 源图像（要融合的图像）
            dst: 目标图像（背景图像）
            mask: 掩码，指定源图像的有效区域
            center: 融合中心位置 (x, y)，在目标图像中的坐标
            mode: 泊松融合模式
        
        Returns:
            融合后的图像
        
        Example:
            >>> result = blender.poisson_fusion(
            ...     src=tile_image,
            ...     dst=background,
            ...     mask=mask,
            ...     center=(512, 512),
            ...     mode=PoissonMode.MIXED
            ... )
        
        Reference:
            Perez, P., Gangnet, M., & Blake, A. (2003). Poisson image editing.
            ACM TOG, 22(3), 313-318.
        """
        # 确保图像是uint8类型
        if src.dtype != np.uint8:
            src = np.clip(src, 0, 255).astype(np.uint8)
        if dst.dtype != np.uint8:
            dst = np.clip(dst, 0, 255).astype(np.uint8)
        
        # 创建默认掩码
        if mask is None:
            mask = np.ones(src.shape[:2], dtype=np.uint8) * 255
        elif mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        
        # 确定融合中心
        if center is None:
            # 默认放在目标图像中心
            center = (dst.shape[1] // 2, dst.shape[0] // 2)
        
        try:
            # 执行泊松融合
            result = cv2.seamlessClone(src, dst, mask, center, mode.value)
            return result
        except Exception as e:
            logger.error(f"Poisson fusion failed: {e}")
            # 失败时返回加权平均融合结果
            logger.warning("Falling back to weighted average fusion")
            return self._fallback_blend(src, dst, mask, center)
    
    def _fallback_blend(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        mask: np.ndarray,
        center: Tuple[int, int]
    ) -> np.ndarray:
        """泊松融合失败时的回退方案"""
        h, w = src.shape[:2]
        cx, cy = center
        
        # 计算放置位置
        x1 = max(0, cx - w // 2)
        y1 = max(0, cy - h // 2)
        x2 = min(dst.shape[1], x1 + w)
        y2 = min(dst.shape[0], y1 + h)
        
        # 调整源图像和掩码
        src_cropped = src[:y2-y1, :x2-x1]
        mask_cropped = mask[:y2-y1, :x2-x1]
        
        # 归一化掩码
        mask_norm = mask_cropped.astype(np.float32) / 255.0
        if len(src_cropped.shape) == 3:
            mask_norm = np.expand_dims(mask_norm, axis=-1)
        
        # 混合
        result = dst.copy()
        roi = result[y1:y2, x1:x2]
        blended = roi * (1 - mask_norm) + src_cropped * mask_norm
        result[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        return result
    
    def weighted_average_fusion(
        self,
        tiles: List[Union[np.ndarray, TileInfo]],
        weights: Optional[List[np.ndarray]] = None,
        weight_type: WeightType = WeightType.COSINE,
        output_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        加权平均融合（快速方案）
        
        简单的加权平均融合，计算速度快但可能产生接缝。
        适用于实时预览或快速原型。
        
        Args:
            tiles: 瓦片列表
            weights: 自定义权重图列表（可选）
            weight_type: 权重类型（线性、余弦）
            output_shape: 输出图像形状
        
        Returns:
            融合后的图像
        
        Example:
            >>> result = blender.weighted_average_fusion(
            ...     tiles=[tile1, tile2],
            ...     weight_type=WeightType.COSINE
            ... )
        """
        # 转换TileInfo
        tile_images = []
        tile_positions = []
        
        for i, tile in enumerate(tiles):
            if isinstance(tile, TileInfo):
                tile_images.append(tile.image)
                tile_positions.append((tile.y, tile.x))
            else:
                tile_images.append(tile)
                tile_h, tile_w = tile.shape[:2]
                grid_size = int(np.ceil(np.sqrt(len(tiles))))
                row, col = i // grid_size, i % grid_size
                tile_positions.append((row * tile_h, col * tile_w))
        
        # 确定输出尺寸
        if output_shape is None:
            max_h = max(pos[0] + img.shape[0] for img, pos in zip(tile_images, tile_positions))
            max_w = max(pos[1] + img.shape[1] for img, pos in zip(tile_images, tile_positions))
            output_shape = (max_h, max_w)
        
        output_h, output_w = output_shape[:2]
        num_channels = tile_images[0].shape[2] if len(tile_images[0].shape) == 3 else 1
        
        # 初始化累加器
        if num_channels == 1:
            weight_accumulator = np.zeros((output_h, output_w), dtype=np.float32)
            result_accumulator = np.zeros((output_h, output_w), dtype=np.float32)
        else:
            weight_accumulator = np.zeros((output_h, output_w), dtype=np.float32)
            result_accumulator = np.zeros((output_h, output_w, num_channels), dtype=np.float32)
        
        # 处理每个瓦片
        for i, (tile_img, (y, x)) in enumerate(zip(tile_images, tile_positions)):
            tile_h, tile_w = tile_img.shape[:2]
            
            if tile_img.dtype != np.float32:
                tile_img = tile_img.astype(np.float32)
            
            # 使用自定义权重或生成权重
            if weights is not None and i < len(weights):
                weight_map = weights[i]
            else:
                weight_map = self._create_distance_weight_map(
                    tile_h, tile_w, weight_type
                )
            
            # 放置到输出位置
            y_end = min(y + tile_h, output_h)
            x_end = min(x + tile_w, output_w)
            tile_h_actual = y_end - y
            tile_w_actual = x_end - x
            
            tile_cropped = tile_img[:tile_h_actual, :tile_w_actual]
            weight_cropped = weight_map[:tile_h_actual, :tile_w_actual]
            
            # 扩展权重维度
            if num_channels > 1 and len(weight_cropped.shape) == 2:
                weight_cropped = np.expand_dims(weight_cropped, axis=-1)
            
            # 累加
            result_accumulator[y:y_end, x:x_end] += tile_cropped * weight_cropped
            weight_accumulator[y:y_end, x:x_end] += weight_cropped.squeeze() if num_channels > 1 else weight_cropped
        
        # 归一化
        weight_accumulator = np.maximum(weight_accumulator, 1e-6)
        if num_channels > 1:
            result = result_accumulator / np.expand_dims(weight_accumulator, axis=-1)
        else:
            result = result_accumulator / weight_accumulator
        
        return np.clip(result, 0, 255).astype(np.uint8)


    # ==================== 质量控制和后处理 ====================
    
    def detect_seams(
        self,
        result: np.ndarray,
        tiles: List[Union[np.ndarray, TileInfo]],
        window_size: int = 16,
        stride: int = 8
    ) -> List[Seam]:
        """
        接缝检测（基于SSIM局部计算）
        
        通过计算融合结果与原始瓦片之间的局部SSIM，
        识别融合质量较差的区域。
        
        Args:
            result: 融合结果图像
            tiles: 原始瓦片列表
            window_size: SSIM计算窗口大小（默认16x16）
            stride: 滑动窗口步长
        
        Returns:
            检测到的接缝列表
        
        Example:
            >>> seams = blender.detect_seams(result, tiles, window_size=16)
            >>> print(f"Detected {len(seams)} seams")
            >>> for seam in seams:
            ...     print(f"Seam at ({seam.x}, {seam.y}): SSIM={seam.ssim_score:.3f}")
        """
        seams = []
        
        # 转换TileInfo
        tile_infos = []
        for tile in tiles:
            if isinstance(tile, TileInfo):
                tile_infos.append(tile)
            else:
                tile_infos.append(TileInfo(tile, 0, 0, 0, 0))
        
        # 对每个瓦片检测接缝
        for tile_info in tile_infos:
            tile_img = tile_info.image
            tx, ty = tile_info.x, tile_info.y
            tile_h, tile_w = tile_img.shape[:2]
            
            # 提取结果中对应区域
            rx1, ry1 = tx, ty
            rx2, ry2 = min(tx + tile_w, result.shape[1]), min(ty + tile_h, result.shape[0])
            
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            
            result_roi = result[ry1:ry2, rx1:rx2]
            tile_roi = tile_img[:ry2-ry1, :rx2-rx1]
            
            # 确保尺寸一致
            if result_roi.shape != tile_roi.shape:
                continue
            
            # 滑动窗口计算SSIM
            roi_h, roi_w = result_roi.shape[:2]
            
            for y in range(0, roi_h - window_size + 1, stride):
                for x in range(0, roi_w - window_size + 1, stride):
                    # 提取窗口
                    result_window = result_roi[y:y+window_size, x:x+window_size]
                    tile_window = tile_roi[y:y+window_size, x:x+window_size]
                    
                    # 计算SSIM
                    ssim_score = self._compute_ssim(tile_window, result_window)
                    
                    # 如果SSIM低于阈值，记录为接缝
                    if ssim_score < self.ssim_threshold:
                        global_x = tx + x
                        global_y = ty + y
                        
                        seam = Seam(
                            x=global_x,
                            y=global_y,
                            width=window_size,
                            height=window_size,
                            ssim_score=ssim_score
                        )
                        seams.append(seam)
        
        # 合并相邻的接缝
        seams = self._merge_adjacent_seams(seams, distance_threshold=window_size)
        
        logger.info(f"Detected {len(seams)} seams")
        return seams
    
    def _compute_ssim(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        k1: float = 0.01,
        k2: float = 0.03,
        L: float = 255.0
    ) -> float:
        """
        计算两幅图像的SSIM（结构相似性指数）
        
        Args:
            img1, img2: 输入图像
            k1, k2: 稳定性常数
            L: 像素值动态范围
        
        Returns:
            SSIM值，范围[0, 1]，越高表示越相似
        """
        # 转换为灰度图（如果是彩色）
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 确保float类型
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # 计算均值
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        # 计算方差和协方差
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        # SSIM常数
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        
        # 计算SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim = numerator / denominator
        
        return float(ssim)
    
    def _merge_adjacent_seams(
        self,
        seams: List[Seam],
        distance_threshold: int = 16
    ) -> List[Seam]:
        """
        合并相邻的接缝
        
        Args:
            seams: 接缝列表
            distance_threshold: 合并距离阈值
        
        Returns:
            合并后的接缝列表
        """
        if not seams:
            return []
        
        # 按位置排序
        seams_sorted = sorted(seams, key=lambda s: (s.y, s.x))
        
        merged = []
        current_group = [seams_sorted[0]]
        
        for seam in seams_sorted[1:]:
            # 检查是否与当前组相邻
            last_seam = current_group[-1]
            distance = np.sqrt((seam.x - last_seam.x)**2 + (seam.y - last_seam.y)**2)
            
            if distance < distance_threshold:
                current_group.append(seam)
            else:
                # 合并当前组
                merged.append(self._merge_seam_group(current_group))
                current_group = [seam]
        
        # 合并最后一组
        if current_group:
            merged.append(self._merge_seam_group(current_group))
        
        return merged
    
    def _merge_seam_group(self, seam_group: List[Seam]) -> Seam:
        """合并一组接缝"""
        if len(seam_group) == 1:
            return seam_group[0]
        
        # 计算包围盒
        x_min = min(s.x for s in seam_group)
        y_min = min(s.y for s in seam_group)
        x_max = max(s.x + s.width for s in seam_group)
        y_max = max(s.y + s.height for s in seam_group)
        
        # 计算平均SSIM
        avg_ssim = np.mean([s.ssim_score for s in seam_group])
        
        return Seam(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min,
            ssim_score=avg_ssim
        )
    
    def color_correction(
        self,
        image: np.ndarray,
        reference_tile: np.ndarray,
        method: str = "histogram",
        local_filter: bool = True
    ) -> np.ndarray:
        """
        色彩一致性校正
        
        通过直方图匹配和引导滤波校正图像色彩，
        使其与参考瓦片保持一致。
        
        Args:
            image: 待校正图像
            reference_tile: 参考瓦片
            method: 校正方法 ('histogram', 'mean_std', 'none')
            local_filter: 是否应用局部引导滤波
        
        Returns:
            校正后的图像
        
        Example:
            >>> corrected = blender.color_correction(
            ...     image=fused_result,
            ...     reference_tile=tiles[0],
            ...     method="histogram",
            ...     local_filter=True
            ... )
        """
        if method == "none":
            return image
        
        # 确保相同数据类型
        image_float = image.astype(np.float32)
        reference_float = reference_tile.astype(np.float32)
        
        if method == "histogram":
            corrected = self._histogram_matching(image_float, reference_float)
        elif method == "mean_std":
            corrected = self._mean_std_matching(image_float, reference_float)
        else:
            corrected = image_float
        
        # 应用局部引导滤波
        if local_filter:
            corrected = self._guided_filter(corrected, image_float, radius=8, eps=0.01)
        
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def _histogram_matching(
        self,
        source: np.ndarray,
        reference: np.ndarray
    ) -> np.ndarray:
        """
        直方图匹配
        
        将源图像的直方图匹配到参考图像。
        """
        result = source.copy()
        
        # 对每个通道分别处理
        num_channels = source.shape[2] if len(source.shape) == 3 else 1
        
        for c in range(num_channels):
            if num_channels == 1:
                src_channel = source
                ref_channel = reference
            else:
                src_channel = source[:, :, c]
                ref_channel = reference[:, :, c]
            
            # 计算直方图
            src_hist, _ = np.histogram(src_channel.flatten(), 256, [0, 256])
            ref_hist, _ = np.histogram(ref_channel.flatten(), 256, [0, 256])
            
            # 计算CDF
            src_cdf = src_hist.cumsum()
            ref_cdf = ref_hist.cumsum()
            
            # 归一化
            src_cdf = (src_cdf / src_cdf[-1]) * 255
            ref_cdf = (ref_cdf / ref_cdf[-1]) * 255
            
            # 创建查找表
            lookup_table = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                # 找到参考CDF中最接近的值
                lookup_table[i] = np.argmin(np.abs(ref_cdf - src_cdf[i]))
            
            # 应用查找表
            if num_channels == 1:
                result = lookup_table[src_channel.astype(np.uint8)]
            else:
                result[:, :, c] = lookup_table[src_channel.astype(np.uint8)]
        
        return result.astype(np.float32)
    
    def _mean_std_matching(
        self,
        source: np.ndarray,
        reference: np.ndarray
    ) -> np.ndarray:
        """
        均值-标准差匹配
        
        简单快速的色彩匹配方法。
        """
        # 计算均值和标准差
        src_mean = np.mean(source, axis=(0, 1))
        src_std = np.std(source, axis=(0, 1))
        ref_mean = np.mean(reference, axis=(0, 1))
        ref_std = np.std(reference, axis=(0, 1))
        
        # 匹配
        if len(source.shape) == 3:
            result = (source - src_mean) * (ref_std / (src_std + 1e-6)) + ref_mean
        else:
            result = (source - src_mean) * (ref_std / (src_std + 1e-6)) + ref_mean
        
        return result
    
    def _guided_filter(
        self,
        guide: np.ndarray,
        src: np.ndarray,
        radius: int = 8,
        eps: float = 0.01
    ) -> np.ndarray:
        """
        引导滤波
        
        边缘保持的平滑滤波器，用于局部色彩校正。
        
        Reference:
            He, K., Sun, J., & Tang, X. (2013). Guided image filtering.
            IEEE TPAMI, 35(6), 1397-1409.
        """
        # 使用OpenCV的ximgproc模块（如果可用）
        try:
            import cv2.ximgproc as ximgproc
            return ximgproc.guidedFilter(guide, src, radius, eps)
        except:
            # 简化实现
            return self._simple_guided_filter(guide, src, radius, eps)
    
    def _simple_guided_filter(
        self,
        guide: np.ndarray,
        src: np.ndarray,
        radius: int,
        eps: float
    ) -> np.ndarray:
        """简化的引导滤波实现"""
        # 均值滤波
        mean_guide = cv2.blur(guide, (radius, radius))
        mean_src = cv2.blur(src, (radius, radius))
        
        # 计算协方差和方差
        mean_guide_src = cv2.blur(guide * src, (radius, radius))
        mean_guide_sq = cv2.blur(guide * guide, (radius, radius))
        
        cov = mean_guide_src - mean_guide * mean_src
        var = mean_guide_sq - mean_guide * mean_guide
        
        # 计算线性系数
        a = cov / (var + eps)
        b = mean_src - a * mean_guide
        
        # 平滑系数
        mean_a = cv2.blur(a, (radius, radius))
        mean_b = cv2.blur(b, (radius, radius))
        
        # 输出
        result = mean_a * guide + mean_b
        
        return result
    
    def repair_seams(
        self,
        image: np.ndarray,
        seams: List[Seam],
        tiles: List[np.ndarray],
        repair_method: str = "auto"
    ) -> np.ndarray:
        """
        接缝修复
        
        根据接缝严重程度选择合适的修复方法。
        
        Args:
            image: 融合结果图像
            seams: 接缝列表
            tiles: 原始瓦片
            repair_method: 修复方法 ('auto', 'blend', 'poisson')
        
        Returns:
            修复后的图像
        """
        result = image.copy()
        
        for seam in seams:
            if repair_method == "auto":
                method = seam.suggested_fix
            else:
                method = repair_method
            
            if method == "none":
                continue
            
            # 提取接缝区域
            x1, y1 = seam.x, seam.y
            x2, y2 = x1 + seam.width, y1 + seam.height
            
            # 扩展修复区域
            padding = max(seam.width, seam.height)
            x1_p = max(0, x1 - padding)
            y1_p = max(0, y1 - padding)
            x2_p = min(image.shape[1], x2 + padding)
            y2_p = min(image.shape[0], y2 + padding)
            
            roi = result[y1_p:y2_p, x1_p:x2_p].copy()
            
            if method == "increase_blend_width":
                # 增大融合宽度 - 使用高斯模糊平滑
                roi = cv2.GaussianBlur(roi, (15, 15), 0)
            elif method == "poisson_refinement":
                # 泊松精修
                # 创建掩码
                mask = np.zeros((y2_p-y1_p, x2_p-x1_p), dtype=np.uint8)
                mask[y1-y1_p:y2-y1_p, x1-x1_p:x2-x1_p] = 255
                
                # 找到最佳匹配瓦片
                best_tile = self._find_best_matching_tile(roi, tiles)
                
                # 泊松融合
                roi = self.poisson_fusion(
                    best_tile[:y2_p-y1_p, :x2_p-x1_p],
                    roi,
                    mask,
                    center=((x2_p-x1_p)//2, (y2_p-y1_p)//2),
                    mode=PoissonMode.MIXED
                )
            
            result[y1_p:y2_p, x1_p:x2_p] = roi
        
        return result
    
    def _find_best_matching_tile(
        self,
        region: np.ndarray,
        tiles: List[np.ndarray]
    ) -> np.ndarray:
        """找到与区域最匹配的瓦片"""
        best_score = -1
        best_tile = tiles[0]
        
        region_h, region_w = region.shape[:2]
        
        for tile in tiles:
            # 调整瓦片尺寸
            tile_resized = cv2.resize(tile, (region_w, region_h))
            
            # 计算相似度
            score = self._compute_ssim(region, tile_resized)
            
            if score > best_score:
                best_score = score
                best_tile = tile
        
        return cv2.resize(best_tile, (region_w, region_h))


    # ==================== 高级融合功能 ====================
    
    def multi_band_fusion(
        self,
        tiles: List[Union[np.ndarray, TileInfo]],
        num_bands: int = 6,
        output_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        多频段融合（Multi-band Blending）
        
        拉普拉斯金字塔融合的变体，分别处理不同频率成分。
        适用于大视差图像融合。
        
        Args:
            tiles: 瓦片列表
            num_bands: 频段数量
            output_shape: 输出图像形状
        
        Returns:
            融合后的图像
        """
        # 这与laplacian_fusion类似，但允许更精细的频率控制
        return self.laplacian_fusion(
            tiles,
            output_shape=output_shape,
            weight_type=WeightType.SIGMOID
        )
    
    def feather_blend(
        self,
        tiles: List[Union[np.ndarray, TileInfo]],
        feather_width: int = 50,
        output_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        羽化融合
        
        简单的羽化边缘融合，适用于小重叠区域。
        
        Args:
            tiles: 瓦片列表
            feather_width: 羽化宽度（像素）
            output_shape: 输出图像形状
        
        Returns:
            融合后的图像
        """
        # 转换TileInfo
        tile_images = []
        tile_positions = []
        
        for i, tile in enumerate(tiles):
            if isinstance(tile, TileInfo):
                tile_images.append(tile.image)
                tile_positions.append((tile.y, tile.x))
            else:
                tile_images.append(tile)
                tile_h, tile_w = tile.shape[:2]
                grid_size = int(np.ceil(np.sqrt(len(tiles))))
                row, col = i // grid_size, i % grid_size
                tile_positions.append((row * tile_h, col * tile_w))
        
        # 确定输出尺寸
        if output_shape is None:
            max_h = max(pos[0] + img.shape[0] for img, pos in zip(tile_images, tile_positions))
            max_w = max(pos[1] + img.shape[1] for img, pos in zip(tile_images, tile_positions))
            output_shape = (max_h, max_w)
        
        output_h, output_w = output_shape[:2]
        num_channels = tile_images[0].shape[2] if len(tile_images[0].shape) == 3 else 1
        
        # 创建距离变换权重
        weight_maps = []
        for tile_img, (y, x) in zip(tile_images, tile_positions):
            tile_h, tile_w = tile_img.shape[:2]
            
            # 创建二值掩码
            mask = np.ones((tile_h, tile_w), dtype=np.float32)
            
            # 距离变换
            dist_transform = cv2.distanceTransform(
                mask.astype(np.uint8),
                cv2.DIST_L2,
                5
            )
            
            # 归一化并应用羽化
            max_dist = np.max(dist_transform)
            if max_dist > 0:
                weight = dist_transform / max_dist
                # 应用余弦过渡
                weight = 0.5 * (1 - np.cos(np.pi * weight))
            else:
                weight = mask
            
            weight_maps.append(weight)
        
        # 加权融合
        if num_channels == 1:
            weight_accumulator = np.zeros((output_h, output_w), dtype=np.float32)
            result_accumulator = np.zeros((output_h, output_w), dtype=np.float32)
        else:
            weight_accumulator = np.zeros((output_h, output_w), dtype=np.float32)
            result_accumulator = np.zeros((output_h, output_w, num_channels), dtype=np.float32)
        
        for tile_img, (y, x), weight_map in zip(tile_images, tile_positions, weight_maps):
            tile_h, tile_w = tile_img.shape[:2]
            
            if tile_img.dtype != np.float32:
                tile_img = tile_img.astype(np.float32)
            
            y_end = min(y + tile_h, output_h)
            x_end = min(x + tile_w, output_w)
            tile_h_actual = y_end - y
            tile_w_actual = x_end - x
            
            tile_cropped = tile_img[:tile_h_actual, :tile_w_actual]
            weight_cropped = weight_map[:tile_h_actual, :tile_w_actual]
            
            if num_channels > 1 and len(weight_cropped.shape) == 2:
                weight_cropped = np.expand_dims(weight_cropped, axis=-1)
            
            result_accumulator[y:y_end, x:x_end] += tile_cropped * weight_cropped
            weight_accumulator[y:y_end, x:x_end] += weight_cropped.squeeze() if num_channels > 1 else weight_cropped
        
        weight_accumulator = np.maximum(weight_accumulator, 1e-6)
        if num_channels > 1:
            result = result_accumulator / np.expand_dims(weight_accumulator, axis=-1)
        else:
            result = result_accumulator / weight_accumulator
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def gradient_domain_fusion(
        self,
        tiles: List[np.ndarray],
        positions: List[Tuple[int, int]],
        output_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        梯度域融合
        
        在梯度域进行融合，保持边缘清晰度。
        适用于需要保持细节的场景。
        
        Args:
            tiles: 瓦片列表
            positions: 每个瓦片的位置列表
            output_shape: 输出形状 (height, width)
        
        Returns:
            融合后的图像
        """
        output_h, output_w = output_shape
        num_channels = tiles[0].shape[2] if len(tiles[0].shape) == 3 else 1
        
        # 初始化梯度累加器
        if num_channels == 1:
            grad_x = np.zeros((output_h, output_w), dtype=np.float32)
            grad_y = np.zeros((output_h, output_w), dtype=np.float32)
            weight_acc = np.zeros((output_h, output_w), dtype=np.float32)
        else:
            grad_x = np.zeros((output_h, output_w, num_channels), dtype=np.float32)
            grad_y = np.zeros((output_h, output_w, num_channels), dtype=np.float32)
            weight_acc = np.zeros((output_h, output_w), dtype=np.float32)
        
        # 计算每个瓦片的梯度并累加
        for tile, (y, x) in zip(tiles, positions):
            if tile.dtype != np.float32:
                tile = tile.astype(np.float32)
            
            tile_h, tile_w = tile.shape[:2]
            
            # 计算梯度
            if num_channels == 1:
                gx = cv2.Sobel(tile, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(tile, cv2.CV_32F, 0, 1, ksize=3)
            else:
                gx = np.zeros_like(tile)
                gy = np.zeros_like(tile)
                for c in range(num_channels):
                    gx[:, :, c] = cv2.Sobel(tile[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
                    gy[:, :, c] = cv2.Sobel(tile[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
            
            # 生成权重
            weight = self._create_distance_weight_map(tile_h, tile_w, WeightType.COSINE)
            
            # 放置到输出位置
            y_end = min(y + tile_h, output_h)
            x_end = min(x + tile_w, output_w)
            
            gx_cropped = gx[:y_end-y, :x_end-x]
            gy_cropped = gy[:y_end-y, :x_end-x]
            weight_cropped = weight[:y_end-y, :x_end-x]
            
            if num_channels > 1:
                weight_cropped = np.expand_dims(weight_cropped, axis=-1)
            
            grad_x[y:y_end, x:x_end] += gx_cropped * weight_cropped
            grad_y[y:y_end, x:x_end] += gy_cropped * weight_cropped
            weight_acc[y:y_end, x:x_end] += weight_cropped.squeeze() if num_channels > 1 else weight_cropped
        
        # 归一化梯度
        weight_acc = np.maximum(weight_acc, 1e-6)
        if num_channels > 1:
            grad_x /= np.expand_dims(weight_acc, axis=-1)
            grad_y /= np.expand_dims(weight_acc, axis=-1)
        else:
            grad_x /= weight_acc
            grad_y /= weight_acc
        
        # 从梯度重建图像（简化实现）
        # 使用泊松重建或积分
        result = self._reconstruct_from_gradients(grad_x, grad_y, output_shape)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _reconstruct_from_gradients(
        self,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        output_shape: Tuple[int, int]
    ) -> np.ndarray:
        """从梯度重建图像"""
        # 简化的积分方法
        # 实际应用中可能需要更复杂的泊松求解器
        
        num_channels = grad_x.shape[2] if len(grad_x.shape) == 3 else 1
        
        if num_channels == 1:
            result = np.zeros(output_shape, dtype=np.float32)
            # 沿x方向积分
            result[:, :] = np.cumsum(grad_x, axis=1)
            # 沿y方向添加
            result[:, :] += np.cumsum(grad_y, axis=0)
            result[:, :] /= 2
        else:
            result = np.zeros((output_shape[0], output_shape[1], num_channels), dtype=np.float32)
            for c in range(num_channels):
                result[:, :, c] = np.cumsum(grad_x[:, :, c], axis=1)
                result[:, :, c] += np.cumsum(grad_y[:, :, c], axis=0)
                result[:, :, c] /= 2
        
        return result


# ==================== 工具函数 ====================

def create_tile_grid(
    images: List[np.ndarray],
    grid_shape: Tuple[int, int],
    overlap: int = 100
) -> Tuple[List[TileInfo], List[OverlapRegion]]:
    """
    创建瓦片网格
    
    Args:
        images: 瓦片图像列表
        grid_shape: 网格形状 (rows, cols)
        overlap: 重叠像素数
    
    Returns:
        (瓦片信息列表, 重叠区域列表)
    """
    rows, cols = grid_shape
    tile_infos = []
    
    tile_h, tile_w = images[0].shape[:2]
    
    # 创建瓦片信息
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        # 计算位置（考虑重叠）
        x = col * (tile_w - overlap)
        y = row * (tile_h - overlap)
        
        tile_infos.append(TileInfo(img, x, y, row, col))
    
    # 计算重叠区域
    overlap_regions = []
    for i, tile1 in enumerate(tile_infos):
        for j, tile2 in enumerate(tile_infos[i+1:], i+1):
            # 检查是否相邻
            if abs(tile1.row - tile2.row) + abs(tile1.col - tile2.col) != 1:
                continue
            
            # 计算重叠
            x1_min, y1_min = tile1.x, tile1.y
            x1_max, y1_max = tile1.x + tile1.image.shape[1], tile1.y + tile1.image.shape[0]
            x2_min, y2_min = tile2.x, tile2.y
            x2_max, y2_max = tile2.x + tile2.image.shape[1], tile2.y + tile2.image.shape[0]
            
            # 计算交集
            x_min = max(x1_min, x2_min)
            y_min = max(y1_min, y2_min)
            x_max = min(x1_max, x2_max)
            y_max = min(y1_max, y2_max)
            
            if x_max > x_min and y_max > y_min:
                direction = 'horizontal' if tile1.row == tile2.row else 'vertical'
                
                overlap_region = OverlapRegion(
                    tile1_idx=i,
                    tile2_idx=j,
                    x1_start=x_min - x1_min,
                    y1_start=y_min - y1_min,
                    x2_start=x_min - x2_min,
                    y2_start=y_min - y2_min,
                    width=x_max - x_min,
                    height=y_max - y_min,
                    direction=direction
                )
                overlap_regions.append(overlap_region)
    
    return tile_infos, overlap_regions


def compute_blend_quality(
    result: np.ndarray,
    tiles: List[np.ndarray],
    positions: List[Tuple[int, int]]
) -> Dict[str, float]:
    """
    计算融合质量指标
    
    Args:
        result: 融合结果
        tiles: 原始瓦片
        positions: 瓦片位置
    
    Returns:
        质量指标字典
    """
    metrics = {}
    
    # 计算平均SSIM
    ssim_scores = []
    for tile, (y, x) in zip(tiles, positions):
        h, w = tile.shape[:2]
        roi = result[y:y+h, x:x+w]
        
        # 调整尺寸
        if roi.shape != tile.shape:
            tile = cv2.resize(tile, (roi.shape[1], roi.shape[0]))
        
        # 计算SSIM
        module = BlendingModule()
        ssim = module._compute_ssim(roi, tile)
        ssim_scores.append(ssim)
    
    metrics['mean_ssim'] = np.mean(ssim_scores)
    metrics['min_ssim'] = np.min(ssim_scores)
    metrics['std_ssim'] = np.std(ssim_scores)
    
    # 计算梯度连续性
    grad_x = cv2.Sobel(result, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(result, cv2.CV_32F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    metrics['mean_gradient'] = np.mean(grad_magnitude)
    metrics['gradient_discontinuity'] = np.std(grad_magnitude)
    
    return metrics


def visualize_seams(
    image: np.ndarray,
    seams: List[Seam],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    可视化接缝
    
    Args:
        image: 输入图像
        seams: 接缝列表
        color: 标记颜色 (B, G, R)
        thickness: 线宽
    
    Returns:
        带接缝标记的图像
    """
    result = image.copy()
    
    for seam in seams:
        # 根据严重程度选择颜色
        if seam.severity == 'high':
            c = (0, 0, 255)  # 红色
        elif seam.severity == 'medium':
            c = (0, 255, 255)  # 黄色
        else:
            c = (0, 255, 0)  # 绿色
        
        # 绘制矩形
        cv2.rectangle(
            result,
            (seam.x, seam.y),
            (seam.x + seam.width, seam.y + seam.height),
            c,
            thickness
        )
        
        # 添加SSIM分数
        cv2.putText(
            result,
            f"{seam.ssim_score:.3f}",
            (seam.x, seam.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            c,
            1
        )
    
    return result


# ==================== 性能优化工具 ====================

class ParallelBlender:
    """并行融合器"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
    
    def blend_tiles_parallel(
        self,
        blender: BlendingModule,
        tile_groups: List[List[TileInfo]],
        output_shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """
        并行融合多组瓦片
        
        Args:
            blender: 融合模块实例
            tile_groups: 瓦片组列表
            output_shape: 输出形状
        
        Returns:
            融合结果列表
        """
        futures = []
        
        for tiles in tile_groups:
            future = self.executor.submit(
                blender.laplacian_fusion,
                tiles,
                None,
                output_shape
            )
            futures.append(future)
        
        results = [f.result() for f in futures]
        return results
    
    def close(self):
        """关闭线程池"""
        self.executor.shutdown()


# ==================== CUDA加速工具 ====================

class CUDABlending:
    """CUDA加速融合工具"""
    
    def __init__(self):
        self.available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.available:
            cv2.cuda.setDevice(0)
    
    def gpu_pyrdown(self, image: np.ndarray) -> np.ndarray:
        """GPU加速下采样"""
        if not self.available:
            return cv2.pyrDown(image)
        
        gpu_img = cv2.cuda.GpuMat()
        gpu_img.upload(image)
        gpu_result = cv2.cuda.pyrDown(gpu_img)
        return gpu_result.download()
    
    def gpu_pyrup(self, image: np.ndarray, dstsize: Tuple[int, int]) -> np.ndarray:
        """GPU加速上采样"""
        if not self.available:
            return cv2.pyrUp(image, dstsize=dstsize)
        
        gpu_img = cv2.cuda.GpuMat()
        gpu_img.upload(image)
        gpu_result = cv2.cuda.pyrUp(gpu_img, dstsize=dstsize)
        return gpu_result.download()
    
    def gpu_add(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """GPU加速加法"""
        if not self.available:
            return cv2.add(img1, img2)
        
        gpu1 = cv2.cuda.GpuMat()
        gpu2 = cv2.cuda.GpuMat()
        gpu1.upload(img1)
        gpu2.upload(img2)
        gpu_result = cv2.cuda.add(gpu1, gpu2)
        return gpu_result.download()
    
    def gpu_multiply(self, img: np.ndarray, scalar: float) -> np.ndarray:
        """GPU加速乘法"""
        if not self.available:
            return img * scalar
        
        gpu_img = cv2.cuda.GpuMat()
        gpu_img.upload(img)
        gpu_result = cv2.cuda.multiply(gpu_img, scalar)
        return gpu_result.download()


# ==================== 示例和测试代码 ====================

def example_laplacian_fusion():
    """
    拉普拉斯金字塔融合示例
    
    演示如何使用拉普拉斯金字塔融合算法融合2x2网格的瓦片。
    """
    print("=" * 60)
    print("示例1: 拉普拉斯金字塔融合")
    print("=" * 60)
    
    # 创建测试瓦片（模拟2x2网格）
    tile_size = 512
    overlap = 100
    
    # 创建渐变测试图像
    tiles = []
    for i in range(4):
        # 创建带有不同颜色的测试瓦片
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        # 添加渐变
        for y in range(tile_size):
            for x in range(tile_size):
                tile[y, x] = [
                    int(128 + 127 * np.sin(x / 50.0 + i)),
                    int(128 + 127 * np.sin(y / 50.0 + i * 0.5)),
                    int(128 + 127 * np.cos((x + y) / 100.0))
                ]
        
        # 添加噪声模拟真实场景
        noise = np.random.normal(0, 5, tile.shape).astype(np.int16)
        tile = np.clip(tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        tiles.append(tile)
    
    # 创建TileInfo
    tile_infos = []
    positions = [(0, 0), (0, tile_size - overlap), 
                 (tile_size - overlap, 0), (tile_size - overlap, tile_size - overlap)]
    
    for i, (tile, (y, x)) in enumerate(zip(tiles, positions)):
        tile_infos.append(TileInfo(tile, x, y, i // 2, i % 2))
    
    # 创建融合模块
    blender = BlendingModule(method='laplacian', num_levels=6)
    
    # 执行融合
    output_shape = (tile_size * 2 - overlap, tile_size * 2 - overlap)
    
    import time
    start_time = time.time()
    result = blender.laplacian_fusion(tile_infos, output_shape=output_shape)
    elapsed = time.time() - start_time
    
    print(f"融合完成！")
    print(f"  输出尺寸: {result.shape}")
    print(f"  融合时间: {elapsed:.3f}秒")
    
    # 检测接缝
    seams = blender.detect_seams(result, tile_infos)
    print(f"  检测到接缝: {len(seams)}个")
    
    for seam in seams[:5]:  # 只显示前5个
        print(f"    - 位置({seam.x}, {seam.y}), SSIM={seam.ssim_score:.3f}, "
              f"严重程度={seam.severity}")
    
    return result, seams


def example_poisson_fusion():
    """
    泊松融合示例
    
    演示如何使用泊松融合将对象无缝插入背景。
    """
    print("\n" + "=" * 60)
    print("示例2: 泊松融合")
    print("=" * 60)
    
    # 创建背景图像
    background = np.zeros((800, 800, 3), dtype=np.uint8)
    background[:, :] = [100, 150, 200]  # 蓝色背景
    
    # 创建源图像（要插入的对象）
    src_size = 400
    src = np.zeros((src_size, src_size, 3), dtype=np.uint8)
    
    # 创建一个圆形渐变
    center = src_size // 2
    for y in range(src_size):
        for x in range(src_size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if dist < center:
                intensity = int(255 * (1 - dist / center))
                src[y, x] = [intensity, intensity // 2, 255 - intensity]
    
    # 创建圆形掩码
    mask = np.zeros((src_size, src_size), dtype=np.uint8)
    cv2.circle(mask, (center, center), center - 10, 255, -1)
    
    # 创建融合模块
    blender = BlendingModule()
    
    # 执行泊松融合
    import time
    start_time = time.time()
    
    result = blender.poisson_fusion(
        src=src,
        dst=background,
        mask=mask,
        center=(400, 400),
        mode=PoissonMode.NORMAL
    )
    
    elapsed = time.time() - start_time
    
    print(f"泊松融合完成！")
    print(f"  输出尺寸: {result.shape}")
    print(f"  融合时间: {elapsed:.3f}秒")
    
    return result


def example_weighted_average():
    """
    加权平均融合示例
    
    演示快速加权平均融合方法。
    """
    print("\n" + "=" * 60)
    print("示例3: 加权平均融合")
    print("=" * 60)
    
    # 创建测试瓦片
    tile_size = 400
    overlap = 80
    
    tiles = []
    for i in range(2):
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        # 创建不同颜色的瓦片
        color = [255, 0, 0] if i == 0 else [0, 0, 255]  # 红色和蓝色
        tile[:, :] = color
        
        # 添加渐变边缘
        for x in range(overlap):
            alpha = x / overlap
            if i == 0:
                tile[:, tile_size - overlap + x] = [
                    int(255 * (1 - alpha)),
                    0,
                    int(255 * alpha)
                ]
        
        tiles.append(tile)
    
    # 创建TileInfo
    tile_infos = [
        TileInfo(tiles[0], 0, 0, 0, 0),
        TileInfo(tiles[1], tile_size - overlap, 0, 0, 1)
    ]
    
    # 创建融合模块
    blender = BlendingModule()
    
    # 测试不同权重类型
    weight_types = [WeightType.LINEAR, WeightType.COSINE, WeightType.SIGMOID]
    
    results = {}
    for wt in weight_types:
        import time
        start_time = time.time()
        
        result = blender.weighted_average_fusion(
            tile_infos,
            weight_type=wt,
            output_shape=(tile_size, tile_size * 2 - overlap)
        )
        
        elapsed = time.time() - start_time
        results[wt.value] = result
        
        print(f"  {wt.value}: {elapsed*1000:.2f}ms")
    
    return results


def example_color_correction():
    """
    色彩校正示例
    
    演示如何校正图像色彩以匹配参考瓦片。
    """
    print("\n" + "=" * 60)
    print("示例4: 色彩校正")
    print("=" * 60)
    
    # 创建参考图像
    reference = np.zeros((400, 400, 3), dtype=np.uint8)
    reference[:, :, 0] = 128  # B通道
    reference[:, :, 1] = 180  # G通道
    reference[:, :, 2] = 220  # R通道
    
    # 创建偏色的待校正图像
    image = reference.copy().astype(np.float32)
    image[:, :, 0] *= 0.7  # B通道偏暗
    image[:, :, 1] *= 1.2  # G通道偏亮
    image[:, :, 2] *= 0.9  # R通道略暗
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # 创建融合模块
    blender = BlendingModule()
    
    # 执行色彩校正
    import time
    start_time = time.time()
    
    corrected = blender.color_correction(
        image=image,
        reference_tile=reference,
        method="histogram",
        local_filter=True
    )
    
    elapsed = time.time() - start_time
    
    print(f"色彩校正完成！")
    print(f"  校正时间: {elapsed*1000:.2f}ms")
    
    # 计算校正前后的差异
    diff_before = np.mean(np.abs(image.astype(float) - reference.astype(float)))
    diff_after = np.mean(np.abs(corrected.astype(float) - reference.astype(float)))
    
    print(f"  校正前平均差异: {diff_before:.2f}")
    print(f"  校正后平均差异: {diff_after:.2f}")
    print(f"  改善比例: {(1 - diff_after/diff_before)*100:.1f}%")
    
    return corrected


def example_seam_detection():
    """
    接缝检测示例
    
    演示如何检测和修复融合结果中的接缝。
    """
    print("\n" + "=" * 60)
    print("示例5: 接缝检测与修复")
    print("=" * 60)
    
    # 创建带有明显接缝的测试图像
    tile_size = 300
    
    # 创建两个有明显差异的瓦片
    tile1 = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 200
    tile2 = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 150
    
    # 添加一些纹理
    tile1[50:100, 50:100] = [255, 0, 0]
    tile2[50:100, 50:100] = [0, 0, 255]
    
    # 简单拼接（会产生明显接缝）
    result = np.zeros((tile_size, tile_size * 2), dtype=np.uint8)
    result[:, :tile_size] = cv2.cvtColor(tile1, cv2.COLOR_BGR2GRAY)
    result[:, tile_size:] = cv2.cvtColor(tile2, cv2.COLOR_BGR2GRAY)
    
    # 转回3通道
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # 创建瓦片信息
    tile_infos = [
        TileInfo(tile1, 0, 0, 0, 0),
        TileInfo(tile2, tile_size, 0, 0, 1)
    ]
    
    # 创建融合模块
    blender = BlendingModule(ssim_threshold=0.95)
    
    # 检测接缝
    seams = blender.detect_seams(result, tile_infos, window_size=16, stride=8)
    
    print(f"检测到 {len(seams)} 个接缝:")
    
    severity_count = {'low': 0, 'medium': 0, 'high': 0}
    for seam in seams:
        severity_count[seam.severity] += 1
    
    print(f"  轻度: {severity_count['low']}")
    print(f"  中度: {severity_count['medium']}")
    print(f"  重度: {severity_count['high']}")
    
    # 可视化接缝
    result_with_seams = visualize_seams(result, seams)
    
    return result_with_seams, seams


def example_compare_methods():
    """
    融合方法对比示例
    
    对比不同融合方法的性能和质量。
    """
    print("\n" + "=" * 60)
    print("示例6: 融合方法对比")
    print("=" * 60)
    
    # 创建测试数据
    tile_size = 400
    overlap = 100
    
    tiles = []
    for i in range(4):
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        # 创建彩色渐变
        for y in range(tile_size):
            for x in range(tile_size):
                tile[y, x] = [
                    int(128 + 100 * np.sin(x / 40.0 + i * 0.5)),
                    int(128 + 100 * np.sin(y / 40.0 + i * 0.3)),
                    int(128 + 100 * np.cos((x + y) / 60.0))
                ]
        
        tiles.append(tile)
    
    # 创建TileInfo
    tile_infos = []
    positions = [(0, 0), (0, tile_size - overlap), 
                 (tile_size - overlap, 0), (tile_size - overlap, tile_size - overlap)]
    
    for i, (tile, (y, x)) in enumerate(zip(tiles, positions)):
        tile_infos.append(TileInfo(tile, x, y, i // 2, i % 2))
    
    output_shape = (tile_size * 2 - overlap, tile_size * 2 - overlap)
    
    # 测试不同方法
    blender = BlendingModule()
    
    methods = {
        'Laplacian': lambda: blender.laplacian_fusion(tile_infos, output_shape=output_shape),
        'Weighted (Cosine)': lambda: blender.weighted_average_fusion(
            tile_infos, weight_type=WeightType.COSINE, output_shape=output_shape
        ),
        'Weighted (Linear)': lambda: blender.weighted_average_fusion(
            tile_infos, weight_type=WeightType.LINEAR, output_shape=output_shape
        ),
        'Feather': lambda: blender.feather_blend(tile_infos, feather_width=50, output_shape=output_shape)
    }
    
    results = {}
    
    print(f"{'方法':<25} {'时间(ms)':<12} {'质量指标'}")
    print("-" * 60)
    
    import time
    
    for name, method in methods.items():
        start = time.time()
        result = method()
        elapsed = (time.time() - start) * 1000
        
        # 计算质量指标
        metrics = compute_blend_quality(result, tiles, positions)
        
        results[name] = {
            'result': result,
            'time': elapsed,
            'metrics': metrics
        }
        
        print(f"{name:<25} {elapsed:<12.2f} SSIM={metrics['mean_ssim']:.4f}")
    
    return results


def example_full_pipeline():
    """
    完整融合流程示例
    
    演示从瓦片到最终结果的完整处理流程。
    """
    print("\n" + "=" * 60)
    print("示例7: 完整融合流程")
    print("=" * 60)
    
    import time
    
    # 步骤1: 创建测试瓦片
    print("\n步骤1: 创建测试瓦片...")
    tile_size = 512
    overlap = 128
    grid_size = 2
    
    tiles = []
    for i in range(grid_size * grid_size):
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        # 创建带纹理的测试图像
        for y in range(tile_size):
            for x in range(tile_size):
                tile[y, x] = [
                    int(128 + 100 * np.sin(x / 50.0 + i)),
                    int(128 + 100 * np.sin(y / 50.0 + i * 0.7)),
                    int(128 + 80 * np.cos((x + y) / 80.0))
                ]
        
        # 添加一些随机变化模拟真实场景
        variation = np.random.normal(0, 10, tile.shape)
        tile = np.clip(tile.astype(float) + variation, 0, 255).astype(np.uint8)
        
        tiles.append(tile)
    
    print(f"  创建了 {len(tiles)} 个瓦片，每个 {tile_size}x{tile_size}")
    
    # 步骤2: 创建瓦片网格
    print("\n步骤2: 创建瓦片网格...")
    tile_infos, overlap_regions = create_tile_grid(
        tiles,
        grid_shape=(grid_size, grid_size),
        overlap=overlap
    )
    print(f"  网格形状: {grid_size}x{grid_size}")
    print(f"  重叠区域: {len(overlap_regions)} 个")
    
    # 步骤3: 执行拉普拉斯融合
    print("\n步骤3: 执行拉普拉斯金字塔融合...")
    blender = BlendingModule(method='laplacian', num_levels=6)
    
    output_shape = (
        tile_size * grid_size - overlap * (grid_size - 1),
        tile_size * grid_size - overlap * (grid_size - 1)
    )
    
    start = time.time()
    fused = blender.laplacian_fusion(tile_infos, output_shape=output_shape)
    fusion_time = time.time() - start
    
    print(f"  融合完成，输出尺寸: {fused.shape}")
    print(f"  融合时间: {fusion_time:.3f}秒")
    
    # 步骤4: 检测接缝
    print("\n步骤4: 检测接缝...")
    seams = blender.detect_seams(fused, tile_infos)
    print(f"  检测到 {len(seams)} 个接缝")
    
    high_seams = [s for s in seams if s.severity == 'high']
    if high_seams:
        print(f"  其中 {len(high_seams)} 个需要修复")
    
    # 步骤5: 修复接缝（如果需要）
    if high_seams:
        print("\n步骤5: 修复接缝...")
        start = time.time()
        repaired = blender.repair_seams(fused, high_seams, tiles)
        repair_time = time.time() - start
        print(f"  修复完成，耗时: {repair_time:.3f}秒")
    else:
        repaired = fused
    
    # 步骤6: 色彩校正
    print("\n步骤6: 色彩校正...")
    start = time.time()
    corrected = blender.color_correction(
        repaired,
        reference_tile=tiles[0],
        method="histogram",
        local_filter=True
    )
    correction_time = time.time() - start
    print(f"  校正完成，耗时: {correction_time:.3f}秒")
    
    # 步骤7: 质量评估
    print("\n步骤7: 质量评估...")
    positions = [(t.y, t.x) for t in tile_infos]
    metrics = compute_blend_quality(corrected, tiles, positions)
    
    print(f"  平均SSIM: {metrics['mean_ssim']:.4f}")
    print(f"  最小SSIM: {metrics['min_ssim']:.4f}")
    print(f"  SSIM标准差: {metrics['std_ssim']:.4f}")
    print(f"  平均梯度: {metrics['mean_gradient']:.2f}")
    
    # 总时间
    total_time = fusion_time + (repair_time if high_seams else 0) + correction_time
    print(f"\n总处理时间: {total_time:.3f}秒")
    
    return corrected


# ==================== 主函数 ====================

if __name__ == "__main__":
    """
    主函数 - 运行所有示例
    
    执行融合模块的所有功能示例。
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "图像融合模块 (Blending Module)")
    print(" " * 10 + "超高分辨率图像生成系统 - 功能演示")
    print("=" * 70)
    
    # 运行所有示例
    try:
        # 示例1: 拉普拉斯金字塔融合
        result1, seams1 = example_laplacian_fusion()
        
        # 示例2: 泊松融合
        result2 = example_poisson_fusion()
        
        # 示例3: 加权平均融合
        results3 = example_weighted_average()
        
        # 示例4: 色彩校正
        result4 = example_color_correction()
        
        # 示例5: 接缝检测
        result5, seams5 = example_seam_detection()
        
        # 示例6: 方法对比
        results6 = example_compare_methods()
        
        # 示例7: 完整流程
        result7 = example_full_pipeline()
        
        print("\n" + "=" * 70)
        print("所有示例执行完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

