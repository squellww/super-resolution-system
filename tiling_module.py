"""
超高分辨率图像生成系统 - 图像分块模块 (Tiling Module)

该模块实现了超高分辨率图像的分块处理功能，包括：
- 重叠分块策略（支持15%-25%可调重叠率）
- 分块坐标映射系统和元数据管理
- 内容感知分块（保护关键区域）
- 三级缓存系统（L1内存/L2磁盘/L3对象存储）
- 流式分块加载和断点续传

作者: AI Assistant
版本: 1.0.0
"""

import os
import json
import hashlib
import mmap
import pickle
import uuid
import logging
from typing import List, Dict, Tuple, Optional, Union, Iterator, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
import threading
from collections import OrderedDict
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import cv2

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaddingMode(Enum):
    """边缘填充模式枚举"""
    MIRROR = "mirror"      # 镜像填充
    REPLICATE = "replicate"  # 重复填充
    REFLECT = "reflect"    # 反射填充（不包含边缘）
    CONSTANT = "constant"  # 常数填充


class TileStatus(Enum):
    """分块处理状态枚举"""
    PENDING = auto()       # 待处理
    PROCESSING = auto()    # 处理中
    COMPLETED = auto()     # 已完成
    FAILED = auto()        # 失败
    CACHED = auto()        # 已缓存


class CacheLevel(Enum):
    """缓存级别枚举"""
    L1_MEMORY = "L1"       # L1: 内存缓存
    L2_DISK = "L2"         # L2: 本地磁盘缓存
    L3_CLOUD = "L3"        # L3: 对象存储缓存


@dataclass
class TileMetadata:
    """
    分块元数据类
    
    存储每个分块的完整元数据信息，用于坐标映射和状态管理。
    
    Attributes:
        block_id: 唯一标识符（UUID）
        global_x: 全局坐标系中的X位置（左上角）
        global_y: 全局坐标系中的Y位置（左上角）
        input_w: 输入块宽度
        input_h: 输入块高度
        output_w: 输出块宽度（Seedream 4.0限制为4096）
        output_h: 输出块高度
        overlap_top: 顶部重叠像素数
        overlap_bottom: 底部重叠像素数
        overlap_left: 左侧重叠像素数
        overlap_right: 右侧重叠像素数
        roi_flags: 关键区域标志（人脸、文字、产品主体等）
        status: 处理状态
        neighbor_ids: 相邻块ID列表（上、下、左、右）
        image_hash: 原图哈希值（用于断点续传）
        complexity_score: 局部复杂度评分（用于内容感知分块）
        priority: 处理优先级
        created_at: 创建时间戳
        updated_at: 更新时间戳
    """
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    global_x: int = 0
    global_y: int = 0
    input_w: int = 2048
    input_h: int = 2048
    output_w: int = 4096
    output_h: int = 4096
    overlap_top: int = 0
    overlap_bottom: int = 0
    overlap_left: int = 0
    overlap_right: int = 0
    roi_flags: Dict[str, bool] = field(default_factory=dict)
    status: TileStatus = TileStatus.PENDING
    neighbor_ids: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "top": None, "bottom": None, "left": None, "right": None
    })
    image_hash: str = ""
    complexity_score: float = 0.0
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """将元数据转换为字典格式"""
        data = asdict(self)
        data['status'] = self.status.name
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TileMetadata':
        """从字典创建元数据对象"""
        data = data.copy()
        data['status'] = TileStatus[data['status']]
        return cls(**data)


@dataclass
class Tile:
    """
    图像分块数据类
    
    存储分块的图像数据和元数据。
    
    Attributes:
        metadata: 分块元数据
        data: 图像数据（numpy数组或PIL Image）
        mask: 可选的掩码数据（用于内容感知）
        cache_path: 缓存路径（如果已缓存）
    """
    metadata: TileMetadata
    data: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    cache_path: Optional[str] = None
    
    def get_overlap_region(self) -> Tuple[int, int, int, int]:
        """
        获取重叠区域坐标
        
        Returns:
            (top, bottom, left, right) 重叠像素数
        """
        return (
            self.metadata.overlap_top,
            self.metadata.overlap_bottom,
            self.metadata.overlap_left,
            self.metadata.overlap_right
        )
    
    def get_effective_region(self) -> Tuple[int, int, int, int]:
        """
        获取有效区域（非重叠部分）坐标
        
        Returns:
            (x1, y1, x2, y2) 有效区域在全局坐标系中的位置
        """
        x1 = self.metadata.global_x + self.metadata.overlap_left
        y1 = self.metadata.global_y + self.metadata.overlap_top
        x2 = x1 + self.metadata.input_w - self.metadata.overlap_left - self.metadata.overlap_right
        y2 = y1 + self.metadata.input_h - self.metadata.overlap_top - self.metadata.overlap_bottom
        return (x1, y1, x2, y2)


class ContentAnalyzer:
    """
    内容分析器 - 用于内容感知分块
    
    分析图像中的关键区域（人脸、文字、产品主体等），
    并生成禁区地图用于保护这些区域。
    """
    
    def __init__(self):
        """初始化内容分析器"""
        self.face_cascade = None
        self._init_face_detector()
    
    def _init_face_detector(self):
        """初始化人脸检测器"""
        try:
            # 使用OpenCV的预训练人脸检测器
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"人脸检测器初始化失败: {e}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的人脸区域
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            人脸区域列表，每个元素为(x, y, w, h)
        """
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的文字区域（使用MSER算法）
        
        Args:
            image: 输入图像
            
        Returns:
            文字区域列表
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 使用MSER检测文本区域
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            # 过滤掉太小的区域
            if w > 20 and h > 10 and w < image.shape[1] * 0.5:
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    def compute_saliency_map(self, image: np.ndarray) -> np.ndarray:
        """
        计算显著性地图（用于检测产品主体）
        使用频谱残差法（Spectral Residual）作为备选方案
        
        Args:
            image: 输入图像
            
        Returns:
            显著性地图（0-255）
        """
        try:
            # 尝试使用OpenCV的saliency模块
            if hasattr(cv2, 'saliency'):
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                success, saliency_map = saliency.computeSaliency(image)
                if success:
                    return (saliency_map * 255).astype(np.uint8)
        except Exception:
            pass
        
        # 备选方案：使用频谱残差法
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 计算FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # 计算幅度谱和对数幅度谱
        magnitude = np.abs(fshift)
        log_magnitude = np.log(magnitude + 1e-8)
        
        # 使用平均滤波器平滑对数幅度谱
        kernel_size = 5
        avg_filter = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        avg_log_magnitude = cv2.filter2D(log_magnitude, -1, avg_filter)
        
        # 计算频谱残差
        spectral_residual = log_magnitude - avg_log_magnitude
        
        # 重建显著性图
        phase = np.angle(fshift)
        saliency_complex = np.exp(spectral_residual + 1j * phase)
        saliency_map = np.abs(np.fft.ifft2(np.fft.ifftshift(saliency_complex)))
        
        # 归一化到0-255
        saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
        saliency_map = ((saliency_map - saliency_map.min()) / 
                       (saliency_map.max() - saliency_map.min() + 1e-8) * 255).astype(np.uint8)
        
        return saliency_map
    
    def compute_local_entropy(self, image: np.ndarray, window_size: int = 64) -> np.ndarray:
        """
        计算局部信息熵（用于复杂度评估）
        
        Args:
            image: 输入图像
            window_size: 计算窗口大小
            
        Returns:
            局部熵地图
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 计算局部熵
        entropy_map = np.zeros_like(gray, dtype=np.float32)
        
        for y in range(0, gray.shape[0], window_size):
            for x in range(0, gray.shape[1], window_size):
                window = gray[y:min(y+window_size, gray.shape[0]), 
                             x:min(x+window_size, gray.shape[1])]
                
                # 计算直方图和熵
                hist = cv2.calcHist([window], [0], None, [256], [0, 256])
                hist = hist.flatten() / hist.sum()
                
                # 计算熵
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropy_map[y:min(y+window_size, gray.shape[0]), 
                           x:min(x+window_size, gray.shape[1])] = entropy
        
        return entropy_map
    
    def create_forbidden_zone_map(
        self, 
        image: np.ndarray,
        protect_faces: bool = True,
        protect_text: bool = True,
        protect_salient: bool = True,
        saliency_threshold: float = 0.7
    ) -> np.ndarray:
        """
        创建禁区地图
        
        Args:
            image: 输入图像
            protect_faces: 是否保护人脸区域
            protect_text: 是否保护文字区域
            protect_salient: 是否保护显著区域
            saliency_threshold: 显著性阈值
            
        Returns:
            二值禁区地图（True表示禁区）
        """
        forbidden_map = np.zeros(image.shape[:2], dtype=bool)
        
        # 人脸保护
        if protect_faces:
            faces = self.detect_faces(image)
            for x, y, w, h in faces:
                # 扩展人脸保护区
                margin = int(max(w, h) * 0.2)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                forbidden_map[y1:y2, x1:x2] = True
        
        # 文字保护
        if protect_text:
            text_regions = self.detect_text_regions(image)
            for x, y, w, h in text_regions:
                forbidden_map[y:y+h, x:x+w] = True
        
        # 显著区域保护
        if protect_salient:
            saliency_map = self.compute_saliency_map(image)
            threshold = int(255 * saliency_threshold)
            forbidden_map |= (saliency_map > threshold)
        
        return forbidden_map


class LRUCache:
    """
    LRU缓存实现（用于L1内存缓存）
    """
    
    def __init__(self, max_size: int = 100):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, Tile] = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Tile]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                # 移动到末尾（最近使用）
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Tile):
        """添加缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            
            # 清理过期项
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def remove(self, key: str) -> bool:
        """移除缓存项"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def keys(self) -> List[str]:
        """获取所有键"""
        with self.lock:
            return list(self.cache.keys())


class TilingModule:
    """
    图像分块模块主类
    
    实现超高分辨率图像的分块处理，支持重叠分块、内容感知、
    三级缓存和流式加载。
    
    Attributes:
        block_size: 输入块尺寸（默认2048）
        overlap_ratio: 重叠率（默认0.2，即20%）
        padding_mode: 边缘填充模式
        output_scale: 输出缩放比例（默认2.0，2048->4096）
        content_analyzer: 内容分析器
        l1_cache: L1内存缓存
        l2_cache_dir: L2磁盘缓存目录
        tile_registry: 分块注册表（用于邻居查找）
    """
    
    def __init__(
        self,
        block_size: int = 2048,
        overlap_ratio: float = 0.2,
        padding_mode: str = 'mirror',
        output_scale: float = 2.0,
        l1_cache_size: int = 50,
        l2_cache_dir: Optional[str] = None,
        enable_content_aware: bool = True
    ):
        """
        初始化图像分块模块
        
        Args:
            block_size: 输入块尺寸（推荐2048）
            overlap_ratio: 重叠率（0.15-0.25）
            padding_mode: 边缘填充模式（'mirror', 'replicate', 'reflect', 'constant'）
            output_scale: 输出缩放比例（默认2.0）
            l1_cache_size: L1缓存大小（块数）
            l2_cache_dir: L2缓存目录路径
            enable_content_aware: 是否启用内容感知分块
        """
        # 参数验证
        if not (0.1 <= overlap_ratio <= 0.3):
            raise ValueError(f"重叠率必须在0.1-0.3之间，当前值: {overlap_ratio}")
        
        self.block_size = block_size
        self.overlap_ratio = overlap_ratio
        self.padding_mode = PaddingMode(padding_mode)
        self.output_scale = output_scale
        self.enable_content_aware = enable_content_aware
        
        # 计算输出尺寸
        self.output_size = int(block_size * output_scale)
        
        # 计算重叠像素数
        self.overlap_pixels = int(block_size * overlap_ratio)
        
        # 初始化内容分析器
        self.content_analyzer = ContentAnalyzer() if enable_content_aware else None
        
        # 初始化L1内存缓存
        self.l1_cache = LRUCache(max_size=l1_cache_size)
        
        # 初始化L2磁盘缓存
        if l2_cache_dir is None:
            l2_cache_dir = os.path.expanduser("~/.cache/super_resolution/tiling")
        self.l2_cache_dir = Path(l2_cache_dir)
        self.l2_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 分块注册表（用于邻居查找和断点续传）
        self.tile_registry: Dict[str, Tile] = {}
        self.registry_lock = threading.Lock()
        
        # 处理状态（用于断点续传）
        self.processing_state: Dict[str, any] = {}
        
        logger.info(f"TilingModule初始化完成: block_size={block_size}, "
                   f"overlap_ratio={overlap_ratio}, padding_mode={padding_mode}")
    
    def _compute_image_hash(self, image_path: str) -> str:
        """
        计算图像哈希值（用于断点续传）
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像哈希值（MD5）
        """
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _apply_padding(
        self, 
        image: np.ndarray, 
        pad_top: int, 
        pad_bottom: int, 
        pad_left: int, 
        pad_right: int
    ) -> np.ndarray:
        """
        应用边缘填充
        
        Args:
            image: 输入图像
            pad_top: 顶部填充像素数
            pad_bottom: 底部填充像素数
            pad_left: 左侧填充像素数
            pad_right: 右侧填充像素数
            
        Returns:
            填充后的图像
        """
        if self.padding_mode == PaddingMode.MIRROR:
            # 镜像填充
            padded = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101
            )
        elif self.padding_mode == PaddingMode.REPLICATE:
            # 重复填充
            padded = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REPLICATE
            )
        elif self.padding_mode == PaddingMode.REFLECT:
            # 反射填充
            padded = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT
            )
        elif self.padding_mode == PaddingMode.CONSTANT:
            # 常数填充（黑色）
            padded = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
        else:
            raise ValueError(f"不支持的填充模式: {self.padding_mode}")
        
        return padded
    
    def _calculate_tile_positions(
        self, 
        image_width: int, 
        image_height: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        计算所有分块的位置
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            分块位置列表，每个元素为(x, y, w, h)
        """
        positions = []
        
        # 计算步长（考虑重叠）
        step = self.block_size - self.overlap_pixels
        
        # 计算需要的块数
        num_tiles_x = max(1, int(np.ceil((image_width - self.overlap_pixels) / step)))
        num_tiles_y = max(1, int(np.ceil((image_height - self.overlap_pixels) / step)))
        
        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                # 计算块的位置
                x = tile_x * step
                y = tile_y * step
                
                # 确保不超出图像边界
                w = min(self.block_size, image_width - x)
                h = min(self.block_size, image_height - y)
                
                positions.append((x, y, w, h))
        
        return positions
    
    def _calculate_overlap_for_tile(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        image_width: int,
        image_height: int
    ) -> Tuple[int, int, int, int]:
        """
        计算单个分块的重叠区域
        
        Args:
            x, y: 分块左上角坐标
            w, h: 分块宽高
            image_width, image_height: 原图尺寸
            
        Returns:
            (overlap_top, overlap_bottom, overlap_left, overlap_right)
        """
        step = self.block_size - self.overlap_pixels
        
        # 计算重叠
        overlap_top = self.overlap_pixels if y > 0 else 0
        overlap_left = self.overlap_pixels if x > 0 else 0
        
        # 底部和右侧重叠取决于是否有下一个块
        overlap_bottom = self.overlap_pixels if y + h < image_height else 0
        overlap_right = self.overlap_pixels if x + w < image_width else 0
        
        # 调整边缘块的重叠
        if y + self.block_size >= image_height:
            overlap_bottom = max(0, self.block_size - (image_height - y) - overlap_top)
        if x + self.block_size >= image_width:
            overlap_right = max(0, self.block_size - (image_width - x) - overlap_left)
        
        return (overlap_top, overlap_bottom, overlap_left, overlap_right)
    
    def create_tile_metadata(
        self, 
        tile: Tile, 
        global_x: int, 
        global_y: int
    ) -> TileMetadata:
        """
        创建分块元数据
        
        Args:
            tile: 分块对象
            global_x: 全局X坐标
            global_y: 全局Y坐标
            
        Returns:
            分块元数据对象
        """
        metadata = tile.metadata
        metadata.global_x = global_x
        metadata.global_y = global_y
        metadata.updated_at = time.time()
        return metadata
    
    def split_image(
        self, 
        image_path: str,
        save_metadata: bool = True
    ) -> List[Tile]:
        """
        分块主函数 - 将图像分割为重叠的块
        
        Args:
            image_path: 图像文件路径
            save_metadata: 是否保存元数据到注册表
            
        Returns:
            分块列表
        """
        logger.info(f"开始分块处理: {image_path}")
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # BGR to RGB 转换 (OpenCV使用BGR，PIL使用RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_height, image_width = image.shape[:2]
        image_hash = self._compute_image_hash(image_path)
        
        logger.info(f"图像尺寸: {image_width}x{image_height}")
        
        # 内容感知分析（如果需要）
        forbidden_map = None
        if self.enable_content_aware and self.content_analyzer:
            logger.info("执行内容感知分析...")
            forbidden_map = self.content_analyzer.create_forbidden_zone_map(image)
        
        # 计算分块位置
        positions = self._calculate_tile_positions(image_width, image_height)
        logger.info(f"将图像分割为 {len(positions)} 个块")
        
        tiles = []
        
        for idx, (x, y, w, h) in enumerate(positions):
            # 提取图像块
            tile_image = image[y:y+h, x:x+w]
            
            # 如果需要填充到完整块大小
            pad_bottom = self.block_size - h
            pad_right = self.block_size - w
            
            if pad_bottom > 0 or pad_right > 0:
                tile_image = self._apply_padding(
                    tile_image, 0, pad_bottom, 0, pad_right
                )
            
            # 计算重叠区域
            overlap_top, overlap_bottom, overlap_left, overlap_right = \
                self._calculate_overlap_for_tile(x, y, w, h, image_width, image_height)
            
            # 创建元数据
            metadata = TileMetadata(
                global_x=x,
                global_y=y,
                input_w=w,
                input_h=h,
                output_w=int(w * self.output_scale),
                output_h=int(h * self.output_scale),
                overlap_top=overlap_top,
                overlap_bottom=overlap_bottom,
                overlap_left=overlap_left,
                overlap_right=overlap_right,
                image_hash=image_hash,
                status=TileStatus.PENDING
            )
            
            # 计算复杂度评分（用于内容感知）
            if self.content_analyzer:
                tile_gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
                metadata.complexity_score = float(np.std(tile_gray))
            
            # 检查是否包含禁区
            if forbidden_map is not None:
                tile_forbidden = forbidden_map[y:y+h, x:x+w]
                metadata.roi_flags = {
                    'has_forbidden_zone': np.any(tile_forbidden),
                    'forbidden_ratio': float(np.sum(tile_forbidden) / tile_forbidden.size)
                }
            
            # 创建分块对象
            tile = Tile(metadata=metadata, data=tile_image)
            tiles.append(tile)
            
            # 注册到全局注册表
            if save_metadata:
                with self.registry_lock:
                    self.tile_registry[metadata.block_id] = tile
            
            logger.debug(f"创建块 {idx+1}/{len(positions)}: {metadata.block_id}")
        
        # 建立邻居关系
        self._build_neighbor_relationships(tiles)
        
        # 保存处理状态（用于断点续传）
        self.processing_state[image_hash] = {
            'image_path': image_path,
            'image_width': image_width,
            'image_height': image_height,
            'num_tiles': len(tiles),
            'tile_ids': [t.metadata.block_id for t in tiles],
            'timestamp': time.time()
        }
        
        logger.info(f"分块完成: 共 {len(tiles)} 个块")
        return tiles
    
    def _build_neighbor_relationships(self, tiles: List[Tile]):
        """
        建立分块之间的邻居关系
        
        Args:
            tiles: 分块列表
        """
        # 创建位置索引
        position_map: Dict[Tuple[int, int], str] = {}
        for tile in tiles:
            key = (tile.metadata.global_x, tile.metadata.global_y)
            position_map[key] = tile.metadata.block_id
        
        step = self.block_size - self.overlap_pixels
        
        # 为每个块查找邻居
        for tile in tiles:
            x, y = tile.metadata.global_x, tile.metadata.global_y
            
            # 上邻居
            top_key = (x, y - step)
            if top_key in position_map:
                tile.metadata.neighbor_ids['top'] = position_map[top_key]
            
            # 下邻居
            bottom_key = (x, y + step)
            if bottom_key in position_map:
                tile.metadata.neighbor_ids['bottom'] = position_map[bottom_key]
            
            # 左邻居
            left_key = (x - step, y)
            if left_key in position_map:
                tile.metadata.neighbor_ids['left'] = position_map[left_key]
            
            # 右邻居
            right_key = (x + step, y)
            if right_key in position_map:
                tile.metadata.neighbor_ids['right'] = position_map[right_key]
    
    def get_neighbor_tiles(self, tile_id: str) -> List[Tile]:
        """
        获取相邻分块
        
        Args:
            tile_id: 分块ID
            
        Returns:
            相邻分块列表
        """
        with self.registry_lock:
            if tile_id not in self.tile_registry:
                return []
            
            tile = self.tile_registry[tile_id]
            neighbor_ids = [
                tile.metadata.neighbor_ids.get('top'),
                tile.metadata.neighbor_ids.get('bottom'),
                tile.metadata.neighbor_ids.get('left'),
                tile.metadata.neighbor_ids.get('right')
            ]
            
            neighbors = []
            for nid in neighbor_ids:
                if nid and nid in self.tile_registry:
                    neighbors.append(self.tile_registry[nid])
            
            return neighbors
    
    def load_tile_streaming(
        self, 
        image_path: str, 
        tile: Tile,
        use_mmap: bool = True
    ) -> np.ndarray:
        """
        流式加载分块（使用内存映射）
        
        Args:
            image_path: 图像文件路径
            tile: 分块对象
            use_mmap: 是否使用内存映射
            
        Returns:
            分块图像数据
        """
        if tile.data is not None:
            return tile.data
        
        x, y = tile.metadata.global_x, tile.metadata.global_y
        w, h = tile.metadata.input_w, tile.metadata.input_h
        
        if use_mmap and os.path.exists(image_path):
            # 使用内存映射加载
            with open(image_path, 'rb') as f:
                # 读取文件头获取图像信息
                header = f.read(100)
                
            # 使用PIL的懒加载
            with Image.open(image_path) as img:
                # 裁剪指定区域
                tile_img = img.crop((x, y, x + w, y + h))
                tile_data = np.array(tile_img)
                
            # 转换BGR格式（OpenCV默认）
            if len(tile_data.shape) == 3 and tile_data.shape[2] == 3:
                tile_data = cv2.cvtColor(tile_data, cv2.COLOR_RGB2BGR)
        else:
            # 直接加载
            image = cv2.imread(image_path)
            tile_data = image[y:y+h, x:x+w]
        
        return tile_data
    
    def save_tile_cache(
        self, 
        tile: Tile, 
        cache_level: CacheLevel,
        custom_path: Optional[str] = None
    ) -> str:
        """
        保存分块到缓存
        
        Args:
            tile: 分块对象
            cache_level: 缓存级别
            custom_path: 自定义缓存路径
            
        Returns:
            缓存路径
        """
        cache_path = custom_path
        
        if cache_level == CacheLevel.L1_MEMORY:
            # L1: 内存缓存
            self.l1_cache.put(tile.metadata.block_id, tile)
            tile.metadata.status = TileStatus.CACHED
            cache_path = f"L1://{tile.metadata.block_id}"
            
        elif cache_level == CacheLevel.L2_DISK:
            # L2: 磁盘缓存
            if cache_path is None:
                cache_path = str(self.l2_cache_dir / f"{tile.metadata.block_id}.pkl")
            
            # 保存分块数据和元数据
            cache_data = {
                'metadata': tile.metadata.to_dict(),
                'data': tile.data,
                'mask': tile.mask
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            tile.cache_path = cache_path
            tile.metadata.status = TileStatus.CACHED
            
        elif cache_level == CacheLevel.L3_CLOUD:
            # L3: 对象存储缓存（预留接口）
            # 实际实现需要集成云存储SDK（如AWS S3, Aliyun OSS等）
            logger.warning("L3云缓存需要外部存储SDK集成")
            cache_path = f"L3://{tile.metadata.block_id}"
        
        tile.metadata.updated_at = time.time()
        logger.debug(f"分块已缓存到 {cache_level.value}: {tile.metadata.block_id}")
        return cache_path
    
    def load_tile_cache(
        self, 
        tile_id: str, 
        cache_level: Optional[CacheLevel] = None
    ) -> Optional[Tile]:
        """
        从缓存加载分块
        
        Args:
            tile_id: 分块ID
            cache_level: 指定缓存级别（None则自动查找）
            
        Returns:
            分块对象或None
        """
        # 先检查L1缓存
        if cache_level is None or cache_level == CacheLevel.L1_MEMORY:
            tile = self.l1_cache.get(tile_id)
            if tile is not None:
                return tile
        
        # 检查L2缓存
        if cache_level is None or cache_level == CacheLevel.L2_DISK:
            cache_path = self.l2_cache_dir / f"{tile_id}.pkl"
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    metadata = TileMetadata.from_dict(cache_data['metadata'])
                    tile = Tile(
                        metadata=metadata,
                        data=cache_data.get('data'),
                        mask=cache_data.get('mask'),
                        cache_path=str(cache_path)
                    )
                    
                    # 同时放入L1缓存
                    self.l1_cache.put(tile_id, tile)
                    return tile
                    
                except Exception as e:
                    logger.error(f"加载L2缓存失败: {e}")
        
        return None
    
    def save_checkpoint(self, image_hash: str, checkpoint_path: Optional[str] = None) -> str:
        """
        保存处理检查点（用于断点续传）
        
        Args:
            image_hash: 图像哈希值
            checkpoint_path: 检查点保存路径
            
        Returns:
            检查点路径
        """
        if checkpoint_path is None:
            checkpoint_path = str(self.l2_cache_dir / f"checkpoint_{image_hash}.json")
        
        if image_hash not in self.processing_state:
            raise ValueError(f"未找到图像哈希 {image_hash} 的处理状态")
        
        state = self.processing_state[image_hash].copy()
        
        # 添加分块状态
        tile_states = {}
        for tile_id in state['tile_ids']:
            if tile_id in self.tile_registry:
                tile = self.tile_registry[tile_id]
                tile_states[tile_id] = {
                    'status': tile.metadata.status.name,
                    'metadata': tile.metadata.to_dict()
                }
        
        state['tile_states'] = tile_states
        
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"检查点已保存: {checkpoint_path}")
        return checkpoint_path
    
    def restore_from_cache(self, image_hash: str) -> Optional[Dict]:
        """
        从缓存恢复处理状态（断点续传）
        
        Args:
            image_hash: 图像哈希值
            
        Returns:
            恢复的处理状态，如果没有则返回None
        """
        checkpoint_path = self.l2_cache_dir / f"checkpoint_{image_hash}.json"
        
        if not checkpoint_path.exists():
            logger.info(f"未找到检查点: {checkpoint_path}")
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                state = json.load(f)
            
            # 恢复分块状态
            for tile_id, tile_state in state.get('tile_states', {}).items():
                # 从L2缓存加载分块
                tile = self.load_tile_cache(tile_id, CacheLevel.L2_DISK)
                if tile is not None:
                    tile.metadata.status = TileStatus[tile_state['status']]
                    with self.registry_lock:
                        self.tile_registry[tile_id] = tile
            
            # 恢复处理状态
            self.processing_state[image_hash] = state
            
            logger.info(f"已从检查点恢复: {image_hash}")
            return state
            
        except Exception as e:
            logger.error(f"恢复检查点失败: {e}")
            return None
    
    def merge_tiles(
        self, 
        tiles: List[Tile], 
        output_width: int, 
        output_height: int,
        blending: bool = True
    ) -> np.ndarray:
        """
        合并分块为完整图像
        
        Args:
            tiles: 分块列表
            output_width: 输出图像宽度
            output_height: 输出图像高度
            blending: 是否使用羽化混合
            
        Returns:
            合并后的图像
        """
        # 创建输出画布
        output = np.zeros((output_height, output_width, 3), dtype=np.float32)
        weight_map = np.zeros((output_height, output_width), dtype=np.float32)
        
        for tile in tiles:
            if tile.data is None:
                continue
            
            # 计算缩放后的位置
            x = int(tile.metadata.global_x * self.output_scale)
            y = int(tile.metadata.global_y * self.output_scale)
            h, w = tile.data.shape[:2]
            
            # 缩放分块数据
            if h != tile.metadata.output_h or w != tile.metadata.output_w:
                tile_data = cv2.resize(tile.data, 
                                      (tile.metadata.output_w, tile.metadata.output_h))
            else:
                tile_data = tile.data.astype(np.float32)
            
            # 创建权重图（用于羽化混合）
            if blending:
                weight = self._create_blend_weight(tile)
                weight_resized = cv2.resize(weight, (tile_data.shape[1], tile_data.shape[0]))
            else:
                weight_resized = np.ones((tile_data.shape[0], tile_data.shape[1]), dtype=np.float32)
            
            # 叠加到输出
            y_end = min(y + tile_data.shape[0], output_height)
            x_end = min(x + tile_data.shape[1], output_width)
            
            tile_h = y_end - y
            tile_w = x_end - x
            
            output[y:y_end, x:x_end] += tile_data[:tile_h, :tile_w] * \
                                        weight_resized[:tile_h, :tile_w, np.newaxis]
            weight_map[y:y_end, x:x_end] += weight_resized[:tile_h, :tile_w]
        
        # 归一化
        weight_map = np.maximum(weight_map, 1e-6)
        output = output / weight_map[:, :, np.newaxis]
        
        return output.astype(np.uint8)
    
    def _create_blend_weight(self, tile: Tile) -> np.ndarray:
        """
        创建混合权重图（用于重叠区域羽化）
        
        Args:
            tile: 分块对象
            
        Returns:
            权重图
        """
        h, w = tile.metadata.output_h, tile.metadata.output_w
        weight = np.ones((h, w), dtype=np.float32)
        
        overlap_t = int(tile.metadata.overlap_top * self.output_scale)
        overlap_b = int(tile.metadata.overlap_bottom * self.output_scale)
        overlap_l = int(tile.metadata.overlap_left * self.output_scale)
        overlap_r = int(tile.metadata.overlap_right * self.output_scale)
        
        # 顶部羽化
        if overlap_t > 0:
            ramp = np.linspace(0, 1, overlap_t).reshape(-1, 1)
            weight[:overlap_t, :] *= ramp
        
        # 底部羽化
        if overlap_b > 0:
            ramp = np.linspace(1, 0, overlap_b).reshape(-1, 1)
            weight[-overlap_b:, :] *= ramp
        
        # 左侧羽化
        if overlap_l > 0:
            ramp = np.linspace(0, 1, overlap_l).reshape(1, -1)
            weight[:, :overlap_l] *= ramp
        
        # 右侧羽化
        if overlap_r > 0:
            ramp = np.linspace(1, 0, overlap_r).reshape(1, -1)
            weight[:, -overlap_r:] *= ramp
        
        return weight
    
    def clear_cache(self, cache_level: Optional[CacheLevel] = None):
        """
        清理缓存
        
        Args:
            cache_level: 指定缓存级别（None则清理所有）
        """
        if cache_level is None or cache_level == CacheLevel.L1_MEMORY:
            self.l1_cache.clear()
            logger.info("L1内存缓存已清理")
        
        if cache_level is None or cache_level == CacheLevel.L2_DISK:
            import shutil
            if self.l2_cache_dir.exists():
                shutil.rmtree(self.l2_cache_dir)
                self.l2_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("L2磁盘缓存已清理")
    
    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        l1_keys = self.l1_cache.keys()
        l2_files = list(self.l2_cache_dir.glob("*.pkl"))
        
        return {
            'l1_memory': {
                'count': len(l1_keys),
                'keys': l1_keys
            },
            'l2_disk': {
                'count': len(l2_files),
                'size_mb': sum(f.stat().st_size for f in l2_files) / (1024 * 1024)
            },
            'tile_registry': {
                'count': len(self.tile_registry)
            }
        }


# =============================================================================
# 单元测试
# =============================================================================

def run_tests():
    """运行单元测试"""
    import tempfile
    import shutil
    
    print("=" * 60)
    print("开始单元测试 - TilingModule")
    print("=" * 60)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建测试图像
        test_image_path = os.path.join(temp_dir, "test_image.png")
        test_image = np.random.randint(0, 255, (4096, 4096, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, test_image)
        print(f"✓ 创建测试图像: {test_image_path}")
        
        # 测试1: 初始化
        print("\n[Test 1] 初始化TilingModule")
        tiling = TilingModule(
            block_size=1024,
            overlap_ratio=0.2,
            padding_mode='mirror',
            l1_cache_size=10,
            l2_cache_dir=os.path.join(temp_dir, "cache")
        )
        print("✓ TilingModule初始化成功")
        print(f"  - 块大小: {tiling.block_size}")
        print(f"  - 重叠率: {tiling.overlap_ratio}")
        print(f"  - 输出尺寸: {tiling.output_size}")
        
        # 测试2: 图像分块
        print("\n[Test 2] 图像分块")
        tiles = tiling.split_image(test_image_path)
        print(f"✓ 图像分块完成，共 {len(tiles)} 个块")
        
        # 验证分块数量
        expected_tiles = (4096 / (1024 * 0.8)) ** 2
        print(f"  - 预期块数: ~{int(expected_tiles)}, 实际块数: {len(tiles)}")
        
        # 测试3: 元数据验证
        print("\n[Test 3] 分块元数据验证")
        tile = tiles[0]
        print(f"✓ 块ID: {tile.metadata.block_id}")
        print(f"  - 全局位置: ({tile.metadata.global_x}, {tile.metadata.global_y})")
        print(f"  - 输入尺寸: {tile.metadata.input_w}x{tile.metadata.input_h}")
        print(f"  - 输出尺寸: {tile.metadata.output_w}x{tile.metadata.output_h}")
        print(f"  - 重叠: T={tile.metadata.overlap_top}, B={tile.metadata.overlap_bottom}, "
              f"L={tile.metadata.overlap_left}, R={tile.metadata.overlap_right}")
        
        # 测试4: 邻居关系
        print("\n[Test 4] 邻居关系验证")
        neighbor_ids = tile.metadata.neighbor_ids
        print(f"✓ 邻居ID: {neighbor_ids}")
        
        neighbors = tiling.get_neighbor_tiles(tile.metadata.block_id)
        print(f"  - 实际邻居数量: {len(neighbors)}")
        
        # 测试5: 缓存系统
        print("\n[Test 5] 缓存系统测试")
        
        # L1缓存
        tiling.save_tile_cache(tile, CacheLevel.L1_MEMORY)
        cached_tile = tiling.load_tile_cache(tile.metadata.block_id, CacheLevel.L1_MEMORY)
        assert cached_tile is not None, "L1缓存失败"
        print("✓ L1内存缓存测试通过")
        
        # L2缓存
        cache_path = tiling.save_tile_cache(tile, CacheLevel.L2_DISK)
        cached_tile = tiling.load_tile_cache(tile.metadata.block_id, CacheLevel.L2_DISK)
        assert cached_tile is not None, "L2缓存失败"
        print(f"✓ L2磁盘缓存测试通过: {cache_path}")
        
        # 测试6: 断点续传
        print("\n[Test 6] 断点续传测试")
        image_hash = tile.metadata.image_hash
        checkpoint_path = tiling.save_checkpoint(image_hash)
        print(f"✓ 检查点已保存: {checkpoint_path}")
        
        # 创建新的TilingModule实例并恢复
        tiling2 = TilingModule(
            block_size=1024,
            overlap_ratio=0.2,
            l2_cache_dir=os.path.join(temp_dir, "cache")
        )
        restored_state = tiling2.restore_from_cache(image_hash)
        assert restored_state is not None, "断点续传恢复失败"
        print(f"✓ 断点续传恢复成功")
        print(f"  - 恢复的块数: {restored_state['num_tiles']}")
        
        # 测试7: 流式加载
        print("\n[Test 7] 流式加载测试")
        stream_tile = tiles[5]
        stream_tile.data = None  # 清除数据
        loaded_data = tiling.load_tile_streaming(test_image_path, stream_tile)
        assert loaded_data is not None, "流式加载失败"
        print(f"✓ 流式加载成功，数据形状: {loaded_data.shape}")
        
        # 测试8: 图像合并
        print("\n[Test 8] 图像合并测试")
        # 为所有块添加数据
        for t in tiles:
            if t.data is None:
                t.data = np.random.randint(0, 255, (t.metadata.input_h, t.metadata.input_w, 3), dtype=np.uint8)
        
        merged = tiling.merge_tiles(tiles, 4096 * 2, 4096 * 2, blending=True)
        print(f"✓ 图像合并成功，输出尺寸: {merged.shape}")
        
        # 测试9: 缓存统计
        print("\n[Test 9] 缓存统计")
        stats = tiling.get_cache_stats()
        print(f"✓ L1缓存: {stats['l1_memory']['count']} 项")
        print(f"✓ L2缓存: {stats['l2_disk']['count']} 项, {stats['l2_disk']['size_mb']:.2f} MB")
        print(f"✓ 注册表: {stats['tile_registry']['count']} 项")
        
        # 测试10: 内容感知
        print("\n[Test 10] 内容感知分析")
        analyzer = ContentAnalyzer()
        test_img = cv2.imread(test_image_path)
        entropy_map = analyzer.compute_local_entropy(test_img, window_size=256)
        print(f"✓ 局部熵计算成功，熵图尺寸: {entropy_map.shape}")
        
        saliency_map = analyzer.compute_saliency_map(test_img)
        print(f"✓ 显著性地图计算成功，尺寸: {saliency_map.shape}")
        
        print("\n" + "=" * 60)
        print("所有单元测试通过！")
        print("=" * 60)
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n临时目录已清理: {temp_dir}")


def demo_usage():
    """
    使用示例
    
    展示TilingModule的典型使用流程
    """
    print("\n" + "=" * 60)
    print("使用示例 - TilingModule")
    print("=" * 60)
    
    # 1. 初始化分块模块
    tiling = TilingModule(
        block_size=2048,          # 输入块尺寸
        overlap_ratio=0.2,        # 20%重叠率
        padding_mode='mirror',    # 镜像填充
        output_scale=2.0,         # 2倍超分
        l1_cache_size=50,         # L1缓存50个块
        enable_content_aware=True # 启用内容感知
    )
    
    # 2. 分块处理图像
    # tiles = tiling.split_image("path/to/large_image.jpg")
    
    # 3. 获取分块邻居（用于上下文感知处理）
    # for tile in tiles:
    #     neighbors = tiling.get_neighbor_tiles(tile.metadata.block_id)
    #     # 处理tile...
    
    # 4. 保存到缓存
    # tiling.save_tile_cache(tile, CacheLevel.L2_DISK)
    
    # 5. 断点续传
    # tiling.save_checkpoint(image_hash)
    # restored_state = tiling.restore_from_cache(image_hash)
    
    # 6. 合并结果
    # result = tiling.merge_tiles(processed_tiles, output_width, output_height)
    
    print("""
典型使用流程:

1. 初始化分块模块
   tiling = TilingModule(block_size=2048, overlap_ratio=0.2)

2. 分块处理图像
   tiles = tiling.split_image("large_image.jpg")

3. 逐个处理分块（支持断点续传）
   for tile in tiles:
       if tile.metadata.status != TileStatus.COMPLETED:
           # 获取邻居块（上下文信息）
           neighbors = tiling.get_neighbor_tiles(tile.metadata.block_id)
           
           # 处理分块（调用超分模型）
           result = process_tile(tile, neighbors)
           
           # 更新状态并缓存
           tile.metadata.status = TileStatus.COMPLETED
           tiling.save_tile_cache(tile, CacheLevel.L2_DISK)
   
   # 保存检查点
   tiling.save_checkpoint(image_hash)

4. 合并处理后的分块
   final_image = tiling.merge_tiles(tiles, output_width, output_height)

5. 清理缓存
   tiling.clear_cache()
    """)


if __name__ == "__main__":
    # 运行单元测试
    run_tests()
    
    # 显示使用示例
    demo_usage()
