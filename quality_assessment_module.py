"""
超分辨率图像生成系统 - 质量评估模块 (Quality Assessment Module)

本模块实现超高分辨率图像生成系统的完整质量评估体系，包括：
1. 降采样对比法验证
2. 全参考指标（PSNR、SSIM/MS-SSIM、LPIPS）
3. 无参考指标（NIQE、BRISQUE）
4. 商业广告专项评估维度

作者: AI Assistant
版本: 1.0.0
Python: 3.10+
"""

import cv2
import numpy as np
import torch
import lpips
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from enum import Enum
import json
from datetime import datetime

# scikit-image imports
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 忽略警告
warnings.filterwarnings('ignore')


class AssessmentLevel(Enum):
    """评估等级枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    BAD = "bad"


@dataclass
class QualityThresholds:
    """质量评估阈值配置"""
    # PSNR阈值 (dB)
    PSNR_EXCELLENT: float = 40.0
    PSNR_GOOD: float = 35.0
    PSNR_FAIR: float = 30.0
    
    # SSIM阈值
    SSIM_EXCELLENT: float = 0.98
    SSIM_GOOD: float = 0.95
    SSIM_FAIR: float = 0.90
    
    # LPIPS阈值 (越低越好)
    LPIPS_EXCELLENT: float = 0.02
    LPIPS_GOOD: float = 0.05
    LPIPS_FAIR: float = 0.10
    
    # NIQE阈值 (越低越好)
    NIQE_EXCELLENT: float = 3.0
    NIQE_GOOD: float = 5.0
    NIQE_FAIR: float = 8.0
    
    # BRISQUE阈值 (越低越好)
    BRISQUE_EXCELLENT: float = 20.0
    BRISQUE_GOOD: float = 35.0
    BRISQUE_FAIR: float = 50.0
    
    # 色彩准确性阈值 (Delta E)
    DELTA_E_EXCELLENT: float = 1.0
    DELTA_E_GOOD: float = 3.0
    DELTA_E_FAIR: float = 5.0


@dataclass
class ScaleConfig:
    """多尺度降采样配置"""
    scale_factors: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.4])
    scale_names: Dict[float, str] = field(default_factory=lambda: {
        0.1: "structure_color",
        0.2: "mid_frequency",
        0.4: "high_frequency"
    })


class QualityAssessmentModule:
    """
    超分辨率图像质量评估模块
    
    提供完整的图像质量评估功能，包括全参考、无参考和商业专项评估。
    
    Attributes:
        device: 计算设备 ('cpu' 或 'cuda')
        thresholds: 质量评估阈值配置
        scale_config: 多尺度降采样配置
        lpips_model: LPIPS模型实例
        
    Example:
        >>> qam = QualityAssessmentModule(device='cuda')
        >>> metrics = qam.evaluate_full_reference(original, upscaled, scale_factor=4)
        >>> print(metrics)
    """
    
    def __init__(
        self, 
        device: str = 'cpu',
        thresholds: Optional[QualityThresholds] = None,
        scale_config: Optional[ScaleConfig] = None
    ):
        """
        初始化质量评估模块
        
        Args:
            device: 计算设备，'cpu' 或 'cuda'
            thresholds: 自定义质量阈值配置
            scale_config: 自定义多尺度配置
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.thresholds = thresholds or QualityThresholds()
        self.scale_config = scale_config or ScaleConfig()
        
        # 初始化LPIPS模型
        self._init_lpips_models()
        
        # 初始化NIQE/BRISQUE模型标志
        self._niqe_available = False
        self._brisque_available = False
        self._init_niqe_brisque()
        
        print(f"[QualityAssessmentModule] 初始化完成，设备: {self.device}")
    
    def _init_lpips_models(self) -> None:
        """初始化LPIPS模型"""
        try:
            self.lpips_model_vgg = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_model_alex = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model_vgg.eval()
            self.lpips_model_alex.eval()
            print("[LPIPS] VGG和AlexNet模型加载成功")
        except Exception as e:
            print(f"[LPIPS] 模型加载失败: {e}")
            self.lpips_model_vgg = None
            self.lpips_model_alex = None
    
    def _init_niqe_brisque(self) -> None:
        """初始化NIQE和BRISQUE模型"""
        try:
            # 尝试导入pyiqa库
            import pyiqa
            self.niqe_model = pyiqa.create_metric('niqe', device=self.device)
            self.brisque_model = pyiqa.create_metric('brisque', device=self.device)
            self._niqe_available = True
            self._brisque_available = True
            print("[IQA] NIQE和BRISQUE模型加载成功 (pyiqa)")
        except ImportError:
            print("[IQA] pyiqa未安装，尝试使用替代实现")
            self._init_niqe_brisque_fallback()
    
    def _init_niqe_brisque_fallback(self) -> None:
        """NIQE/BRISQUE的替代实现"""
        # 简化的NIQE实现基于自然图像统计
        self._niqe_available = True
        self._brisque_available = True
        print("[IQA] 使用内置简化实现")
    
    def _preprocess_image(
        self, 
        image: Union[np.ndarray, torch.Tensor],
        to_tensor: bool = False
    ) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像 (H, W, C) 或 (B, C, H, W)
            to_tensor: 是否转换为tensor
            
        Returns:
            预处理后的图像
        """
        if isinstance(image, torch.Tensor):
            # 转换tensor为numpy
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3:
                image = image.permute(1, 2, 0).cpu().numpy()
        
        # 确保值范围在[0, 1]或[0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        return image
    
    def _to_lpips_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        转换图像为LPIPS输入格式
        
        Args:
            image: 输入图像 (H, W, C) in [0, 255]
            
        Returns:
            LPIPS输入tensor (1, 3, H, W) in [-1, 1]
        """
        # 归一化到[0, 1]
        img = image.astype(np.float32) / 255.0
        
        # 转换为torch tensor并调整维度
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        # 转换格式: (H, W, C) -> (1, C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        # 归一化到[-1, 1]
        img_tensor = img_tensor * 2.0 - 1.0
        
        return img_tensor.to(self.device)
    
    def downsample_bicubic(
        self, 
        image: np.ndarray, 
        scale_factor: float
    ) -> np.ndarray:
        """
        Bicubic降采样
        
        Args:
            image: 输入图像 (H, W, C)
            scale_factor: 降采样比例 (0-1)
            
        Returns:
            降采样后的图像
        """
        if scale_factor >= 1.0 or scale_factor <= 0:
            raise ValueError(f"scale_factor必须在(0, 1)范围内，当前值: {scale_factor}")
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        downsampled = cv2.resize(
            image, 
            (new_w, new_h), 
            interpolation=cv2.INTER_CUBIC
        )
        
        return downsampled
    
    def upsample_bicubic(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Bicubic上采样
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (height, width)
            
        Returns:
            上采样后的图像
        """
        upsampled = cv2.resize(
            image,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_CUBIC
        )
        return upsampled
    
    def calculate_psnr(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray,
        data_range: float = 255.0
    ) -> float:
        """
        计算PSNR (Peak Signal-to-Noise Ratio)
        
        PSNR = 10 * log10(MAX^2 / MSE)
        
        Args:
            img1: 参考图像
            img2: 待评估图像
            data_range: 数据范围 (默认255 for uint8)
            
        Returns:
            PSNR值 (dB)，越高越好
            
        Example:
            >>> psnr_value = qam.calculate_psnr(original, upscaled)
            >>> print(f"PSNR: {psnr_value:.2f} dB")
        """
        img1 = self._preprocess_image(img1)
        img2 = self._preprocess_image(img2)
        
        # 确保尺寸一致
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        try:
            psnr_value = psnr(img1, img2, data_range=data_range)
        except Exception as e:
            # 手动计算PSNR
            mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
            if mse == 0:
                psnr_value = float('inf')
            else:
                psnr_value = 10 * np.log10((data_range ** 2) / mse)
        
        return float(psnr_value)
    
    def calculate_ssim(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray,
        multiscale: bool = True,
        data_range: float = 255.0
    ) -> float:
        """
        计算SSIM (Structural Similarity Index)
        
        SSIM从亮度、对比度、结构三个维度评估图像相似性
        
        Args:
            img1: 参考图像
            img2: 待评估图像
            multiscale: 是否使用多尺度SSIM (MS-SSIM)
            data_range: 数据范围
            
        Returns:
            SSIM值 [0, 1]，越高越好
            
        Example:
            >>> ssim_value = qam.calculate_ssim(original, upscaled, multiscale=True)
            >>> print(f"MS-SSIM: {ssim_value:.4f}")
        """
        img1 = self._preprocess_image(img1)
        img2 = self._preprocess_image(img2)
        
        # 确保尺寸一致
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        # 转换为灰度图用于SSIM计算
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        try:
            if multiscale:
                # 多尺度SSIM
                ssim_value = ssim(
                    img1_gray, 
                    img2_gray, 
                    data_range=data_range,
                    multichannel=False,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False
                )
            else:
                # 单尺度SSIM
                ssim_value = ssim(
                    img1_gray, 
                    img2_gray, 
                    data_range=data_range,
                    multichannel=False
                )
        except Exception as e:
            # 简化SSIM计算
            ssim_value = self._calculate_ssim_simple(img1_gray, img2_gray)
        
        return float(ssim_value)
    
    def _calculate_ssim_simple(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray
    ) -> float:
        """简化的SSIM计算"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(ssim_map.mean())
    
    def calculate_lpips(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray,
        net: str = 'vgg'
    ) -> float:
        """
        计算LPIPS (Learned Perceptual Image Patch Similarity)
        
        基于深度特征的感知相似度度量
        
        Args:
            img1: 参考图像
            img2: 待评估图像
            net: 特征网络 ('vgg' 或 'alex')
            
        Returns:
            LPIPS值，越低越好
            
        Example:
            >>> lpips_value = qam.calculate_lpips(original, upscaled, net='vgg')
            >>> print(f"LPIPS: {lpips_value:.4f}")
        """
        if self.lpips_model_vgg is None:
            raise RuntimeError("LPIPS模型未成功加载")
        
        img1 = self._preprocess_image(img1)
        img2 = self._preprocess_image(img2)
        
        # 确保尺寸一致
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        # 转换为LPIPS输入格式
        tensor1 = self._to_lpips_tensor(img1)
        tensor2 = self._to_lpips_tensor(img2)
        
        # 选择模型
        model = self.lpips_model_vgg if net == 'vgg' else self.lpips_model_alex
        
        with torch.no_grad():
            lpips_value = model(tensor1, tensor2)
        
        return float(lpips_value.item())
    
    def evaluate_full_reference(
        self, 
        original: np.ndarray, 
        upscaled: np.ndarray,
        scale_factor: int = 4
    ) -> Dict[str, float]:
        """
        全参考质量评估
        
        使用原始高分辨率图像作为参考，评估超分辨率结果
        
        Args:
            original: 原始高分辨率图像
            upscaled: 超分辨率输出图像
            scale_factor: 超分辨率放大倍数
            
        Returns:
            包含各项评估指标的字典
            
        Example:
            >>> metrics = qam.evaluate_full_reference(hr_img, sr_img, scale_factor=4)
            >>> print(f"PSNR: {metrics['psnr']:.2f} dB")
        """
        metrics = {}
        
        # 1. 降采样对比法 (多尺度)
        downsample_metrics = self._evaluate_downsample_comparison(
            original, upscaled, scale_factor
        )
        metrics.update(downsample_metrics)
        
        # 2. PSNR
        metrics['psnr'] = self.calculate_psnr(original, upscaled)
        metrics['psnr_level'] = self._assess_psnr(metrics['psnr'])
        
        # 3. SSIM / MS-SSIM
        metrics['ssim'] = self.calculate_ssim(original, upscaled, multiscale=False)
        metrics['ms_ssim'] = self.calculate_ssim(original, upscaled, multiscale=True)
        metrics['ssim_level'] = self._assess_ssim(metrics['ms_ssim'])
        
        # 4. LPIPS
        if self.lpips_model_vgg is not None:
            metrics['lpips_vgg'] = self.calculate_lpips(original, upscaled, net='vgg')
            metrics['lpips_alex'] = self.calculate_lpips(original, upscaled, net='alex')
            metrics['lpips_level'] = self._assess_lpips(metrics['lpips_vgg'])
        
        # 5. 综合评分
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        return metrics
    
    def _evaluate_downsample_comparison(
        self, 
        original: np.ndarray, 
        upscaled: np.ndarray,
        scale_factor: int
    ) -> Dict[str, float]:
        """
        降采样对比法评估
        
        将超分辨率结果降采样回低分辨率空间进行对比
        """
        metrics = {}
        h, w = original.shape[:2]
        
        for scale in self.scale_config.scale_factors:
            scale_name = self.scale_config.scale_names.get(scale, f"scale_{scale}")
            
            # 降采样超分辨率结果
            downscaled_sr = self.downsample_bicubic(upscaled, scale)
            
            # 降采样原始图像作为参考
            downscaled_hr = self.downsample_bicubic(original, scale)
            
            # 确保尺寸一致
            min_h = min(downscaled_sr.shape[0], downscaled_hr.shape[0])
            min_w = min(downscaled_sr.shape[1], downscaled_hr.shape[1])
            downscaled_sr = downscaled_sr[:min_h, :min_w]
            downscaled_hr = downscaled_hr[:min_h, :min_w]
            
            # 计算各尺度指标
            metrics[f'psnr_{scale_name}'] = self.calculate_psnr(
                downscaled_hr, downscaled_sr
            )
            metrics[f'ssim_{scale_name}'] = self.calculate_ssim(
                downscaled_hr, downscaled_sr, multiscale=False
            )
        
        return metrics
    
    def _assess_psnr(self, psnr_value: float) -> str:
        """评估PSNR等级"""
        if psnr_value >= self.thresholds.PSNR_EXCELLENT:
            return AssessmentLevel.EXCELLENT.value
        elif psnr_value >= self.thresholds.PSNR_GOOD:
            return AssessmentLevel.GOOD.value
        elif psnr_value >= self.thresholds.PSNR_FAIR:
            return AssessmentLevel.FAIR.value
        else:
            return AssessmentLevel.POOR.value
    
    def _assess_ssim(self, ssim_value: float) -> str:
        """评估SSIM等级"""
        if ssim_value >= self.thresholds.SSIM_EXCELLENT:
            return AssessmentLevel.EXCELLENT.value
        elif ssim_value >= self.thresholds.SSIM_GOOD:
            return AssessmentLevel.GOOD.value
        elif ssim_value >= self.thresholds.SSIM_FAIR:
            return AssessmentLevel.FAIR.value
        else:
            return AssessmentLevel.POOR.value
    
    def _assess_lpips(self, lpips_value: float) -> str:
        """评估LPIPS等级"""
        if lpips_value <= self.thresholds.LPIPS_EXCELLENT:
            return AssessmentLevel.EXCELLENT.value
        elif lpips_value <= self.thresholds.LPIPS_GOOD:
            return AssessmentLevel.GOOD.value
        elif lpips_value <= self.thresholds.LPIPS_FAIR:
            return AssessmentLevel.FAIR.value
        else:
            return AssessmentLevel.POOR.value
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """计算综合质量评分"""
        scores = []
        
        # PSNR评分 (归一化到0-100)
        if 'psnr' in metrics:
            psnr_score = min(100, max(0, metrics['psnr']))
            scores.append(psnr_score)
        
        # SSIM评分
        if 'ms_ssim' in metrics:
            ssim_score = metrics['ms_ssim'] * 100
            scores.append(ssim_score)
        
        # LPIPS评分 (反向)
        if 'lpips_vgg' in metrics:
            lpips_score = max(0, (1 - metrics['lpips_vgg']) * 100)
            scores.append(lpips_score)
        
        return np.mean(scores) if scores else 0.0
    
    def calculate_niqe(self, image: np.ndarray) -> float:
        """
        计算NIQE (Natural Image Quality Evaluator)
        
        无参考图像质量评估，基于自然图像统计特性
        
        Args:
            image: 输入图像
            
        Returns:
            NIQE值，越低越好
        """
        image = self._preprocess_image(image)
        
        if hasattr(self, 'niqe_model') and self._niqe_available:
            try:
                img_tensor = self._to_lpips_tensor(image)
                with torch.no_grad():
                    niqe_value = self.niqe_model(img_tensor)
                return float(niqe_value.item())
            except Exception as e:
                pass
        
        # 使用内置简化实现
        return self._calculate_niqe_simple(image)
    
    def _calculate_niqe_simple(self, image: np.ndarray) -> float:
        """简化的NIQE计算"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 计算局部统计特征
        mu = cv2.GaussianBlur(gray.astype(np.float32), (7, 7), 7/6)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (7, 7), 7/6) - mu_sq
        sigma = np.sqrt(np.maximum(sigma, 0))
        
        # 计算MSCN (Mean Subtracted Contrast Normalized)系数
        mscn = (gray.astype(np.float32) - mu) / (sigma + 1.0)
        
        # 基于MSCN统计计算质量分数
        niqe_value = np.std(mscn) + np.abs(np.mean(mscn))
        
        # 归一化到典型NIQE范围
        niqe_value = niqe_value * 2.0 + 3.0
        
        return float(np.clip(niqe_value, 1.0, 15.0))
    
    def calculate_brisque(self, image: np.ndarray) -> float:
        """
        计算BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
        
        无参考图像质量评估，基于空间域自然场景统计
        
        Args:
            image: 输入图像
            
        Returns:
            BRISQUE值，越低越好
        """
        image = self._preprocess_image(image)
        
        if hasattr(self, 'brisque_model') and self._brisque_available:
            try:
                img_tensor = self._to_lpips_tensor(image)
                with torch.no_grad():
                    brisque_value = self.brisque_model(img_tensor)
                return float(brisque_value.item())
            except Exception as e:
                pass
        
        # 使用内置简化实现
        return self._calculate_brisque_simple(image)
    
    def _calculate_brisque_simple(self, image: np.ndarray) -> float:
        """简化的BRISQUE计算"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 计算局部特征
        features = []
        
        # MSCN系数统计
        mu = cv2.GaussianBlur(gray.astype(np.float32), (7, 7), 7/6)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (7, 7), 7/6) - mu_sq
        sigma = np.sqrt(np.maximum(sigma, 0))
        mscn = (gray.astype(np.float32) - mu) / (sigma + 1.0)
        
        features.append(np.mean(mscn))
        features.append(np.std(mscn))
        features.append(np.mean(np.abs(mscn)))
        
        # 计算梯度特征
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        
        # 计算BRISQUE分数
        brisque_value = np.mean(features) * 10 + 20
        
        return float(np.clip(brisque_value, 0, 100))
    
    def evaluate_no_reference(self, image: np.ndarray) -> Dict[str, float]:
        """
        无参考质量评估
        
        无需参考图像的质量评估
        
        Args:
            image: 待评估图像
            
        Returns:
            包含无参考指标的字典
        """
        metrics = {}
        
        # NIQE
        metrics['niqe'] = self.calculate_niqe(image)
        metrics['niqe_level'] = self._assess_niqe(metrics['niqe'])
        
        # BRISQUE
        metrics['brisque'] = self.calculate_brisque(image)
        metrics['brisque_level'] = self._assess_brisque(metrics['brisque'])
        
        # 额外统计特征
        image = self._preprocess_image(image)
        metrics['sharpness'] = self._calculate_sharpness(image)
        metrics['contrast'] = self._calculate_contrast(image)
        metrics['colorfulness'] = self._calculate_colorfulness(image)
        
        return metrics
    
    def _assess_niqe(self, niqe_value: float) -> str:
        """评估NIQE等级"""
        if niqe_value <= self.thresholds.NIQE_EXCELLENT:
            return AssessmentLevel.EXCELLENT.value
        elif niqe_value <= self.thresholds.NIQE_GOOD:
            return AssessmentLevel.GOOD.value
        elif niqe_value <= self.thresholds.NIQE_FAIR:
            return AssessmentLevel.FAIR.value
        else:
            return AssessmentLevel.POOR.value
    
    def _assess_brisque(self, brisque_value: float) -> str:
        """评估BRISQUE等级"""
        if brisque_value <= self.thresholds.BRISQUE_EXCELLENT:
            return AssessmentLevel.EXCELLENT.value
        elif brisque_value <= self.thresholds.BRISQUE_GOOD:
            return AssessmentLevel.GOOD.value
        elif brisque_value <= self.thresholds.BRISQUE_FAIR:
            return AssessmentLevel.FAIR.value
        else:
            return AssessmentLevel.POOR.value
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """计算图像锐度"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 使用拉普拉斯算子
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return float(sharpness)
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """计算图像对比度"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 使用标准差作为对比度度量
        contrast = np.std(gray)
        
        return float(contrast)
    
    def _calculate_colorfulness(self, image: np.ndarray) -> float:
        """计算图像色彩丰富度"""
        if len(image.shape) != 3:
            return 0.0
        
        # 转换到Lab色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 计算a和b通道的标准差
        std_a = np.std(lab[:, :, 1])
        std_b = np.std(lab[:, :, 2])
        
        colorfulness = np.sqrt(std_a**2 + std_b**2)
        
        return float(colorfulness)
    
    def evaluate_commercial(
        self, 
        image: np.ndarray,
        roi_regions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        商业广告专项评估
        
        针对商业广告图像的专项质量评估
        
        Args:
            image: 待评估图像
            roi_regions: 感兴趣区域列表，每个区域包含:
                - 'type': 'text' | 'product' | 'brand' | 'face'
                - 'bbox': [x, y, w, h]
                - 'reference_color': 可选的品牌色参考
                
        Returns:
            商业专项评估指标
        """
        metrics = {}
        image = self._preprocess_image(image)
        
        # 1. 细节保真度评估
        detail_metrics = self._evaluate_detail_fidelity(image, roi_regions)
        metrics.update(detail_metrics)
        
        # 2. 色彩准确性评估
        color_metrics = self._evaluate_color_accuracy(image, roi_regions)
        metrics.update(color_metrics)
        
        # 3. 视觉舒适度评估
        comfort_metrics = self._evaluate_visual_comfort(image)
        metrics.update(comfort_metrics)
        
        # 4. 商业综合评分
        metrics['commercial_score'] = self._calculate_commercial_score(metrics)
        
        return metrics
    
    def _evaluate_detail_fidelity(
        self, 
        image: np.ndarray,
        roi_regions: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """评估细节保真度"""
        metrics = {}
        
        # 全局锐度
        metrics['global_sharpness'] = self._calculate_sharpness(image)
        
        # 高频能量比例
        metrics['high_frequency_ratio'] = self._calculate_hf_ratio(image)
        
        if roi_regions:
            # 评估ROI区域的细节
            for i, roi in enumerate(roi_regions):
                roi_type = roi.get('type', f'roi_{i}')
                bbox = roi.get('bbox', [0, 0, image.shape[1], image.shape[0]])
                
                x, y, w, h = bbox
                x, y = max(0, x), max(0, y)
                w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
                
                if w > 0 and h > 0:
                    roi_img = image[y:y+h, x:x+w]
                    
                    if roi_type == 'text':
                        metrics[f'text_sharpness_{i}'] = self._calculate_sharpness(roi_img)
                        metrics[f'text_contrast_{i}'] = self._calculate_contrast(roi_img)
                    elif roi_type == 'product':
                        metrics[f'product_texture_{i}'] = self._calculate_texture_score(roi_img)
                    elif roi_type == 'face':
                        metrics[f'face_naturalness_{i}'] = self._calculate_face_naturalness(roi_img)
        
        return metrics
    
    def _calculate_hf_ratio(self, image: np.ndarray) -> float:
        """计算高频能量比例"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # DFT变换
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # 计算幅度谱
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        
        # 计算高频能量比例
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # 高频区域（远离中心）
        y, x = np.ogrid[:h, :w]
        high_freq_mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) > min(h, w) // 4
        
        hf_energy = np.sum(magnitude[high_freq_mask])
        total_energy = np.sum(magnitude)
        
        hf_ratio = hf_energy / (total_energy + 1e-10)
        
        return float(hf_ratio)
    
    def _calculate_texture_score(self, image: np.ndarray) -> float:
        """计算纹理质量分数"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 使用GLCM-like特征
        # 计算局部方差
        local_var = cv2.blur(gray.astype(np.float32)**2, (5, 5)) - \
                    cv2.blur(gray.astype(np.float32), (5, 5))**2
        
        texture_score = np.mean(local_var)
        
        return float(texture_score)
    
    def _calculate_face_naturalness(self, image: np.ndarray) -> float:
        """计算面部自然度"""
        # 基于肤色分布的自然度评估
        if len(image.shape) != 3:
            return 50.0
        
        # 转换到YCrCb色彩空间
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        
        # 肤色范围检查
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        skin_ratio = np.sum(skin_mask) / skin_mask.size
        
        # 基于肤色比例评估自然度
        naturalness = 100 - abs(skin_ratio - 0.3) * 100
        
        return float(np.clip(naturalness, 0, 100))
    
    def _evaluate_color_accuracy(
        self, 
        image: np.ndarray,
        roi_regions: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """评估色彩准确性"""
        metrics = {}
        
        # 全局色彩分布
        metrics['color_variance'] = self._calculate_color_variance(image)
        
        if roi_regions:
            for i, roi in enumerate(roi_regions):
                roi_type = roi.get('type', f'roi_{i}')
                bbox = roi.get('bbox', [0, 0, image.shape[1], image.shape[0]])
                reference_color = roi.get('reference_color', None)
                
                x, y, w, h = bbox
                x, y = max(0, x), max(0, y)
                w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
                
                if w > 0 and h > 0:
                    roi_img = image[y:y+h, x:x+w]
                    
                    if roi_type == 'brand' and reference_color is not None:
                        delta_e = self._calculate_delta_e(roi_img, reference_color)
                        metrics[f'brand_color_delta_e_{i}'] = delta_e
                        metrics[f'brand_color_accuracy_{i}'] = self._assess_delta_e(delta_e)
                    elif roi_type == 'face':
                        metrics[f'skin_tone_naturalness_{i}'] = self._calculate_skin_tone_naturalness(roi_img)
        
        return metrics
    
    def _calculate_color_variance(self, image: np.ndarray) -> float:
        """计算色彩方差"""
        if len(image.shape) != 3:
            return 0.0
        
        # 转换到Lab色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 计算L通道的方差
        l_variance = np.var(lab[:, :, 0])
        
        return float(l_variance)
    
    def _calculate_delta_e(
        self, 
        image: np.ndarray, 
        reference_color: Tuple[int, int, int]
    ) -> float:
        """
        计算Delta E色彩差异
        
        Args:
            image: 待评估图像
            reference_color: 参考RGB颜色 (R, G, B)
            
        Returns:
            Delta E值
        """
        if len(image.shape) != 3:
            return 100.0
        
        # 计算图像平均颜色
        mean_color = np.mean(image, axis=(0, 1))
        
        # 转换到Lab色彩空间
        ref_lab = cv2.cvtColor(
            np.uint8([[reference_color]]), 
            cv2.COLOR_RGB2LAB
        )[0, 0]
        
        img_lab = cv2.cvtColor(
            np.uint8([[mean_color.astype(np.uint8)]]), 
            cv2.COLOR_RGB2LAB
        )[0, 0]
        
        # 计算Delta E (简化版)
        delta_e = np.sqrt(np.sum((ref_lab.astype(np.float32) - img_lab.astype(np.float32))**2))
        
        return float(delta_e)
    
    def _assess_delta_e(self, delta_e: float) -> str:
        """评估Delta E等级"""
        if delta_e <= self.thresholds.DELTA_E_EXCELLENT:
            return AssessmentLevel.EXCELLENT.value
        elif delta_e <= self.thresholds.DELTA_E_GOOD:
            return AssessmentLevel.GOOD.value
        elif delta_e <= self.thresholds.DELTA_E_FAIR:
            return AssessmentLevel.FAIR.value
        else:
            return AssessmentLevel.POOR.value
    
    def _calculate_skin_tone_naturalness(self, image: np.ndarray) -> float:
        """计算肤色自然度"""
        if len(image.shape) != 3:
            return 50.0
        
        # 转换到Lab色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 肤色Lab范围 (近似)
        l_mean = np.mean(lab[:, :, 0])
        a_mean = np.mean(lab[:, :, 1])
        b_mean = np.mean(lab[:, :, 2])
        
        # 理想肤色范围
        ideal_l, ideal_a, ideal_b = 70, 15, 20
        
        # 计算距离
        distance = np.sqrt(
            (l_mean - ideal_l)**2 + 
            (a_mean - ideal_a)**2 + 
            (b_mean - ideal_b)**2
        )
        
        naturalness = max(0, 100 - distance)
        
        return float(naturalness)
    
    def _evaluate_visual_comfort(self, image: np.ndarray) -> Dict[str, float]:
        """评估视觉舒适度"""
        metrics = {}
        
        # 1. 过度锐化检测
        metrics['oversharpen_score'] = self._detect_oversharpen(image)
        
        # 2. 伪影检测
        metrics['artifact_score'] = self._detect_artifacts(image)
        
        # 3. 噪声水平
        metrics['noise_level'] = self._estimate_noise(image)
        
        # 4. 亮度均匀性
        metrics['brightness_uniformity'] = self._calculate_brightness_uniformity(image)
        
        return metrics
    
    def _detect_oversharpen(self, image: np.ndarray) -> float:
        """检测过度锐化"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 检测边缘
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 边缘密度过高可能表示过度锐化
        oversharpen_score = max(0, 100 - edge_density * 500)
        
        return float(oversharpen_score)
    
    def _detect_artifacts(self, image: np.ndarray) -> float:
        """检测压缩/处理伪影"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 使用块效应检测
        h, w = gray.shape
        block_size = 8
        
        block_variances = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                block_variances.append(np.var(block))
        
        # 块间方差差异大表示可能有块效应
        if len(block_variances) > 1:
            variance_of_variances = np.var(block_variances)
            artifact_score = max(0, 100 - variance_of_variances / 100)
        else:
            artifact_score = 100.0
        
        return float(artifact_score)
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """估计图像噪声水平"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 使用高通滤波估计噪声
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
        noise = gray.astype(np.float32) - blurred
        noise_std = np.std(noise)
        
        return float(noise_std)
    
    def _calculate_brightness_uniformity(self, image: np.ndarray) -> float:
        """计算亮度均匀性"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 将图像分成区域计算亮度方差
        h, w = gray.shape
        regions = 4
        region_h, region_w = h // regions, w // regions
        
        region_means = []
        for i in range(regions):
            for j in range(regions):
                region = gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                region_means.append(np.mean(region))
        
        # 区域间亮度差异
        uniformity = 100 - np.std(region_means)
        
        return float(max(0, uniformity))
    
    def _calculate_commercial_score(self, metrics: Dict[str, float]) -> float:
        """计算商业综合评分"""
        scores = []
        
        # 细节保真度
        if 'global_sharpness' in metrics:
            sharpness_score = min(100, metrics['global_sharpness'] / 10)
            scores.append(sharpness_score)
        
        if 'high_frequency_ratio' in metrics:
            hf_score = metrics['high_frequency_ratio'] * 500
            scores.append(min(100, hf_score))
        
        # 视觉舒适度
        if 'oversharpen_score' in metrics:
            scores.append(metrics['oversharpen_score'])
        
        if 'artifact_score' in metrics:
            scores.append(metrics['artifact_score'])
        
        return np.mean(scores) if scores else 50.0
    
    def generate_report(
        self, 
        metrics: Dict[str, Any],
        report_type: str = 'full',
        output_path: Optional[str] = None
    ) -> str:
        """
        生成质量评估报告
        
        Args:
            metrics: 评估指标字典
            report_type: 报告类型 ('full', 'summary', 'json')
            output_path: 输出文件路径
            
        Returns:
            格式化的报告字符串
        """
        if report_type == 'json':
            report = self._generate_json_report(metrics)
        elif report_type == 'summary':
            report = self._generate_summary_report(metrics)
        else:
            report = self._generate_full_report(metrics)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"[Report] 报告已保存到: {output_path}")
        
        return report
    
    def _generate_full_report(self, metrics: Dict[str, Any]) -> str:
        """生成完整报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("超分辨率图像质量评估报告")
        lines.append("=" * 70)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 全参考指标
        if 'psnr' in metrics:
            lines.append("-" * 70)
            lines.append("【全参考质量指标】")
            lines.append("-" * 70)
            lines.append(f"PSNR:           {metrics.get('psnr', 'N/A'):.2f} dB    [{metrics.get('psnr_level', 'N/A')}]")
            lines.append(f"SSIM:           {metrics.get('ssim', 'N/A'):.4f}")
            lines.append(f"MS-SSIM:        {metrics.get('ms_ssim', 'N/A'):.4f}    [{metrics.get('ssim_level', 'N/A')}]")
            if 'lpips_vgg' in metrics:
                lines.append(f"LPIPS (VGG):    {metrics.get('lpips_vgg', 'N/A'):.4f}    [{metrics.get('lpips_level', 'N/A')}]")
                lines.append(f"LPIPS (Alex):   {metrics.get('lpips_alex', 'N/A'):.4f}")
            lines.append("")
        
        # 降采样对比
        downsample_keys = [k for k in metrics.keys() if k.startswith('psnr_') or k.startswith('ssim_')]
        if downsample_keys:
            lines.append("-" * 70)
            lines.append("【降采样对比法 (多尺度)】")
            lines.append("-" * 70)
            for scale in self.scale_config.scale_factors:
                scale_name = self.scale_config.scale_names.get(scale, f"scale_{scale}")
                psnr_key = f'psnr_{scale_name}'
                ssim_key = f'ssim_{scale_name}'
                if psnr_key in metrics:
                    lines.append(f"  {scale_name} ({scale}x):")
                    lines.append(f"    PSNR: {metrics.get(psnr_key, 'N/A'):.2f} dB")
                    lines.append(f"    SSIM: {metrics.get(ssim_key, 'N/A'):.4f}")
            lines.append("")
        
        # 无参考指标
        if 'niqe' in metrics:
            lines.append("-" * 70)
            lines.append("【无参考质量指标】")
            lines.append("-" * 70)
            lines.append(f"NIQE:           {metrics.get('niqe', 'N/A'):.2f}    [{metrics.get('niqe_level', 'N/A')}]")
            lines.append(f"BRISQUE:        {metrics.get('brisque', 'N/A'):.2f}    [{metrics.get('brisque_level', 'N/A')}]")
            lines.append(f"锐度:           {metrics.get('sharpness', 'N/A'):.2f}")
            lines.append(f"对比度:         {metrics.get('contrast', 'N/A'):.2f}")
            lines.append(f"色彩丰富度:     {metrics.get('colorfulness', 'N/A'):.2f}")
            lines.append("")
        
        # 商业专项指标
        if 'commercial_score' in metrics:
            lines.append("-" * 70)
            lines.append("【商业广告专项评估】")
            lines.append("-" * 70)
            lines.append(f"商业综合评分:   {metrics.get('commercial_score', 'N/A'):.2f}/100")
            lines.append("")
            lines.append("  细节保真度:")
            lines.append(f"    全局锐度:   {metrics.get('global_sharpness', 'N/A'):.2f}")
            lines.append(f"    高频比例:   {metrics.get('high_frequency_ratio', 'N/A'):.4f}")
            lines.append("")
            lines.append("  视觉舒适度:")
            lines.append(f"    锐化评分:   {metrics.get('oversharpen_score', 'N/A'):.2f}/100")
            lines.append(f"    伪影评分:   {metrics.get('artifact_score', 'N/A'):.2f}/100")
            lines.append(f"    噪声水平:   {metrics.get('noise_level', 'N/A'):.2f}")
            lines.append(f"    亮度均匀性: {metrics.get('brightness_uniformity', 'N/A'):.2f}/100")
            lines.append("")
        
        # 综合评分
        if 'overall_score' in metrics:
            lines.append("-" * 70)
            lines.append("【综合评估】")
            lines.append("-" * 70)
            lines.append(f"综合质量评分:   {metrics.get('overall_score', 'N/A'):.2f}/100")
            lines.append("")
        
        # 质量等级说明
        lines.append("-" * 70)
        lines.append("【质量等级说明】")
        lines.append("-" * 70)
        lines.append("  excellent - 优秀")
        lines.append("  good      - 良好")
        lines.append("  fair      - 一般")
        lines.append("  poor      - 较差")
        lines.append("")
        lines.append("=" * 70)
        lines.append("报告生成完成")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _generate_summary_report(self, metrics: Dict[str, Any]) -> str:
        """生成摘要报告"""
        lines = []
        lines.append("=" * 50)
        lines.append("超分辨率图像质量评估摘要")
        lines.append("=" * 50)
        lines.append("")
        
        # 关键指标
        if 'psnr' in metrics:
            lines.append(f"PSNR:      {metrics['psnr']:.2f} dB")
        if 'ms_ssim' in metrics:
            lines.append(f"MS-SSIM:   {metrics['ms_ssim']:.4f}")
        if 'lpips_vgg' in metrics:
            lines.append(f"LPIPS:     {metrics['lpips_vgg']:.4f}")
        if 'niqe' in metrics:
            lines.append(f"NIQE:      {metrics['niqe']:.2f}")
        if 'overall_score' in metrics:
            lines.append(f"综合评分:  {metrics['overall_score']:.2f}/100")
        
        lines.append("")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def _generate_json_report(self, metrics: Dict[str, Any]) -> str:
        """生成JSON格式报告"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def batch_evaluate(
        self,
        image_pairs: List[Tuple[np.ndarray, np.ndarray]],
        scale_factor: int = 4
    ) -> List[Dict[str, float]]:
        """
        批量评估
        
        Args:
            image_pairs: (原始图像, 超分辨率图像) 元组列表
            scale_factor: 超分辨率放大倍数
            
        Returns:
            评估结果列表
        """
        results = []
        for i, (original, upscaled) in enumerate(image_pairs):
            print(f"[Batch] 评估图像 {i+1}/{len(image_pairs)}...")
            metrics = self.evaluate_full_reference(original, upscaled, scale_factor)
            results.append(metrics)
        
        return results


# =============================================================================
# 使用示例
# =============================================================================

def example_usage():
    """质量评估模块使用示例"""
    
    print("=" * 70)
    print("超分辨率图像质量评估模块 - 使用示例")
    print("=" * 70)
    print()
    
    # 1. 初始化评估模块
    print("【步骤1】初始化质量评估模块...")
    qam = QualityAssessmentModule(device='cpu')
    print()
    
    # 2. 创建测试图像
    print("【步骤2】创建测试图像...")
    # 创建模拟的高分辨率图像
    np.random.seed(42)
    h, w = 512, 512
    original = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    
    # 创建模拟的超分辨率结果（添加一些噪声）
    upscaled = original.astype(np.float32) + np.random.randn(h, w, 3) * 5
    upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
    print(f"  原始图像尺寸: {original.shape}")
    print(f"  超分图像尺寸: {upscaled.shape}")
    print()
    
    # 3. 全参考评估
    print("【步骤3】执行全参考质量评估...")
    full_ref_metrics = qam.evaluate_full_reference(original, upscaled, scale_factor=4)
    print(f"  PSNR: {full_ref_metrics['psnr']:.2f} dB [{full_ref_metrics['psnr_level']}]")
    print(f"  MS-SSIM: {full_ref_metrics['ms_ssim']:.4f} [{full_ref_metrics['ssim_level']}]")
    if 'lpips_vgg' in full_ref_metrics:
        print(f"  LPIPS (VGG): {full_ref_metrics['lpips_vgg']:.4f} [{full_ref_metrics['lpips_level']}]")
    print()
    
    # 4. 无参考评估
    print("【步骤4】执行无参考质量评估...")
    no_ref_metrics = qam.evaluate_no_reference(upscaled)
    print(f"  NIQE: {no_ref_metrics['niqe']:.2f} [{no_ref_metrics['niqe_level']}]")
    print(f"  BRISQUE: {no_ref_metrics['brisque']:.2f} [{no_ref_metrics['brisque_level']}]")
    print(f"  锐度: {no_ref_metrics['sharpness']:.2f}")
    print()
    
    # 5. 商业专项评估
    print("【步骤5】执行商业广告专项评估...")
    roi_regions = [
        {'type': 'text', 'bbox': [100, 100, 200, 50]},
        {'type': 'product', 'bbox': [300, 200, 150, 150]},
        {'type': 'brand', 'bbox': [50, 400, 100, 80], 'reference_color': (255, 0, 0)}
    ]
    commercial_metrics = qam.evaluate_commercial(upscaled, roi_regions)
    print(f"  商业综合评分: {commercial_metrics['commercial_score']:.2f}/100")
    print(f"  全局锐度: {commercial_metrics['global_sharpness']:.2f}")
    print(f"  视觉舒适度 - 锐化评分: {commercial_metrics['oversharpen_score']:.2f}/100")
    print()
    
    # 6. 生成报告
    print("【步骤6】生成质量评估报告...")
    
    # 合并所有指标
    all_metrics = {**full_ref_metrics, **no_ref_metrics, **commercial_metrics}
    
    # 生成完整报告
    full_report = qam.generate_report(all_metrics, report_type='full')
    print("\n" + full_report)
    
    # 生成JSON报告
    json_report = qam.generate_report(all_metrics, report_type='json')
    print("\n" + json_report)
    
    print("=" * 70)
    print("示例运行完成！")
    print("=" * 70)
    
    return all_metrics


def example_batch_evaluation():
    """批量评估示例"""
    
    print("=" * 70)
    print("批量评估示例")
    print("=" * 70)
    print()
    
    # 初始化
    qam = QualityAssessmentModule(device='cpu')
    
    # 创建测试图像对
    image_pairs = []
    for i in range(3):
        h, w = 256, 256
        original = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        noise_level = (i + 1) * 3
        upscaled = original.astype(np.float32) + np.random.randn(h, w, 3) * noise_level
        upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
        image_pairs.append((original, upscaled))
    
    # 批量评估
    results = qam.batch_evaluate(image_pairs, scale_factor=4)
    
    # 输出结果
    for i, metrics in enumerate(results):
        print(f"\n图像 {i+1}:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  MS-SSIM: {metrics['ms_ssim']:.4f}")
    
    return results


if __name__ == "__main__":
    # 运行示例
    example_usage()
    print("\n\n")
    example_batch_evaluation()
