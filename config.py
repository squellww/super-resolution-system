#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超高分辨率图像生成系�?- 配置文件
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class APIConfig:
    """API配置"""
    # 火山引擎认证
    volc_ak: str = field(default_factory=lambda: os.getenv('VOLC_AK', ''))
    volc_sk: str = field(default_factory=lambda: os.getenv('VOLC_SK', ''))
    volc_region: str = 'cn-beijing'
    
    # Seedream 4.0 API
    seedream_endpoint: str = 'https://operator.las.cn-beijing.volces.com/api/v1/online/images/generations'
    seedream_model: str = 'doubao-seedream-4-0-250828'
    
    # veImageX API
    veimagex_endpoint: str = 'https://imagex.volcengineapi.com'
    
    # 请求配置
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class TilingConfig:
    """分块配置"""
    # 分块尺寸
    block_size: int = 2048  # 输入块尺�?    output_block_size: int = 4096  # 输出块尺寸（Seedream限制�?    
    # 重叠设置
    overlap_ratio: float = 0.2  # 20%重叠
    min_overlap_ratio: float = 0.1
    max_overlap_ratio: float = 0.3
    
    # 边缘填充
    padding_mode: str = 'mirror'  # mirror, replicate, reflect, constant
    
    # 内容感知
    enable_content_aware: bool = True
    face_protection_distance: float = 0.5  # 人脸保护距离（倍脸宽）
    
    # 缓存
    cache_dir: str = './cache'
    enable_l1_cache: bool = True
    enable_l2_cache: bool = True
    l1_cache_size: int = 50  # 内存缓存数量


@dataclass
class SuperResolutionConfig:
    """超分辨率配置"""
    # 目标分辨�?    target_resolution: str = '100MP'  # 100MP, 150MP, 200MP
    
    # Seedream参数
    seedream_strength: float = 0.5  # 0.0-1.0
    seedream_steps: int = 50  # 1-100
    seedream_sizes: List[str] = field(default_factory=lambda: [
        '1024x1024', '2048x2048', '4096x4096'
    ])
    
    # veImageX参数
    veimagex_template: str = 'system_workflow_ai_super_resolution'
    
    # 混合策略
    enable_hybrid: bool = False
    hybrid_stages: List[Dict] = field(default_factory=lambda: [
        {'engine': 'veimagex', 'scale': 2.0},
        {'engine': 'seedream', 'scale': 2.0},
        {'engine': 'veimagex', 'scale': 1.0}
    ])
    
    # Prompt模板
    default_category: str = 'general'
    prompt_templates: Dict[str, Dict] = field(default_factory=lambda: {
        'beauty': {
            'subject': '高端化妆品商业摄影，柔光棚拍',
            'style': '8K超高清，细腻肤质，专业广告品质，无噪�?,
            'quality': '锐利边缘，精确色彩还原，印刷级精�?,
            'negative': '模糊，变形，多余元素，色彩偏移，压缩伪影'
        },
        '3c': {
            'subject': '精密数码产品摄影，科技感十�?,
            'style': '金属光泽，精密工艺，未来感设�?,
            'quality': '超高清细节，材质真实感，专业灯光',
            'negative': '模糊，反光过曝，塑料感，低质�?
        },
        'food': {
            'subject': '美食摄影，新鲜诱�?,
            'style': '色彩饱和，质感细腻，食欲感强',
            'quality': '清晰纹理，自然光泽，专业布光',
            'negative': '暗淡，模糊，不新鲜，色彩失真'
        },
        'fashion': {
            'subject': '时尚服装摄影，高端质�?,
            'style': '面料纹理清晰，剪裁精致，高级�?,
            'quality': '细节丰富，色彩准确，专业品质',
            'negative': '褶皱，色差，模糊，廉价感'
        },
        'jewelry': {
            'subject': '珠宝首饰摄影，奢华精�?,
            'style': '璀璨光泽，精细工艺，高贵典�?,
            'quality': '反射清晰，切割精准，质感真实',
            'negative': '模糊，反光混乱，塑料感，低品�?
        },
        'furniture': {
            'subject': '家具产品摄影，品质生�?,
            'style': '材质真实，设计精美，空间感强',
            'quality': '纹理清晰，色彩自然，专业灯光',
            'negative': '变形，色差，模糊，廉价感'
        },
        'automotive': {
            'subject': '汽车摄影，动感流�?,
            'style': '金属漆质感，光影效果，高端大�?,
            'quality': '细节锐利，反射真实，专业品质',
            'negative': '模糊，反光过曝，塑料感，低质�?
        },
        'general': {
            'subject': '高品质商业摄�?,
            'style': '8K超高清，专业广告品质',
            'quality': '锐利边缘，精确色彩，印刷级精�?,
            'negative': '模糊，变形，色彩偏移，压缩伪�?
        }
    })


@dataclass
class BlendingConfig:
    """融合配置"""
    # 融合算法
    method: str = 'laplacian'  # laplacian, poisson, weighted
    
    # 金字塔参�?    num_pyramid_levels: int = 6
    
    # 权重函数
    weight_function: str = 'cosine'  # linear, cosine, sigmoid
    
    # 质量控制
    seam_detection_threshold: float = 0.95
    enable_color_correction: bool = True
    
    # 泊松融合参数
    poisson_mode: str = 'NORMAL'  # NORMAL, MIXED, MONOCHROME_TRANSFER


@dataclass
class SchedulerConfig:
    """调度器配�?""
    # Agent集群
    max_agents: int = 100
    max_concurrent: int = 60
    
    # 负载均衡
    enable_load_balancing: bool = True
    weight_factors: Dict[str, float] = field(default_factory=lambda: {
        'queue_depth': 0.4,
        'avg_processing_time': 0.3,
        'network_latency': 0.3
    })
    
    # 动态扩缩容
    enable_auto_scaling: bool = True
    scale_up_threshold: int = 50  # 队列深度阈�?    scale_down_threshold: int = 10
    min_agents: int = 10
    max_agents_limit: int = 100
    
    # 故障恢复
    max_retries: int = 3
    retry_delays: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    enable_degradation: bool = True


@dataclass
class QualityAssessmentConfig:
    """质量评估配置"""
    # 启用评估
    enable_qa: bool = True
    
    # 设备
    device: str = 'cpu'  # cpu, cuda
    
    # 评估指标阈�?    psnr_threshold: float = 35.0
    ssim_threshold: float = 0.98
    lpips_threshold: float = 0.05
    niqe_threshold: float = 3.0
    brisque_threshold: float = 20.0
    
    # 多尺度对�?    scales: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.4])
    
    # 商业评估
    enable_commercial_eval: bool = True
    text_clarity_weight: float = 0.3
    color_accuracy_weight: float = 0.4
    visual_comfort_weight: float = 0.3


@dataclass
class WebUIConfig:
    """WebUI配置"""
    # 基本设置
    page_title: str = '超高分辨率图像生成系�?
    page_icon: str = '🖼�?
    layout: str = 'wide'
    
    # 上传限制
    max_upload_size: int = 500  # MB
    supported_formats: List[str] = field(default_factory=lambda: [
        'jpg', 'jpeg', 'png', 'tiff', 'tif', 'raw', 'cr2', 'nef', 'arw'
    ])
    
    # 输出设置
    default_output_format: str = 'TIFF'
    output_formats: List[str] = field(default_factory=lambda: [
        'TIFF', 'PNG', 'JPEG', 'JXL'
    ])
    default_quality: int = 95
    
    # 预设分辨�?    resolution_presets: Dict[str, tuple] = field(default_factory=lambda: {
        '100MP (12245×8163)': (12245, 8163),
        '150MP (15000×10000)': (15000, 10000),
        '200MP (17320×11547)': (17320, 11547),
    })


@dataclass
class SystemConfig:
    """系统整体配置"""
    api: APIConfig = field(default_factory=APIConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    super_resolution: SuperResolutionConfig = field(default_factory=SuperResolutionConfig)
    blending: BlendingConfig = field(default_factory=BlendingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    quality: QualityAssessmentConfig = field(default_factory=QualityAssessmentConfig)
    webui: WebUIConfig = field(default_factory=WebUIConfig)
    
    # 日志
    log_level: str = 'INFO'
    log_file: str = 'super_resolution.log'
    
    # 输出
    output_dir: str = './output'
    temp_dir: str = './temp'
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """从环境变量加载配�?""
        config = cls()
        
        # API配置
        config.api.volc_ak = os.getenv('VOLC_AK', config.api.volc_ak)
        config.api.volc_sk = os.getenv('VOLC_SK', config.api.volc_sk)
        config.api.volc_region = os.getenv('VOLC_REGION', config.api.volc_region)
        
        # 其他环境变量...
        config.tiling.block_size = int(os.getenv('BLOCK_SIZE', config.tiling.block_size))
        config.tiling.overlap_ratio = float(os.getenv('OVERLAP_RATIO', config.tiling.overlap_ratio))
        config.super_resolution.target_resolution = os.getenv('TARGET_RESOLUTION', config.super_resolution.target_resolution)
        config.scheduler.max_concurrent = int(os.getenv('MAX_CONCURRENT', config.scheduler.max_concurrent))
        config.quality.device = os.getenv('QA_DEVICE', config.quality.device)
        
        return config


# 全局配置实例
config = SystemConfig.from_env()

