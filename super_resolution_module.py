#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超分辨率处理模块 (Super-Resolution Module)

本模块实现超高分辨率图像生成系统的核心超分功能，包括：
1. Seedream 4.0 API集成 - 基于扩散模型的智能超分
2. veImageX备选方案 - 轻量化CNN快速超分
3. 混合策略 - 多级串联超分优化
4. Prompt模板系统 - 行业场景化提示词构建
5. 风格统一化机制 - 确定性种子与色彩锁定

作者: AI Image Generation Engineer
版本: 1.0.0
"""

import asyncio
import base64
import hashlib
import hmac
import io
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode, urlparse

import httpx
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UpscaleProvider(Enum):
    """超分服务提供商枚举"""
    SEEDREAM = "seedream"
    VEIMAGEX = "veimagex"
    HYBRID = "hybrid"


class VeImageXTemplate(Enum):
    """veImageX超分模板ID"""
    AI_SUPER_RESOLUTION = "system_workflow_ai_super_resolution"  # 2倍AI超分
    STANDARD_SR = "system_workflow_sr"  # 1.5x-4x标准超分
    FAST_SR = "system_workflow_fast_sr"  # 快速超分


@dataclass
class UpscaleConfig:
    """超分配置数据类
    
    Attributes:
        provider: 超分服务提供商
        target_scale: 目标放大倍数
        strength: Seedream强度参数 (0.0-1.0)
        num_inference_steps: 推理步数
        seed: 随机种子 (None表示随机)
        quality: 输出质量 (1-100)
        preserve_style: 是否保持风格一致
    """
    provider: UpscaleProvider = UpscaleProvider.SEEDREAM
    target_scale: float = 2.0
    strength: float = 0.5
    num_inference_steps: int = 30
    seed: Optional[int] = None
    quality: int = 95
    preserve_style: bool = True


@dataclass
class SuperResolutionResult:
    """超分结果数据类
    
    Attributes:
        image: 超分后的PIL Image对象
        original_size: 原始图像尺寸 (width, height)
        upscaled_size: 超分后尺寸 (width, height)
        scale_factor: 实际放大倍数
        provider: 使用的超分提供商
        processing_time: 处理时间(秒)
        metadata: 附加元数据
    """
    image: Image.Image
    original_size: Tuple[int, int]
    upscaled_size: Tuple[int, int]
    scale_factor: float
    provider: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptTemplateManager:
    """Prompt模板管理器
    
    提供行业场景化的超分Prompt模板，支持结构化模板构建。
    """
    
    # 行业场景Prompt模板字典
    TEMPLATES: Dict[str, Dict[str, str]] = {
        "beauty": {
            "name": "美妆护肤",
            "subject": "premium beauty product, skincare cosmetic, elegant packaging",
            "style": "soft lighting, clean background, professional product photography",
            "quality": "8K ultra HD, photorealistic, sharp details, vibrant colors",
            "negative": "blurry, low quality, distorted, oversaturated, artificial look"
        },
        "3c": {
            "name": "3C数码",
            "subject": "modern electronic device, sleek gadget, premium technology product",
            "style": "minimalist design, studio lighting, reflective surface, tech aesthetic",
            "quality": "8K ultra HD, crystal clear, precise edges, professional rendering",
            "negative": "noise, grain, blur, low resolution, cheap look, plastic texture"
        },
        "food": {
            "name": "食品饮料",
            "subject": "delicious gourmet food, fresh ingredients, appetizing presentation",
            "style": "natural lighting, food photography style, rich textures, steam effect",
            "quality": "8K ultra HD, mouth-watering detail, vivid colors, professional food shot",
            "negative": "unappetizing, dull colors, blurry, overprocessed, artificial"
        },
        "fashion": {
            "name": "服装时尚",
            "subject": "elegant fashion item, premium fabric, stylish clothing",
            "style": "high-end fashion photography, soft bokeh, model aesthetic",
            "quality": "8K ultra HD, fabric texture detail, true color reproduction",
            "negative": "wrinkled, cheap fabric look, distorted pattern, color cast"
        },
        "jewelry": {
            "name": "珠宝首饰",
            "subject": "luxury jewelry, precious gemstone, fine craftsmanship",
            "style": "macro photography, sparkle effect, elegant composition",
            "quality": "8K ultra HD, brilliant cut detail, realistic metal reflection",
            "negative": "dull, cloudy gemstone, cheap metal look, inaccurate color"
        },
        "furniture": {
            "name": "家居家具",
            "subject": "modern furniture, interior design piece, home decor",
            "style": "lifestyle photography, natural setting, warm atmosphere",
            "quality": "8K ultra HD, material texture, realistic wood grain, fabric detail",
            "negative": "cluttered background, harsh lighting, distorted perspective"
        },
        "automotive": {
            "name": "汽车配件",
            "subject": "automotive part, car accessory, vehicle component",
            "style": "automotive photography, dynamic angle, metallic finish",
            "quality": "8K ultra HD, precise engineering detail, realistic metal surface",
            "negative": "scratches, dust, poor lighting, inaccurate proportions"
        },
        "general": {
            "name": "通用商品",
            "subject": "high quality product, commercial item, retail merchandise",
            "style": "professional product photography, clean composition, neutral background",
            "quality": "8K ultra HD, sharp focus, accurate colors, commercial quality",
            "negative": "amateur photo, poor lighting, distracting background, blur"
        }
    }
    
    @classmethod
    def get_template(cls, category: str) -> Dict[str, str]:
        """获取指定类别的Prompt模板
        
        Args:
            category: 行业类别代码 (beauty, 3c, food, fashion, jewelry, furniture, automotive, general)
            
        Returns:
            Prompt模板字典
        """
        return cls.TEMPLATES.get(category, cls.TEMPLATES["general"])
    
    @classmethod
    def build_prompt(
        cls,
        category: str,
        custom_desc: Optional[str] = None,
        include_negative: bool = False
    ) -> str:
        """构建完整的超分Prompt
        
        Args:
            category: 行业类别
            custom_desc: 自定义主体描述（可选）
            include_negative: 是否包含负面提示
            
        Returns:
            完整的Prompt字符串
        """
        template = cls.get_template(category)
        
        # 构建主体描述
        if custom_desc:
            subject = f"{custom_desc}, {template['subject']}"
        else:
            subject = template['subject']
        
        # 组合Prompt
        prompt_parts = [
            subject,
            template['style'],
            template['quality']
        ]
        
        prompt = ", ".join(prompt_parts)
        
        if include_negative:
            prompt = f"{prompt}###{template['negative']}"
        
        return prompt
    
    @classmethod
    def list_categories(cls) -> List[str]:
        """列出所有可用的行业类别"""
        return list(cls.TEMPLATES.keys())


class SuperResolutionModule:
    """超分辨率处理模块
    
    实现超高分辨率图像生成的核心功能，集成Seedream 4.0和veImageX两种超分方案，
    支持混合策略和Prompt模板系统。
    
    Attributes:
        ak: 火山引擎Access Key
        sk: 火山引擎Secret Key
        region: 服务区域 (默认cn-beijing)
        seedream_endpoint: Seedream API端点
        veimagex_endpoint: veImageX API端点
    
    Example:
        >>> sr_module = SuperResolutionModule(ak="your_ak", sk="your_sk")
        >>> result = await sr_module.upscale_seedream(image, prompt="enhance quality")
        >>> result.image.save("upscaled.png")
    """
    
    # API端点配置
    SEEDREAM_ENDPOINT = "https://operator.las.cn-beijing.volces.com/api/v1/online/images/generations"
    VEIMAGEX_ENDPOINT = "https://imagex.volcengineapi.com"
    
    # 服务配置
    DEFAULT_REGION = "cn-beijing"
    DEFAULT_SERVICE = "las"
    DEFAULT_VERSION = "2024-05-01"
    
    # 重试配置
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # 基础延迟(秒)
    MAX_DELAY = 8.0   # 最大延迟(秒)
    
    # 图像尺寸映射
    SIZE_MAP = {
        1.5: "1536x1536",
        2.0: "2048x2048",
        3.0: "3072x3072",
        4.0: "4096x4096"
    }
    
    def __init__(
        self,
        ak: str,
        sk: str,
        region: str = "cn-beijing",
        timeout: float = 120.0
    ):
        """初始化超分辨率模块
        
        Args:
            ak: 火山引擎Access Key
            sk: 火山引擎Secret Key
            region: 服务区域
            timeout: API请求超时时间(秒)
        """
        self.ak = ak
        self.sk = sk
        self.region = region
        self.timeout = timeout
        self.prompt_manager = PromptTemplateManager()
        
        # 初始化HTTP客户端
        self._client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"SuperResolutionModule initialized (region: {region})")
    
    @property
    async def client(self) -> httpx.AsyncClient:
        """获取或创建HTTP客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
        return self._client
    
    async def close(self):
        """关闭HTTP客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    def generate_signature(
        self,
        method: str,
        uri: str,
        query_params: Dict[str, str],
        headers: Dict[str, str],
        body: str = ""
    ) -> Tuple[str, str]:
        """生成HMAC-SHA256签名
        
        实现火山引擎AK/SK签名认证机制，符合官方签名规范。
        
        Args:
            method: HTTP方法 (GET/POST等)
            uri: 请求URI路径
            query_params: URL查询参数
            headers: 请求头字典
            body: 请求体字符串
            
        Returns:
            Tuple[签名字符串, X-Date时间戳]
            
        Example:
            >>> signature, x_date = sr_module.generate_signature(
            ...     "POST", "/api/v1/images", {}, {}, "{}"
            ... )
        """
        # 生成X-Date时间戳 (ISO8601格式)
        x_date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        
        # 构建规范请求
        # 1. HTTP方法
        canonical_method = method.upper()
        
        # 2. URI路径
        canonical_uri = uri if uri else "/"
        
        # 3. 规范查询字符串 (按key排序)
        canonical_query = ""
        if query_params:
            sorted_params = sorted(query_params.items())
            canonical_query = urlencode(sorted_params, safe="~")
        
        # 4. 规范头 (只包含host和content-type)
        canonical_headers = ""
        signed_headers = "host;x-date"
        
        host = headers.get("host", f"operator.las.{self.region}.volces.com")
        canonical_headers = f"host:{host}\nx-date:{x_date}\n"
        
        # 5. 请求体哈希 (HexEncode(Hash(body)))
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        
        # 组合规范请求
        canonical_request = (
            f"{canonical_method}\n"
            f"{canonical_uri}\n"
            f"{canonical_query}\n"
            f"{canonical_headers}\n"
            f"{signed_headers}\n"
            f"{body_hash}"
        )
        
        # 构建待签名字符串
        credential_scope = f"{x_date[:8]}/{self.region}/{self.DEFAULT_SERVICE}/request"
        canonical_request_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        
        string_to_sign = (
            f"HMAC-SHA256\n"
            f"{x_date}\n"
            f"{credential_scope}\n"
            f"{canonical_request_hash}"
        )
        
        # 计算签名
        # 1. 派生签名密钥
        k_date = hmac.new(
            f"TC3{self.sk}".encode("utf-8"),
            x_date[:8].encode("utf-8"),
            hashlib.sha256
        ).digest()
        
        k_region = hmac.new(
            k_date,
            self.region.encode("utf-8"),
            hashlib.sha256
        ).digest()
        
        k_service = hmac.new(
            k_region,
            self.DEFAULT_SERVICE.encode("utf-8"),
            hashlib.sha256
        ).digest()
        
        k_signing = hmac.new(
            k_service,
            b"request",
            hashlib.sha256
        ).digest()
        
        # 2. 计算最终签名
        signature = hmac.new(
            k_signing,
            string_to_sign.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature, x_date
    
    def _build_auth_header(
        self,
        method: str,
        uri: str,
        query_params: Dict[str, str],
        headers: Dict[str, str],
        body: str
    ) -> Dict[str, str]:
        """构建认证请求头
        
        Args:
            method: HTTP方法
            uri: 请求URI
            query_params: 查询参数
            headers: 基础请求头
            body: 请求体
            
        Returns:
            完整的请求头字典
        """
        signature, x_date = self.generate_signature(
            method, uri, query_params, headers, body
        )
        
        credential_scope = f"{x_date[:8]}/{self.region}/{self.DEFAULT_SERVICE}/request"
        
        auth_header = (
            f"HMAC-SHA256 "
            f"Credential={self.ak}/{credential_scope}, "
            f"SignedHeaders=host;x-date, "
            f"Signature={signature}"
        )
        
        return {
            "Authorization": auth_header,
            "X-Date": x_date,
            "Content-Type": "application/json",
            **headers
        }
    
    def _image_to_base64(self, image: Union[Image.Image, str, bytes]) -> str:
        """将图像转换为Base64编码
        
        Args:
            image: PIL Image对象、文件路径或字节数据
            
        Returns:
            Base64编码的图像字符串
        """
        if isinstance(image, str):
            # 文件路径
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image, bytes):
            # 字节数据
            return base64.b64encode(image).decode("utf-8")
        elif isinstance(image, Image.Image):
            # PIL Image对象
            buffer = io.BytesIO()
            image_format = image.format if image.format else "PNG"
            image.save(buffer, format=image_format)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
    
    def _deterministic_seed(self, image: Image.Image, block_id: str = "") -> int:
        """生成确定性种子
        
        基于图像内容和block_id生成确定性种子，确保风格一致性。
        
        Args:
            image: 输入图像
            block_id: 区块标识符
            
        Returns:
            确定性种子值
        """
        # 计算图像内容的哈希
        img_bytes = io.BytesIO()
        # 使用缩略图进行哈希以提高性能
        thumb = image.copy()
        thumb.thumbnail((64, 64))
        thumb.save(img_bytes, format="PNG")
        img_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()
        
        # 组合block_id和图像哈希
        seed_str = f"{block_id}:{img_hash}"
        seed_hash = hashlib.md5(seed_str.encode()).hexdigest()
        
        # 转换为整数种子 (限制在32位范围内)
        return int(seed_hash[:8], 16) % (2**31)
    
    async def retry_with_backoff(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        **kwargs
    ) -> Any:
        """指数退避重试机制
        
        实现指数退避重试策略：1s → 2s → 4s
        
        Args:
            func: 要执行的异步函数
            *args: 函数位置参数
            max_retries: 最大重试次数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            Exception: 所有重试失败后抛出最后一次异常
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    # 计算退避延迟: 2^attempt 秒
                    delay = min(self.BASE_DELAY * (2 ** attempt), self.MAX_DELAY)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        raise last_exception
    
    async def upscale_seedream(
        self,
        image: Union[Image.Image, str, bytes],
        prompt: str,
        strength: float = 0.5,
        size: str = "4096x4096",
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        preserve_original: bool = True
    ) -> SuperResolutionResult:
        """使用Seedream 4.0进行超分辨率处理
        
        Seedream 4.0是基于扩散模型的智能超分方案，支持图像到图像的生成，
        能够在保持原图结构的同时提升分辨率和细节质量。
        
        Args:
            image: 输入图像 (PIL Image/文件路径/字节数据)
            prompt: 超分提示词，描述期望的输出风格和质量
            strength: 变换强度 (0.0-1.0)，越高变化越大
            size: 目标尺寸 (如 "4096x4096")
            seed: 随机种子 (None则自动生成确定性种子)
            num_inference_steps: 推理步数 (越多质量越高但越慢)
            preserve_original: 是否保持原图结构
            
        Returns:
            SuperResolutionResult: 超分结果对象
            
        Raises:
            httpx.HTTPError: API请求失败
            ValueError: 参数验证失败
            
        Example:
            >>> result = await sr_module.upscale_seedream(
            ...     image="input.png",
            ...     prompt="high quality product photo, 8K ultra HD",
            ...     strength=0.4,
            ...     size="4096x4096"
            ... )
        """
        start_time = time.time()
        
        # 加载图像
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        else:
            img = image.copy()
        
        original_size = img.size
        
        # 验证参数
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength必须在0.0-1.0之间，当前值: {strength}")
        
        # 生成确定性种子（如果未提供）
        if seed is None:
            seed = self._deterministic_seed(img)
        
        # 转换图像为Base64
        image_base64 = self._image_to_base64(img)
        
        # 构建请求体
        request_body = {
            "model": "seedream-4.0",
            "image": image_base64,
            "prompt": prompt,
            "size": size,
            "strength": strength,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "preserve_original": preserve_original
        }
        
        body_json = json.dumps(request_body)
        
        # 构建请求头
        headers = {
            "host": f"operator.las.{self.region}.volces.com"
        }
        
        parsed_url = urlparse(self.SEEDREAM_ENDPOINT)
        auth_headers = self._build_auth_header(
            method="POST",
            uri=parsed_url.path,
            query_params={},
            headers=headers,
            body=body_json
        )
        
        # 发送请求
        client = await self.client
        
        try:
            response = await client.post(
                self.SEEDREAM_ENDPOINT,
                headers=auth_headers,
                content=body_json
            )
            response.raise_for_status()
            
            result_data = response.json()
            
            # 解析响应
            if "data" in result_data and len(result_data["data"]) > 0:
                image_data = result_data["data"][0]
                
                # 解码Base64图像
                if "b64_json" in image_data:
                    image_bytes = base64.b64decode(image_data["b64_json"])
                elif "url" in image_data:
                    # 从URL下载图像
                    img_response = await client.get(image_data["url"])
                    img_response.raise_for_status()
                    image_bytes = img_response.content
                else:
                    raise ValueError("响应中未找到图像数据")
                
                result_image = Image.open(io.BytesIO(image_bytes))
                
                processing_time = time.time() - start_time
                
                # 计算实际放大倍数
                scale_factor = (
                    result_image.width / original_size[0],
                    result_image.height / original_size[1]
                )
                
                return SuperResolutionResult(
                    image=result_image,
                    original_size=original_size,
                    upscaled_size=result_image.size,
                    scale_factor=scale_factor[0],
                    provider="seedream-4.0",
                    processing_time=processing_time,
                    metadata={
                        "seed": seed,
                        "strength": strength,
                        "steps": num_inference_steps,
                        "prompt": prompt
                    }
                )
            else:
                raise ValueError(f"API响应格式错误: {result_data}")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Seedream API HTTP错误: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Seedream超分失败: {e}")
            raise
    
    async def upscale_veimagex(
        self,
        image: Union[Image.Image, str, bytes],
        template_id: str = VeImageXTemplate.AI_SUPER_RESOLUTION.value,
        scale_factor: float = 2.0
    ) -> SuperResolutionResult:
        """使用veImageX进行超分辨率处理
        
        veImageX提供轻量级CNN超分方案，速度优先，适合快速预览和批量处理。
        
        Args:
            image: 输入图像
            template_id: 超分模板ID
            scale_factor: 放大倍数 (1.5-4.0)
            
        Returns:
            SuperResolutionResult: 超分结果对象
            
        Raises:
            httpx.HTTPError: API请求失败
            ValueError: 参数验证失败
        """
        start_time = time.time()
        
        # 加载图像
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        else:
            img = image.copy()
        
        original_size = img.size
        
        # 转换图像为Base64
        image_base64 = self._image_to_base64(img)
        
        # 构建请求体
        request_body = {
            "TemplateId": template_id,
            "Image": image_base64,
            "ScaleFactor": scale_factor
        }
        
        body_json = json.dumps(request_body)
        
        # 构建请求头
        headers = {
            "host": "imagex.volcengineapi.com"
        }
        
        auth_headers = self._build_auth_header(
            method="POST",
            uri="/",
            query_params={"Action": "GetImageEnhancement", "Version": "2018-08-01"},
            headers=headers,
            body=body_json
        )
        
        # 发送请求
        client = await self.client
        endpoint = f"{self.VEIMAGEX_ENDPOINT}/?Action=GetImageEnhancement&Version=2018-08-01"
        
        try:
            response = await client.post(
                endpoint,
                headers=auth_headers,
                content=body_json
            )
            response.raise_for_status()
            
            result_data = response.json()
            
            # 解析响应
            if "Result" in result_data:
                result = result_data["Result"]
                
                if "Image" in result:
                    # 解码Base64图像
                    image_bytes = base64.b64decode(result["Image"])
                    result_image = Image.open(io.BytesIO(image_bytes))
                    
                    processing_time = time.time() - start_time
                    
                    return SuperResolutionResult(
                        image=result_image,
                        original_size=original_size,
                        upscaled_size=result_image.size,
                        scale_factor=scale_factor,
                        provider=f"veimagex-{template_id}",
                        processing_time=processing_time,
                        metadata={
                            "template_id": template_id,
                            "scale_factor": scale_factor
                        }
                    )
                elif "ImageUrl" in result:
                    # 从URL下载
                    img_response = await client.get(result["ImageUrl"])
                    img_response.raise_for_status()
                    result_image = Image.open(io.BytesIO(img_response.content))
                    
                    processing_time = time.time() - start_time
                    
                    return SuperResolutionResult(
                        image=result_image,
                        original_size=original_size,
                        upscaled_size=result_image.size,
                        scale_factor=scale_factor,
                        provider=f"veimagex-{template_id}",
                        processing_time=processing_time,
                        metadata={
                            "template_id": template_id,
                            "scale_factor": scale_factor,
                            "url": result["ImageUrl"]
                        }
                    )
                else:
                    raise ValueError(f"veImageX响应中未找到图像数据: {result}")
            else:
                raise ValueError(f"veImageX API响应格式错误: {result_data}")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"veImageX API HTTP错误: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"veImageX超分失败: {e}")
            raise
    
    async def hybrid_upscale(
        self,
        image: Union[Image.Image, str, bytes],
        target_scale: float = 4.0,
        category: str = "general",
        custom_desc: Optional[str] = None,
        config: Optional[UpscaleConfig] = None
    ) -> SuperResolutionResult:
        """混合策略超分辨率处理
        
        实现三级串联超分策略：
        1. 预处理: veImageX 2×快速超分
        2. 主超分: Seedream 2×-4×智能超分
        3. 后处理: veImageX优化锐化
        
        Args:
            image: 输入图像
            target_scale: 目标放大倍数
            category: 行业类别 (用于Prompt模板)
            custom_desc: 自定义描述
            config: 超分配置 (可选)
            
        Returns:
            SuperResolutionResult: 超分结果对象
        """
        start_time = time.time()
        
        # 加载图像
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        else:
            img = image.copy()
        
        original_size = img.size
        current_image = img
        processing_history = []
        
        try:
            # 阶段1: 预处理 - veImageX 2×快速超分
            if target_scale >= 2.0:
                logger.info("阶段1: veImageX预处理 (2×)")
                
                try:
                    result = await self.retry_with_backoff(
                        self.upscale_veimagex,
                        current_image,
                        VeImageXTemplate.AI_SUPER_RESOLUTION.value,
                        2.0,
                        max_retries=self.MAX_RETRIES
                    )
                    current_image = result.image
                    processing_history.append({
                        "stage": "preprocess",
                        "provider": result.provider,
                        "scale": 2.0,
                        "time": result.processing_time
                    })
                except Exception as e:
                    logger.warning(f"预处理阶段失败，跳过: {e}")
            
            # 阶段2: 主超分 - Seedream智能超分
            remaining_scale = target_scale / 2.0 if target_scale >= 2.0 else target_scale
            
            if remaining_scale >= 1.5:
                logger.info(f"阶段2: Seedream主超分 ({remaining_scale}×)")
                
                # 构建Prompt
                prompt = self.build_prompt(category, custom_desc)
                
                # 确定目标尺寸
                if remaining_scale >= 4.0:
                    size = "4096x4096"
                elif remaining_scale >= 3.0:
                    size = "3072x3072"
                elif remaining_scale >= 2.0:
                    size = "2048x2048"
                else:
                    size = "1536x1536"
                
                # 配置参数
                strength = config.strength if config else 0.5
                steps = config.num_inference_steps if config else 30
                seed = config.seed if config else None
                
                try:
                    result = await self.retry_with_backoff(
                        self.upscale_seedream,
                        current_image,
                        prompt,
                        strength,
                        size,
                        seed,
                        steps,
                        max_retries=self.MAX_RETRIES
                    )
                    current_image = result.image
                    processing_history.append({
                        "stage": "main",
                        "provider": result.provider,
                        "scale": remaining_scale,
                        "time": result.processing_time,
                        "prompt": prompt
                    })
                except Exception as e:
                    logger.error(f"主超分阶段失败: {e}")
                    # 降级到veImageX
                    logger.info("降级到veImageX备选方案")
                    result = await self.upscale_veimagex(
                        current_image,
                        VeImageXTemplate.STANDARD_SR.value,
                        remaining_scale
                    )
                    current_image = result.image
                    processing_history.append({
                        "stage": "main_fallback",
                        "provider": result.provider,
                        "scale": remaining_scale,
                        "time": result.processing_time
                    })
            
            # 阶段3: 后处理 - veImageX优化
            logger.info("阶段3: veImageX后处理优化")
            
            try:
                result = await self.upscale_veimagex(
                    current_image,
                    VeImageXTemplate.FAST_SR.value,
                    1.0  # 保持尺寸，仅优化质量
                )
                current_image = result.image
                processing_history.append({
                    "stage": "postprocess",
                    "provider": result.provider,
                    "scale": 1.0,
                    "time": result.processing_time
                })
            except Exception as e:
                logger.warning(f"后处理阶段失败，跳过: {e}")
            
            # 计算总处理时间
            total_time = time.time() - start_time
            
            # 计算实际放大倍数
            actual_scale = (
                current_image.width / original_size[0],
                current_image.height / original_size[1]
            )
            
            return SuperResolutionResult(
                image=current_image,
                original_size=original_size,
                upscaled_size=current_image.size,
                scale_factor=actual_scale[0],
                provider="hybrid",
                processing_time=total_time,
                metadata={
                    "target_scale": target_scale,
                    "actual_scale": actual_scale,
                    "processing_history": processing_history,
                    "category": category
                }
            )
            
        except Exception as e:
            logger.error(f"混合超分策略失败: {e}")
            raise
    
    def build_prompt(
        self,
        category: str = "general",
        custom_desc: Optional[str] = None,
        include_negative: bool = False
    ) -> str:
        """构建超分Prompt
        
        使用Prompt模板系统构建结构化的超分提示词。
        
        Args:
            category: 行业类别
            custom_desc: 自定义描述
            include_negative: 是否包含负面提示
            
        Returns:
            完整的Prompt字符串
        """
        return self.prompt_manager.build_prompt(category, custom_desc, include_negative)
    
    async def upscale(
        self,
        image: Union[Image.Image, str, bytes],
        config: UpscaleConfig
    ) -> SuperResolutionResult:
        """通用超分接口
        
        根据配置自动选择超分方案。
        
        Args:
            image: 输入图像
            config: 超分配置
            
        Returns:
            SuperResolutionResult: 超分结果
        """
        if config.provider == UpscaleProvider.SEEDREAM:
            prompt = self.build_prompt("general")
            size = self.SIZE_MAP.get(config.target_scale, "4096x4096")
            
            return await self.retry_with_backoff(
                self.upscale_seedream,
                image,
                prompt,
                config.strength,
                size,
                config.seed,
                config.num_inference_steps,
                max_retries=self.MAX_RETRIES
            )
            
        elif config.provider == UpscaleProvider.VEIMAGEX:
            template = (
                VeImageXTemplate.AI_SUPER_RESOLUTION.value 
                if config.target_scale <= 2.0 
                else VeImageXTemplate.STANDARD_SR.value
            )
            
            return await self.retry_with_backoff(
                self.upscale_veimagex,
                image,
                template,
                config.target_scale,
                max_retries=self.MAX_RETRIES
            )
            
        else:  # HYBRID
            return await self.hybrid_upscale(
                image,
                config.target_scale,
                config.provider.value if isinstance(config.provider, Enum) else "general"
            )


# ==================== API调用示例 ====================

async def example_seedream_upscale():
    """Seedream 4.0超分示例"""
    # 初始化模块
    sr = SuperResolutionModule(
        ak="your_access_key",
        sk="your_secret_key",
        region="cn-beijing"
    )
    
    try:
        # 使用美妆模板进行超分
        prompt = sr.build_prompt(
            category="beauty",
            custom_desc="luxury skincare serum bottle"
        )
        
        result = await sr.upscale_seedream(
            image="input_product.jpg",
            prompt=prompt,
            strength=0.4,
            size="4096x4096",
            num_inference_steps=35
        )
        
        print(f"超分完成!")
        print(f"原始尺寸: {result.original_size}")
        print(f"超分尺寸: {result.upscaled_size}")
        print(f"放大倍数: {result.scale_factor:.2f}×")
        print(f"处理时间: {result.processing_time:.2f}秒")
        print(f"提供商: {result.provider}")
        
        # 保存结果
        result.image.save("output_seedream.png", quality=95)
        
    finally:
        await sr.close()


async def example_veimagex_upscale():
    """veImageX超分示例"""
    sr = SuperResolutionModule(
        ak="your_access_key",
        sk="your_secret_key"
    )
    
    try:
        result = await sr.upscale_veimagex(
            image="input_product.jpg",
            template_id=VeImageXTemplate.AI_SUPER_RESOLUTION.value,
            scale_factor=2.0
        )
        
        print(f"veImageX超分完成!")
        print(f"处理时间: {result.processing_time:.2f}秒")
        
        result.image.save("output_veimagex.png", quality=95)
        
    finally:
        await sr.close()


async def example_hybrid_upscale():
    """混合策略超分示例"""
    sr = SuperResolutionModule(
        ak="your_access_key",
        sk="your_secret_key"
    )
    
    try:
        # 使用混合策略进行4倍超分
        result = await sr.hybrid_upscale(
            image="input_product.jpg",
            target_scale=4.0,
            category="3c",
            custom_desc="premium wireless earbuds"
        )
        
        print(f"混合超分完成!")
        print(f"原始尺寸: {result.original_size}")
        print(f"超分尺寸: {result.upscaled_size}")
        print(f"实际放大倍数: {result.scale_factor:.2f}×")
        print(f"总处理时间: {result.processing_time:.2f}秒")
        print(f"处理历史: {result.metadata.get('processing_history', [])}")
        
        result.image.save("output_hybrid.png", quality=95)
        
    finally:
        await sr.close()


async def example_with_retry():
    """带重试机制的超分示例"""
    sr = SuperResolutionModule(
        ak="your_access_key",
        sk="your_secret_key"
    )
    
    try:
        # 使用重试机制
        result = await sr.retry_with_backoff(
            sr.upscale_seedream,
            "input_product.jpg",
            "high quality product photo",
            0.5,
            "4096x4096",
            max_retries=3
        )
        
        print(f"重试后超分成功!")
        result.image.save("output_retry.png", quality=95)
        
    except Exception as e:
        print(f"所有重试均失败: {e}")
        
    finally:
        await sr.close()


async def example_config_based_upscale():
    """基于配置的通用超分示例"""
    sr = SuperResolutionModule(
        ak="your_access_key",
        sk="your_secret_key"
    )
    
    try:
        # 创建配置
        config = UpscaleConfig(
            provider=UpscaleProvider.HYBRID,
            target_scale=4.0,
            strength=0.4,
            num_inference_steps=35,
            quality=95,
            preserve_style=True
        )
        
        # 执行超分
        result = await sr.upscale("input_product.jpg", config)
        
        print(f"配置化超分完成!")
        result.image.save("output_config.png", quality=config.quality)
        
    finally:
        await sr.close()


# 运行示例
if __name__ == "__main__":
    # 示例1: Seedream超分
    # asyncio.run(example_seedream_upscale())
    
    # 示例2: veImageX超分
    # asyncio.run(example_veimagex_upscale())
    
    # 示例3: 混合策略超分
    # asyncio.run(example_hybrid_upscale())
    
    # 示例4: 带重试机制
    # asyncio.run(example_with_retry())
    
    # 示例5: 基于配置
    # asyncio.run(example_config_based_upscale())
    
    print("超分辨率模块已加载完成")
    print(f"可用行业模板: {PromptTemplateManager.list_categories()}")
