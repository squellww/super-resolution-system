#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARK API 图像生成模块
用于调用火山引擎ARK API进行图像生成
"""

import os
import base64
import json
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ARKImageConfig:
    """ARK图像生成配置"""
    api_key: str = 'sk-xIr6z0QlYiu498lwe406xbeuxXeIE6Mp6neFxkhABigECvQ9'
    endpoint: str = 'https://ark.cn-beijing.volces.com/api/v3/images/generations'
    model: str = 'ep-20260228221135-66v8k'
    size: str = '2K'
    watermark: bool = True


class ARKImageGenerator:
    """ARK图像生成器"""
    
    def __init__(self, config: Optional[ARKImageConfig] = None):
        self.config = config or ARKImageConfig()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(
        self,
        prompt: str,
        size: Optional[str] = None,
        watermark: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        生成图像
        
        Args:
            prompt: 图像描述提示词
            size: 图像尺寸 (如 '2K', '1024x1024')
            watermark: 是否添加水印
            
        Returns:
            API响应结果
        """
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.config.api_key}'
        }
        
        payload = {
            'model': self.config.model,
            'prompt': prompt,
            'sequential_image_generation': 'disabled',
            'response_format': 'url',
            'size': size or self.config.size,
            'stream': False,
            'watermark': watermark if watermark is not None else self.config.watermark
        }
        
        async with self.session.post(
            self.config.endpoint,
            headers=headers,
            json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()


# 便捷函数
async def generate_image(prompt: str, **kwargs) -> Dict[str, Any]:
    """快速生成图像"""
    async with ARKImageGenerator() as generator:
        return await generator.generate(prompt, **kwargs)


if __name__ == '__main__':
    # 测试
    import asyncio
    result = asyncio.run(generate_image(
        'A beautiful landscape with mountains and sunset'
    ))
    print(json.dumps(result, indent=2, ensure_ascii=False))
