#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超分辨率模块配置示例

使用前请复制此文件为 config.py 并填入您的实际配置
"""

# 火山引擎认证配置
VOLCANO_ENGINE_CONFIG = {
    "ak": "your_access_key_here",  # 替换为您的Access Key
    "sk": "your_secret_key_here",  # 替换为您的Secret Key
    "region": "cn-beijing",         # 服务区域
}

# 超分服务配置
UPSCALE_CONFIG = {
    # 默认超时时间(秒)
    "timeout": 120.0,
    
    # 重试配置
    "max_retries": 3,
    "base_delay": 1.0,  # 基础延迟(秒)
    "max_delay": 8.0,   # 最大延迟(秒)
    
    # Seedream配置
    "seedream": {
        "default_strength": 0.5,
        "default_steps": 30,
        "max_steps": 50,
    },
    
    # veImageX配置
    "veimagex": {
        "default_template": "system_workflow_ai_super_resolution",
        "fast_template": "system_workflow_fast_sr",
    }
}

# 输出质量配置
OUTPUT_CONFIG = {
    "default_quality": 95,
    "supported_formats": ["PNG", "JPEG", "WEBP"],
    "max_dimension": 8192,  # 最大输出尺寸
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
