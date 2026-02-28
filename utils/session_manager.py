"""
会话状态管理器 - Session State Manager
管理Streamlit应用的所有会话状态
"""

import streamlit as st
from typing import Any, Dict, List, Optional


# 默认会话状态配置
DEFAULT_SESSION_STATE = {
    # 页面导航
    'current_page': 'upload',
    
    # 上传相关
    'uploaded_file': None,
    'source_image': None,
    'cropped_image': None,
    'crop_region': None,
    
    # 配置参数
    'target_resolution': 100000000,
    'tile_size': 1024,
    'overlap_rate': 0.20,
    'max_tiles': 100,
    'seedream_version': 'Seedream v3.0 (推荐)',
    'fusion_algorithm': '拉普拉斯金字塔',
    'guidance_scale': 7.5,
    'num_inference_steps': 50,
    'seed': -1,
    
    # Prompt
    'prompt_text': '',
    'negative_prompt': 'blurry, low quality, distorted, deformed, ugly, duplicate, watermark, signature, text',
    
    # 处理状态
    'processing_started': False,
    'processing_complete': False,
    'processing_paused': False,
    'current_progress': 0,
    'processed_tiles': 0,
    'total_tiles': 0,
    
    # 日志
    'logs': [],
    
    # 结果
    'result_image': None,
    
    # 系统状态
    'online_agents': 12,
    'queue_depth': 3,
    
    # 高级功能
    'config_saved': False,
    'show_history': False,
    'advanced_tab': 'batch',
}


def initialize_session_state():
    """
    初始化会话状态
    确保所有必需的键都存在
    """
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_session_value(key: str, default: Any = None) -> Any:
    """
    安全地获取会话状态值
    
    Args:
        key: 状态键名
        default: 默认值
        
    Returns:
        状态值或默认值
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any):
    """
    设置会话状态值
    
    Args:
        key: 状态键名
        value: 要设置的值
    """
    st.session_state[key] = value


def clear_session_key(key: str):
    """
    清除指定的会话状态键
    
    Args:
        key: 要清除的键名
    """
    if key in st.session_state:
        del st.session_state[key]


def reset_session(keys: Optional[List[str]] = None):
    """
    重置会话状态
    
    Args:
        keys: 要重置的键列表，为None则重置所有
    """
    if keys is None:
        # 重置所有状态
        for key in list(st.session_state.keys()):
            if key in DEFAULT_SESSION_STATE:
                st.session_state[key] = DEFAULT_SESSION_STATE[key]
            else:
                del st.session_state[key]
    else:
        # 重置指定键
        for key in keys:
            if key in DEFAULT_SESSION_STATE:
                st.session_state[key] = DEFAULT_SESSION_STATE[key]


def clear_processing_state():
    """清除处理相关状态"""
    processing_keys = [
        'processing_started',
        'processing_complete',
        'processing_paused',
        'current_progress',
        'processed_tiles',
        'total_tiles',
        'logs',
        'result_image'
    ]
    reset_session(processing_keys)


def clear_upload_state():
    """清除上传相关状态"""
    upload_keys = [
        'uploaded_file',
        'source_image',
        'cropped_image',
        'crop_region'
    ]
    reset_session(upload_keys)


def get_config_summary() -> Dict[str, Any]:
    """
    获取配置摘要
    
    Returns:
        配置参数字典
    """
    return {
        'target_resolution': st.session_state.get('target_resolution', 100000000),
        'tile_size': st.session_state.get('tile_size', 1024),
        'overlap_rate': st.session_state.get('overlap_rate', 0.20),
        'max_tiles': st.session_state.get('max_tiles', 100),
        'seedream_version': st.session_state.get('seedream_version', 'Seedream v3.0'),
        'fusion_algorithm': st.session_state.get('fusion_algorithm', '拉普拉斯金字塔'),
        'guidance_scale': st.session_state.get('guidance_scale', 7.5),
        'num_inference_steps': st.session_state.get('num_inference_steps', 50),
        'seed': st.session_state.get('seed', -1),
        'prompt': st.session_state.get('prompt_text', ''),
        'negative_prompt': st.session_state.get('negative_prompt', ''),
    }


def save_config_to_storage():
    """
    将配置保存到持久化存储
    这里可以实现与数据库或本地文件的集成
    """
    config = get_config_summary()
    # TODO: 实现实际的存储逻辑
    st.session_state.config_saved = True
    return config


def load_config_from_storage(config_id: str) -> Optional[Dict[str, Any]]:
    """
    从持久化存储加载配置
    
    Args:
        config_id: 配置ID
        
    Returns:
        配置字典或None
    """
    # TODO: 实现实际的加载逻辑
    return None
