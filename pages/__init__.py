"""
页面模块包
"""

from .upload_page import render_upload_page
from .config_page import render_config_page
from .monitor_page import render_monitor_page
from .result_page import render_result_page
from .advanced_page import render_advanced_page

__all__ = [
    'render_upload_page',
    'render_config_page',
    'render_monitor_page',
    'render_result_page',
    'render_advanced_page'
]
