"""
超高分辨率图像生成系统 - Streamlit WebUI
Super Resolution Image Generation System - Web Interface

技术文档第6章 - WebUI界面实现
"""

import streamlit as st
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 页面模块导入
from pages.upload_page import render_upload_page
from pages.config_page import render_config_page
from pages.monitor_page import render_monitor_page
from pages.result_page import render_result_page
from pages.advanced_page import render_advanced_page

# 工具模块
from utils.session_manager import initialize_session_state
from styles.custom_css import apply_custom_css

# 页面配置
st.set_page_config(
    page_title="Super Resolution System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.super-resolution.ai',
        'Report a bug': 'https://github.com/super-resolution/issues',
        'About': '# Super Resolution System v2.0\nAI-powered ultra-high resolution image generation'
    }
)

# 初始化会话状态
initialize_session_state()

# 应用自定义CSS
apply_custom_css()


def render_sidebar():
    """渲染侧边栏导航"""
    with st.sidebar:
        # Logo和标题
        st.markdown("""
        <div class="sidebar-logo">
            <h1>🔮 SuperRes</h1>
            <p class="version">v2.0.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # 页面导航
        st.markdown("<p class='nav-header'>📑 导航菜单</p>", unsafe_allow_html=True)
        
        pages = {
            "upload": "📤 上传与预览",
            "config": "⚙️ 参数配置", 
            "monitor": "📊 处理监控",
            "result": "🖼️ 结果展示",
            "advanced": "🔧 高级功能"
        }
        
        for page_id, page_name in pages.items():
            if st.button(
                page_name,
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if st.session_state.current_page == page_id else "secondary"
            ):
                st.session_state.current_page = page_id
                st.rerun()
        
        st.divider()
        
        # 系统状态概览
        st.markdown("<p class='nav-header'>📈 系统状态</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("在线Agent", st.session_state.get('online_agents', 12), delta="+2")
        with col2:
            st.metric("队列深度", st.session_state.get('queue_depth', 3), delta="-1")
        
        # API配额
        st.progress(0.75, text="API配额: 75% (750/1000)")
        
        st.divider()
        
        # 快捷操作
        st.markdown("<p class='nav-header'>⚡ 快捷操作</p>", unsafe_allow_html=True)
        
        if st.button("🆕 新建任务", use_container_width=True):
            st.session_state.current_page = "upload"
            st.session_state.uploaded_file = None
            st.session_state.processing_complete = False
            st.rerun()
        
        if st.button("📋 任务历史", use_container_width=True):
            st.session_state.show_history = True
        
        # 底部信息
        st.divider()
        st.markdown("""
        <div class="sidebar-footer">
            <p>© 2024 SuperRes AI</p>
            <p class="small">Powered by Seedream v3.0</p>
        </div>
        """, unsafe_allow_html=True)


def render_header():
    """渲染页面头部"""
    page_titles = {
        "upload": ("📤 上传与预览", "Upload & Preview"),
        "config": ("⚙️ 参数配置", "Configuration"),
        "monitor": ("📊 处理监控", "Processing Monitor"),
        "result": ("🖼️ 结果展示", "Results"),
        "advanced": ("🔧 高级功能", "Advanced Features")
    }
    
    title, subtitle = page_titles.get(st.session_state.current_page, ("", ""))
    
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        <p class="subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """主应用入口"""
    # 渲染侧边栏
    render_sidebar()
    
    # 渲染页面头部
    render_header()
    
    # 根据当前页面渲染内容
    current_page = st.session_state.current_page
    
    if current_page == "upload":
        render_upload_page()
    elif current_page == "config":
        render_config_page()
    elif current_page == "monitor":
        render_monitor_page()
    elif current_page == "result":
        render_result_page()
    elif current_page == "advanced":
        render_advanced_page()
    else:
        st.error(f"未知页面: {current_page}")


if __name__ == "__main__":
    main()
