"""
自定义CSS样式 - Custom CSS Styles
"""

import streamlit as st


def apply_custom_css():
    """应用自定义CSS样式"""
    
    st.markdown("""
    <style>
        /* ==================== 全局样式 ==================== */
        
        /* 页面背景 */
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
        
        /* 主内容区域 */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* 文字颜色 */
        .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
            color: #e0e0e0 !important;
        }
        
        /* ==================== 侧边栏样式 ==================== */
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .sidebar-logo {
            text-align: center;
            padding: 1rem 0;
        }
        
        .sidebar-logo h1 {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        
        .sidebar-logo .version {
            font-size: 0.8rem;
            color: #888;
            margin: 0;
        }
        
        .nav-header {
            font-size: 0.75rem;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 1rem 0 0.5rem 0;
        }
        
        .sidebar-footer {
            text-align: center;
            padding: 1rem 0;
            font-size: 0.75rem;
            color: #666;
        }
        
        .sidebar-footer .small {
            font-size: 0.65rem;
            color: #555;
        }
        
        /* ==================== 页面头部样式 ==================== */
        
        .page-header {
            padding: 1rem 0 2rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
        }
        
        .page-header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #fff, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .page-header .subtitle {
            font-size: 1rem;
            color: #888;
            margin: 0;
        }
        
        /* ==================== 章节标题样式 ==================== */
        
        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #00d4ff !important;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(0, 212, 255, 0.3);
        }
        
        .subsection-title {
            font-size: 1rem;
            font-weight: 500;
            color: #aaa !important;
            margin: 1rem 0 0.5rem 0;
        }
        
        /* ==================== 按钮样式 ==================== */
        
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #00d4ff, #7b2cbf);
            border: none;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        }
        
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
        }
        
        .stButton > button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #e0e0e0;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.3);
        }
        
        /* ==================== 输入框样式 ==================== */
        
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #e0e0e0;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #00d4ff;
            box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2);
        }
        
        /* ==================== 选择框样式 ==================== */
        
        .stSelectbox > div > div > div,
        .stMultiselect > div > div > div {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        
        /* ==================== 滑块样式 ==================== */
        
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        }
        
        /* ==================== 进度条样式 ==================== */
        
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            border-radius: 10px;
        }
        
        /* ==================== 指标卡片样式 ==================== */
        
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 700;
            color: #00d4ff !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem;
            color: #888;
        }
        
        /* ==================== 文件上传样式 ==================== */
        
        .stFileUploader > div > div {
            background: rgba(255, 255, 255, 0.05);
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 2rem;
        }
        
        .stFileUploader > div > div:hover {
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.05);
        }
        
        /* ==================== 扩展面板样式 ==================== */
        
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            font-weight: 500;
        }
        
        .streamlit-expanderContent {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 0 0 8px 8px;
        }
        
        /* ==================== 标签页样式 ==================== */
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(0, 212, 255, 0.2) !important;
            border-bottom: 2px solid #00d4ff;
        }
        
        /* ==================== 数据表格样式 ==================== */
        
        .stDataFrame {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
        }
        
        /* ==================== 警告/信息框样式 ==================== */
        
        .stAlert {
            border-radius: 8px;
            border: none;
        }
        
        .stAlert[data-baseweb="notification"] {
            background: rgba(0, 212, 255, 0.1);
            border-left: 4px solid #00d4ff;
        }
        
        /* ==================== 空状态样式 ==================== */
        
        .empty-preview {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 4rem 2rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            border: 2px dashed rgba(255, 255, 255, 0.1);
        }
        
        .empty-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }
        
        .empty-preview p {
            color: #666;
            font-size: 1rem;
        }
        
        /* ==================== 格式支持标签 ==================== */
        
        .format-support {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1rem;
        }
        
        .format-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 0.5rem 0;
        }
        
        .badge {
            background: linear-gradient(135deg, #00d4ff, #7b2cbf);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .limit-text {
            color: #888;
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }
        
        /* ==================== 预估卡片样式 ==================== */
        
        .estimate-card {
            background: rgba(0, 212, 255, 0.1);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin: 1rem 0 0.5rem 0;
        }
        
        .estimate-card h4 {
            color: #00d4ff !important;
            margin: 0;
            font-size: 0.9rem;
        }
        
        /* ==================== 日志样式 ==================== */
        
        .log-entry {
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            padding: 2px 0;
        }
        
        /* ==================== 分隔线样式 ==================== */
        
        hr {
            border-color: rgba(255, 255, 255, 0.1);
            margin: 2rem 0;
        }
        
        /* ==================== 滚动条样式 ==================== */
        
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(0, 212, 255, 0.3);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 212, 255, 0.5);
        }
        
        /* ==================== 响应式调整 ==================== */
        
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
            }
            
            .page-header h1 {
                font-size: 1.5rem;
            }
            
            .section-title {
                font-size: 1.1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)


def get_button_style(color: str = "primary") -> str:
    """
    获取按钮样式类名
    
    Args:
        color: 颜色主题 (primary, secondary, danger, success)
        
    Returns:
        CSS类名
    """
    styles = {
        "primary": "btn-primary",
        "secondary": "btn-secondary",
        "danger": "btn-danger",
        "success": "btn-success"
    }
    return styles.get(color, "btn-primary")


def get_card_style() -> str:
    """获取卡片样式"""
    return """
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    """
