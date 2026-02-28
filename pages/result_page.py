"""
结果展示页面 - Results Page
多尺度对比、质量雷达图、导出选项
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import base64


def create_comparison_slider(before_img, after_img):
    """创建滑动对比组件"""
    # 由于Streamlit原生不支持滑动对比，使用两列布局替代
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='text-align: center;'>Before (原始)</h4>", unsafe_allow_html=True)
        st.image(before_img, use_container_width=True)
    
    with col2:
        st.markdown("<h4 style='text-align: center;'>After (增强后)</h4>", unsafe_allow_html=True)
        st.image(after_img, use_container_width=True)


def create_quality_radar_chart():
    """创建质量雷达图"""
    try:
        import plotly.graph_objects as go
        
        # 六维质量指标
        categories = ['锐度', '细节', '色彩', '对比度', '噪声', '自然度']
        
        # 原始图像评分 (1-10)
        before_scores = [5.5, 4.8, 7.2, 6.5, 6.0, 8.0]
        
        # 增强后评分
        after_scores = [9.2, 9.5, 8.8, 8.5, 8.2, 8.5]
        
        fig = go.Figure()
        
        # 添加原始图像数据
        fig.add_trace(go.Scatterpolar(
            r=before_scores + [before_scores[0]],  # 闭合
            theta=categories + [categories[0]],
            fill='toself',
            name='原始图像',
            line_color='rgba(255, 99, 71, 0.8)',
            fillcolor='rgba(255, 99, 71, 0.2)'
        ))
        
        # 添加增强后数据
        fig.add_trace(go.Scatterpolar(
            r=after_scores + [after_scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='增强后',
            line_color='rgba(0, 191, 255, 0.8)',
            fillcolor='rgba(0, 191, 255, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True,
            title="图像质量六维评估",
            height=400
        )
        
        return fig
    except ImportError:
        return None


def render_quality_metrics():
    """渲染质量指标面板"""
    st.markdown("<h3 class='section-title'>📊 质量指标</h3>", unsafe_allow_html=True)
    
    # 主要指标
    metric_cols = st.columns(4)
    
    metrics = [
        ("PSNR", "42.3", "dB", "+8.5"),
        ("SSIM", "0.96", "", "+0.12"),
        ("LPIPS", "0.04", "", "-0.15"),
        ("FID", "12.5", "", "-25.3")
    ]
    
    for i, (name, value, unit, delta) in enumerate(metrics):
        with metric_cols[i]:
            st.metric(f"{name}", f"{value} {unit}", delta=delta)
    
    # 详细指标
    st.markdown("<h4>详细评估</h4>", unsafe_allow_html=True)
    
    detail_data = {
        "指标": ["边缘锐度", "纹理细节", "色彩保真度", "对比度", "噪声水平", "结构相似性", "感知质量"],
        "原始": ["6.2/10", "5.5/10", "8.0/10", "7.2/10", "6.5/10", "0.84", "6.8/10"],
        "增强后": ["9.5/10", "9.8/10", "9.2/10", "8.8/10", "8.5/10", "0.96", "9.2/10"],
        "提升": ["+53%", "+78%", "+15%", "+22%", "+31%", "+14%", "+35%"]
    }
    
    st.dataframe(
        detail_data,
        use_container_width=True,
        hide_index=True
    )


def render_export_options():
    """渲染导出选项"""
    st.markdown("<h3 class='section-title'>💾 导出选项</h3>", unsafe_allow_html=True)
    
    # 输出格式
    st.markdown("<h4>输出格式</h4>", unsafe_allow_html=True)
    
    format_col1, format_col2, format_col3 = st.columns(3)
    with format_col1:
        output_format = st.selectbox(
            "文件格式",
            ["PNG", "JPEG", "TIFF", "WebP"],
            index=0
        )
    with format_col2:
        if output_format in ["JPEG", "WebP"]:
            quality = st.slider("压缩质量", 1, 100, 95)
        else:
            quality = None
            st.info("无损格式")
    with format_col3:
        color_space = st.selectbox(
            "色彩空间",
            ["sRGB", "Adobe RGB", "ProPhoto RGB", "CMYK"],
            index=0
        )
    
    # 高级选项
    with st.expander("🔧 高级导出选项"):
        col1, col2 = st.columns(2)
        with col1:
            bit_depth = st.selectbox("位深度", ["8-bit", "16-bit"], index=1)
            include_metadata = st.toggle("包含元数据", value=True)
        with col2:
            embed_icc = st.toggle("嵌入ICC配置文件", value=True)
            progressive = st.toggle("渐进式编码", value=False)
    
    # 导出按钮
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # 模拟下载按钮
        if st.button("📥 下载结果图像", use_container_width=True, type="primary"):
            st.success("✅ 导出成功!")
            
            # 显示导出信息
            st.info(f"""
            **导出详情:**
            - 格式: {output_format}
            - 质量: {quality or '无损'}
            - 色彩空间: {color_space}
            - 位深度: {bit_depth if 'bit_depth' in locals() else '8-bit'}
            """)
    
    with export_col2:
        if st.button("📋 复制分享链接", use_container_width=True):
            st.code("https://superres.ai/share/abc123xyz", language=None)
            st.success("链接已生成!")
    
    with export_col3:
        if st.button("☁️ 保存到云端", use_container_width=True):
            with st.spinner("上传中..."):
                time.sleep(1)
            st.success("已保存到云端存储")


def render_result_page():
    """渲染结果展示页面"""
    
    # 检查是否有处理结果
    if not st.session_state.get('processing_complete'):
        st.warning("⚠️ 尚未完成图像处理")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ 前往监控", use_container_width=True):
                st.session_state.current_page = "monitor"
                st.rerun()
        with col2:
            if st.button("🔄 模拟完成 (演示)", use_container_width=True, type="primary"):
                st.session_state.processing_complete = True
                st.rerun()
        return
    
    # 创建模拟结果图像
    if 'result_image' not in st.session_state:
        # 使用源图像或创建模拟图像
        if 'source_image' in st.session_state:
            source = st.session_state.source_image
            # 模拟放大2倍
            result_size = (source.width * 2, source.height * 2)
            st.session_state.result_image = source.resize(result_size, Image.LANCZOS)
        else:
            # 创建模拟图像
            st.session_state.result_image = Image.new('RGB', (2048, 2048), color=(100, 150, 200))
    
    # 顶部：对比视图
    st.markdown("<h3 class='section-title'>🔄 对比视图</h3>", unsafe_allow_html=True)
    
    # 对比模式选择
    compare_mode = st.segmented_control(
        "对比模式",
        ["并排对比", "滑动对比 (模拟)", "差异热力图"],
        default="并排对比"
    )
    
    source_img = st.session_state.get('cropped_image') or st.session_state.get('source_image')
    result_img = st.session_state.result_image
    
    if compare_mode == "并排对比":
        create_comparison_slider(source_img, result_img)
    
    elif compare_mode == "滑动对比 (模拟)":
        # 使用列布局模拟滑动效果
        ratio = st.slider("对比比例", 0, 100, 50)
        col1, col2 = st.columns([ratio, 100-ratio])
        with col1:
            st.markdown("<p style='text-align: center;'>Before</p>", unsafe_allow_html=True)
            st.image(source_img, use_container_width=True)
        with col2:
            st.markdown("<p style='text-align: center;'>After</p>", unsafe_allow_html=True)
            st.image(result_img, use_container_width=True)
    
    else:  # 差异热力图
        st.markdown("<h4>差异热力图</h4>", unsafe_allow_html=True)
        
        # 创建模拟差异图
        diff_array = np.random.randint(0, 100, (200, 200), dtype=np.uint8)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(diff_array, caption="差异强度", use_container_width=True)
        with col2:
            st.markdown("""
            **图例说明:**
            - 🔴 红色: 高差异区域 (大幅增强)
            - 🟡 黄色: 中等差异
            - 🟢 绿色: 低差异区域 (保持原样)
            """)
    
    st.divider()
    
    # 中部：质量评估
    quality_col1, quality_col2 = st.columns([1, 1])
    
    with quality_col1:
        render_quality_metrics()
    
    with quality_col2:
        st.markdown("<h3 class='section-title'>📈 质量雷达图</h3>", unsafe_allow_html=True)
        
        radar_chart = create_quality_radar_chart()
        if radar_chart:
            st.plotly_chart(radar_chart, use_container_width=True)
        else:
            st.info("请安装 plotly 以查看雷达图: `pip install plotly`")
            
            # 使用柱状图替代
            import plotly.graph_objects as go
            categories = ['锐度', '细节', '色彩', '对比度', '噪声', '自然度']
            before = [5.5, 4.8, 7.2, 6.5, 6.0, 8.0]
            after = [9.2, 9.5, 8.8, 8.5, 8.2, 8.5]
            
            fig = go.Figure(data=[
                go.Bar(name='原始', x=categories, y=before, marker_color='rgba(255, 99, 71, 0.8)'),
                go.Bar(name='增强后', x=categories, y=after, marker_color='rgba(0, 191, 255, 0.8)')
            ])
            fig.update_layout(barmode='group', title="质量对比", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 底部：导出选项
    render_export_options()
    
    # 底部操作栏
    st.divider()
    
    action_col1, action_col2, action_col3 = st.columns([1, 1, 1])
    with action_col1:
        if st.button("🔄 处理新图像", use_container_width=True):
            # 重置状态
            for key in ['uploaded_file', 'source_image', 'cropped_image', 
                       'processing_started', 'processing_complete', 'result_image']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_page = "upload"
            st.rerun()
    
    with action_col2:
        if st.button("⚙️ 调整参数重试", use_container_width=True):
            st.session_state.processing_complete = False
            st.session_state.current_page = "config"
            st.rerun()
    
    with action_col3:
        if st.button("📊 查看历史", use_container_width=True):
            st.session_state.current_page = "advanced"
            st.session_state.advanced_tab = "history"
            st.rerun()
