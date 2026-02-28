"""
参数配置页面 - Configuration Page
目标分辨率、重叠率、算法选择、Prompt编辑
"""

import streamlit as st
import math


# 行业模板
INDUSTRY_TEMPLATES = {
    "通用增强": "Ultra high resolution, detailed texture, sharp focus, professional quality",
    "风景摄影": "Breathtaking landscape, ultra detailed, natural colors, dramatic lighting, 8K quality",
    "人像摄影": "Professional portrait, skin texture detail, natural skin tones, soft lighting, high resolution",
    "建筑摄影": "Architectural photography, geometric precision, crisp lines, detailed textures, professional",
    "产品摄影": "Product photography, clean background, sharp details, accurate colors, commercial quality",
    "医学影像": "Medical imaging, high contrast, precise detail, diagnostic quality, clear visualization",
    "卫星遥感": "Satellite imagery, geographic detail, accurate color representation, scientific quality",
    "艺术创作": "Artistic creation, painterly style, rich colors, expressive details, masterpiece quality",
}


def calculate_estimates():
    """计算预估信息"""
    if 'source_image' not in st.session_state or st.session_state.source_image is None:
        return None
    
    image = st.session_state.get('cropped_image') or st.session_state.source_image
    
    # 获取配置参数
    target_pixels = st.session_state.get('target_resolution', 100000000)
    tile_size = st.session_state.get('tile_size', 1024)
    overlap = st.session_state.get('overlap_rate', 0.2)
    
    # 计算目标尺寸
    current_pixels = image.width * image.height
    scale_factor = math.sqrt(target_pixels / current_pixels)
    target_width = int(image.width * scale_factor)
    target_height = int(image.height * scale_factor)
    
    # 计算块数
    effective_tile = int(tile_size * (1 - overlap))
    tiles_x = math.ceil(target_width / effective_tile)
    tiles_y = math.ceil(target_height / effective_tile)
    total_tiles = tiles_x * tiles_y
    
    # API调用次数（考虑失败重试）
    api_calls = int(total_tiles * 1.2)
    
    # 预估费用 (假设每100万次调用 $5)
    estimated_cost = (api_calls / 1000000) * 5
    
    # 预估时间 (假设每块 5-15秒)
    min_time = total_tiles * 5
    max_time = total_tiles * 15
    
    return {
        'target_width': target_width,
        'target_height': target_height,
        'scale_factor': scale_factor,
        'tiles_x': tiles_x,
        'tiles_y': tiles_y,
        'total_tiles': total_tiles,
        'api_calls': api_calls,
        'estimated_cost': estimated_cost,
        'min_time': min_time,
        'max_time': max_time
    }


def render_config_page():
    """渲染参数配置页面"""
    
    # 检查是否有源图像
    if 'source_image' not in st.session_state or st.session_state.source_image is None:
        st.warning("⚠️ 请先在上传页面选择图像")
        if st.button("⬅️ 前往上传页面", type="primary"):
            st.session_state.current_page = "upload"
            st.rerun()
        return
    
    # 创建三列布局
    left_col, center_col, right_col = st.columns([1.2, 1, 1])
    
    with left_col:
        st.markdown("<h3 class='section-title'>🎯 目标设置</h3>", unsafe_allow_html=True)
        
        # 目标分辨率预设
        resolution_options = {
            "1亿像素 (100MP)": 100000000,
            "1.5亿像素 (150MP)": 150000000,
            "2亿像素 (200MP)": 200000000,
            "自定义": 0
        }
        
        selected_resolution = st.selectbox(
            "目标分辨率",
            list(resolution_options.keys()),
            index=0
        )
        
        if selected_resolution == "自定义":
            custom_pixels = st.number_input(
                "自定义像素数 (百万)",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
            st.session_state.target_resolution = custom_pixels * 1000000
        else:
            st.session_state.target_resolution = resolution_options[selected_resolution]
        
        st.divider()
        
        # 分块参数
        st.markdown("<h3 class='section-title'>🧩 分块参数</h3>", unsafe_allow_html=True)
        
        st.session_state.tile_size = st.slider(
            "块大小 (Tile Size)",
            min_value=512,
            max_value=4096,
            value=1024,
            step=256,
            help="每个处理块的大小，越大处理越快但内存占用越高"
        )
        
        st.session_state.overlap_rate = st.slider(
            "重叠率 (Overlap)",
            min_value=0.10,
            max_value=0.30,
            value=0.20,
            step=0.05,
            format="%.0f%%",
            help="块之间的重叠比例，用于平滑融合边界"
        )
        
        st.session_state.max_tiles = st.number_input(
            "分块数量上限",
            min_value=1,
            max_value=1000,
            value=100,
            help="最大允许的块数量，超出将报错"
        )
        
        st.divider()
        
        # AI模型设置
        st.markdown("<h3 class='section-title'>🤖 AI模型设置</h3>", unsafe_allow_html=True)
        
        st.session_state.seedream_version = st.selectbox(
            "Seedream版本",
            ["Seedream v3.0 (推荐)", "Seedream v2.5", "Seedream v2.0", "Seedream v1.5"],
            index=0,
            help="选择Seedream模型版本，v3.0提供最佳质量"
        )
        
        st.session_state.fusion_algorithm = st.radio(
            "融合算法",
            ["拉普拉斯金字塔", "泊松融合", "加权平均"],
            index=0,
            help="选择块融合算法，拉普拉斯金字塔通常效果最佳"
        )
        
        # 高级选项
        with st.expander("🔧 高级选项"):
            st.session_state.guidance_scale = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=20.0,
                value=7.5,
                step=0.5,
                help="控制生成图像与提示词的匹配程度"
            )
            
            st.session_state.num_inference_steps = st.slider(
                "推理步数",
                min_value=20,
                max_value=100,
                value=50,
                step=5,
                help="扩散模型的推理步数，越多质量越高但速度越慢"
            )
            
            st.session_state.seed = st.number_input(
                "随机种子",
                min_value=-1,
                max_value=2147483647,
                value=-1,
                help="-1表示随机种子"
            )
    
    with center_col:
        st.markdown("<h3 class='section-title'>📝 Prompt 编辑</h3>", unsafe_allow_html=True)
        
        # 行业模板选择
        template = st.selectbox(
            "选择行业模板",
            list(INDUSTRY_TEMPLATES.keys()),
            index=0
        )
        
        # Prompt编辑区
        if 'prompt_text' not in st.session_state:
            st.session_state.prompt_text = INDUSTRY_TEMPLATES[template]
        
        prompt = st.text_area(
            "正向提示词 (Positive Prompt)",
            value=st.session_state.prompt_text,
            height=150,
            placeholder="描述你想要生成的图像特征...",
            help="详细的描述将帮助AI生成更好的结果"
        )
        st.session_state.prompt_text = prompt
        
        # 负向提示词
        negative_prompt = st.text_area(
            "负向提示词 (Negative Prompt)",
            value="blurry, low quality, distorted, deformed, ugly, duplicate, watermark, signature, text",
            height=80,
            placeholder="描述你不希望出现的特征..."
        )
        st.session_state.negative_prompt = negative_prompt
        
        # 快速标签
        st.markdown("**快速添加标签:**")
        tag_cols = st.columns(3)
        quick_tags = [
            "8K", "HDR", "detailed",
            "sharp", "vibrant", "professional",
            "realistic", "artistic", "cinematic"
        ]
        
        for i, tag in enumerate(quick_tags):
            with tag_cols[i % 3]:
                if st.button(f"+ {tag}", key=f"tag_{tag}", use_container_width=True):
                    st.session_state.prompt_text = prompt + f", {tag}"
                    st.rerun()
        
        # Prompt分析
        with st.expander("📊 Prompt 分析"):
            word_count = len(prompt.split())
            st.metric("词数", word_count)
            
            # 简单的关键词检测
            keywords = ["detail", "quality", "resolution", "sharp", "professional"]
            detected = [k for k in keywords if k.lower() in prompt.lower()]
            st.write(f"**检测到的关键词:** {', '.join(detected) if detected else '无'}")
            
            if word_count < 10:
                st.warning("⚠️ 提示词较短，建议添加更多描述以获得更好效果")
            elif word_count > 100:
                st.info("ℹ️ 提示词较长，可能会被截断")
    
    with right_col:
        st.markdown("<h3 class='section-title'>📊 实时预估</h3>", unsafe_allow_html=True)
        
        estimates = calculate_estimates()
        
        if estimates:
            # 使用卡片样式显示预估信息
            st.markdown("""
            <div class="estimate-card">
                <h4>🎯 目标尺寸</h4>
            </div>
            """, unsafe_allow_html=True)
            
            est_col1, est_col2 = st.columns(2)
            with est_col1:
                st.metric("目标宽度", f"{estimates['target_width']:,}")
            with est_col2:
                st.metric("目标高度", f"{estimates['target_height']:,}")
            
            st.metric("放大倍数", f"{estimates['scale_factor']:.2f}x")
            
            st.markdown("""
            <div class="estimate-card">
                <h4>🧩 分块信息</h4>
            </div>
            """, unsafe_allow_html=True)
            
            tile_col1, tile_col2 = st.columns(2)
            with tile_col1:
                st.metric("X方向块数", estimates['tiles_x'])
            with tile_col2:
                st.metric("Y方向块数", estimates['tiles_y'])
            
            st.metric("总块数", estimates['total_tiles'], 
                     delta="⚠️ 超出上限!" if estimates['total_tiles'] > st.session_state.max_tiles else None)
            
            st.markdown("""
            <div class="estimate-card">
                <h4>💰 资源预估</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("API调用次数", f"{estimates['api_calls']:,}")
            st.metric("预估费用", f"${estimates['estimated_cost']:.4f}")
            
            min_min = estimates['min_time'] // 60
            max_min = estimates['max_time'] // 60
            st.metric("预估时间", f"{min_min}-{max_min} 分钟")
            
            # 警告信息
            if estimates['total_tiles'] > st.session_state.max_tiles:
                st.error(f"⚠️ 总块数 ({estimates['total_tiles']}) 超过上限 ({st.session_state.max_tiles})，请调整参数")
            elif estimates['total_tiles'] > 50:
                st.warning("⚠️ 块数较多，处理时间可能较长")
        else:
            st.info("配置参数后将显示预估信息")
        
        # 源图像预览
        st.markdown("<h4 class='subsection-title'>源图像</h4>", unsafe_allow_html=True)
        display_image = st.session_state.get('cropped_image') or st.session_state.source_image
        st.image(display_image, use_container_width=True)
    
    # 底部操作栏
    st.divider()
    
    action_col1, action_col2, action_col3 = st.columns([1, 1, 1])
    with action_col1:
        if st.button("⬅️ 返回上传", use_container_width=True):
            st.session_state.current_page = "upload"
            st.rerun()
    
    with action_col2:
        if st.button("💾 保存配置", use_container_width=True):
            # 保存配置到session state
            st.session_state.config_saved = True
            st.success("✅ 配置已保存")
    
    with action_col3:
        can_proceed = estimates and estimates['total_tiles'] <= st.session_state.max_tiles
        if st.button(
            "➡️ 开始处理",
            use_container_width=True,
            type="primary",
            disabled=not can_proceed
        ):
            # 初始化处理状态
            st.session_state.processing_started = True
            st.session_state.processing_complete = False
            st.session_state.current_progress = 0
            st.session_state.processed_tiles = 0
            st.session_state.current_page = "monitor"
            st.rerun()
