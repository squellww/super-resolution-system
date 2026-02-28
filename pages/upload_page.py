"""
上传与预览页面 - Upload & Preview Page
支持多格式上传、元信息解析、交互式裁剪
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import base64


def render_upload_page():
    """渲染上传与预览页面"""
    
    # 创建两列布局
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("<h3 class='section-title'>📥 图像上传</h3>", unsafe_allow_html=True)
        
        # 文件上传组件
        uploaded_file = st.file_uploader(
            "选择图像文件",
            type=['jpg', 'jpeg', 'png', 'tiff', 'tif', 'raw', 'cr2', 'nef', 'arw'],
            accept_multiple_files=False,
            help="支持格式: JPG/PNG/TIFF/RAW, 最大500MB"
        )
        
        if uploaded_file is not None:
            # 保存到session state
            st.session_state.uploaded_file = uploaded_file
            
            # 显示文件信息
            st.markdown("<h4 class='subsection-title'>📋 文件信息</h4>", unsafe_allow_html=True)
            
            file_info_col1, file_info_col2, file_info_col3 = st.columns(3)
            with file_info_col1:
                st.metric("文件名", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
            with file_info_col2:
                size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.metric("文件大小", f"{size_mb:.2f} MB")
            with file_info_col3:
                st.metric("格式", uploaded_file.name.split('.')[-1].upper())
            
            # 处理图像
            try:
                image = Image.open(uploaded_file)
                st.session_state.source_image = image
                
                # 图像元信息
                st.markdown("<h4 class='subsection-title'>🔍 图像元信息</h4>", unsafe_allow_html=True)
                
                meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                with meta_col1:
                    st.metric("分辨率", f"{image.width} × {image.height}")
                with meta_col2:
                    total_pixels = image.width * image.height
                    st.metric("总像素", f"{total_pixels/1e6:.2f}M")
                with meta_col3:
                    mode_map = {'L': '灰度', 'RGB': 'RGB', 'RGBA': 'RGBA', 'CMYK': 'CMYK'}
                    st.metric("色彩模式", mode_map.get(image.mode, image.mode))
                with meta_col4:
                    # 尝试获取位深度
                    if hasattr(image, 'bits'):
                        bits = image.bits
                    else:
                        bits = 8 if image.mode in ['L', 'RGB'] else '未知'
                    st.metric("位深度", f"{bits} bit")
                
                # EXIF信息（如果有）
                if hasattr(image, '_getexif') and image._getexif():
                    with st.expander("📷 EXIF 详细信息"):
                        exif = image._getexif()
                        exif_data = {}
                        for tag_id, value in exif.items():
                            from PIL.ExifTags import TAGS
                            tag = TAGS.get(tag_id, tag_id)
                            exif_data[tag] = value
                        
                        exif_col1, exif_col2 = st.columns(2)
                        with exif_col1:
                            st.write("**相机信息**")
                            st.text(f"厂商: {exif_data.get('Make', 'N/A')}")
                            st.text(f"型号: {exif_data.get('Model', 'N/A')}")
                            st.text(f"镜头: {exif_data.get('LensModel', 'N/A')}")
                        with exif_col2:
                            st.write("**拍摄参数**")
                            st.text(f"光圈: f/{exif_data.get('FNumber', 'N/A')}")
                            st.text(f"快门: {exif_data.get('ExposureTime', 'N/A')}")
                            st.text(f"ISO: {exif_data.get('ISOSpeedRatings', 'N/A')}")
                
            except Exception as e:
                st.error(f"无法读取图像: {str(e)}")
        
        else:
            # 显示上传提示
            st.info("👆 请上传图像文件开始处理")
            
            # 示例格式支持
            st.markdown("""
            <div class="format-support">
                <p><strong>支持的格式:</strong></p>
                <div class="format-badges">
                    <span class="badge">JPG/JPEG</span>
                    <span class="badge">PNG</span>
                    <span class="badge">TIFF</span>
                    <span class="badge">RAW</span>
                    <span class="badge">CR2</span>
                    <span class="badge">NEF</span>
                </div>
                <p class="limit-text">最大文件大小: 500MB</p>
            </div>
            """, unsafe_allow_html=True)
    
    with right_col:
        st.markdown("<h3 class='section-title'>👁️ 图像预览</h3>", unsafe_allow_html=True)
        
        if 'source_image' in st.session_state and st.session_state.source_image:
            image = st.session_state.source_image
            
            # 显示原图预览
            st.markdown("<h4 class='subsection-title'>原始图像</h4>", unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption=f"{image.width} × {image.height} px")
            
            # 交互式裁剪工具
            st.markdown("<h4 class='subsection-title'>✂️ ROI 裁剪工具</h4>", unsafe_allow_html=True)
            
            with st.expander("展开裁剪选项", expanded=False):
                crop_type = st.radio(
                    "裁剪类型",
                    ["矩形裁剪", "多边形裁剪 (开发中)"],
                    horizontal=True
                )
                
                if crop_type == "矩形裁剪":
                    col1, col2 = st.columns(2)
                    with col1:
                        crop_left = st.number_input("左边距", 0, image.width-1, 0)
                        crop_top = st.number_input("上边距", 0, image.height-1, 0)
                    with col2:
                        crop_right = st.number_input("右边距", crop_left+1, image.width, image.width)
                        crop_bottom = st.number_input("下边距", crop_top+1, image.height, image.height)
                    
                    # 应用裁剪
                    if st.button("✅ 应用裁剪", use_container_width=True):
                        cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
                        st.session_state.cropped_image = cropped
                        st.session_state.crop_region = (crop_left, crop_top, crop_right, crop_bottom)
                        st.success(f"裁剪完成: {cropped.width} × {cropped.height}")
                        st.rerun()
                    
                    # 快速预设
                    st.markdown("**快速预设:**")
                    preset_cols = st.columns(3)
                    with preset_cols[0]:
                        if st.button("🎯 中心区域", use_container_width=True):
                            w, h = image.width, image.height
                            cx, cy = w // 2, h // 2
                            size = min(w, h) // 2
                            st.session_state.crop_region = (cx-size, cy-size, cx+size, cy+size)
                            st.rerun()
                    with preset_cols[1]:
                        if st.button("🖼️ 全图", use_container_width=True):
                            st.session_state.crop_region = None
                            st.session_state.cropped_image = None
                            st.rerun()
                    with preset_cols[2]:
                        if st.button("📐 1:1 正方形", use_container_width=True):
                            w, h = image.width, image.height
                            size = min(w, h)
                            cx, cy = w // 2, h // 2
                            st.session_state.crop_region = (cx-size//2, cy-size//2, cx+size//2, cy+size//2)
                            st.rerun()
            
            # 显示裁剪后的图像
            if 'cropped_image' in st.session_state and st.session_state.cropped_image:
                st.markdown("<h4 class='subsection-title'>裁剪预览</h4>", unsafe_allow_html=True)
                st.image(st.session_state.cropped_image, use_container_width=True)
                
                crop_info_col1, crop_info_col2 = st.columns(2)
                with crop_info_col1:
                    st.info(f"裁剪尺寸: {st.session_state.cropped_image.width} × {st.session_state.cropped_image.height}")
                with crop_info_col2:
                    if st.button("🗑️ 清除裁剪", use_container_width=True):
                        del st.session_state.cropped_image
                        del st.session_state.crop_region
                        st.rerun()
        
        else:
            # 空状态
            st.markdown("""
            <div class="empty-preview">
                <div class="empty-icon">🖼️</div>
                <p>上传图像后将在此处预览</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 底部操作栏
    st.divider()
    
    action_col1, action_col2, action_col3 = st.columns([1, 1, 1])
    with action_col1:
        if st.button("🔄 重新上传", use_container_width=True):
            # 清除所有上传相关状态
            for key in ['uploaded_file', 'source_image', 'cropped_image', 'crop_region']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with action_col2:
        if 'source_image' in st.session_state:
            # 下载原图
            buf = io.BytesIO()
            st.session_state.source_image.save(buf, format='PNG')
            st.download_button(
                "⬇️ 下载原图",
                buf.getvalue(),
                file_name="source_image.png",
                mime="image/png",
                use_container_width=True
            )
    
    with action_col3:
        if 'source_image' in st.session_state:
            if st.button("➡️ 下一步: 参数配置", use_container_width=True, type="primary"):
                st.session_state.current_page = "config"
                st.rerun()
