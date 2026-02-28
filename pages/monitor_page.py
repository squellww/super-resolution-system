"""
处理监控页面 - Processing Monitor Page
整体进度、Agent状态、实时日志、中间预览
"""

import streamlit as st
import time
import random
from datetime import datetime, timedelta


def generate_mock_logs():
    """生成模拟日志数据"""
    log_types = [
        ("INFO", "初始化分块处理器", "text"),
        ("INFO", f"加载模型: {st.session_state.get('seedream_version', 'Seedream v3.0')}", "text"),
        ("INFO", "开始图像分块", "text"),
        ("SUCCESS", "分块完成，共生成 {} 个块", "success"),
        ("INFO", "启动并行处理队列", "text"),
        ("INFO", "Agent-01 开始处理块 (0,0)", "text"),
        ("INFO", "Agent-02 开始处理块 (0,1)", "text"),
        ("INFO", "Agent-03 开始处理块 (1,0)", "text"),
        ("SUCCESS", "块 (0,0) 处理完成，耗时 8.2s", "success"),
        ("SUCCESS", "块 (0,1) 处理完成，耗时 7.9s", "success"),
        ("INFO", "Agent-01 开始处理块 (1,1)", "text"),
        ("INFO", "Agent-02 开始处理块 (2,0)", "text"),
        ("SUCCESS", "块 (1,0) 处理完成，耗时 9.1s", "success"),
        ("INFO", "应用融合算法: 拉普拉斯金字塔", "text"),
        ("SUCCESS", "块 (1,1) 处理完成，耗时 8.5s", "success"),
        ("SUCCESS", "块 (2,0) 处理完成，耗时 8.8s", "success"),
        ("INFO", "开始图像融合", "text"),
        ("INFO", "融合进度: 25%", "text"),
        ("INFO", "融合进度: 50%", "text"),
        ("INFO", "融合进度: 75%", "text"),
        ("SUCCESS", "图像融合完成", "success"),
        ("INFO", "执行后处理: 色彩校正", "text"),
        ("INFO", "执行后处理: 锐化", "text"),
        ("SUCCESS", "后处理完成", "success"),
        ("INFO", "生成输出文件", "text"),
        ("SUCCESS", "处理完成!", "success"),
    ]
    return log_types


def render_agent_status():
    """渲染Agent状态面板"""
    st.markdown("<h3 class='section-title'>🤖 Agent 状态</h3>", unsafe_allow_html=True)
    
    # Agent状态数据
    agents = [
        {"id": "Agent-01", "status": "processing", "task": "Block (2,3)", "progress": 65},
        {"id": "Agent-02", "status": "processing", "task": "Block (2,4)", "progress": 42},
        {"id": "Agent-03", "status": "idle", "task": "-", "progress": 0},
        {"id": "Agent-04", "status": "processing", "task": "Block (3,2)", "progress": 78},
        {"id": "Agent-05", "status": "processing", "task": "Block (3,3)", "progress": 31},
        {"id": "Agent-06", "status": "idle", "task": "-", "progress": 0},
        {"id": "Agent-07", "status": "offline", "task": "-", "progress": 0},
        {"id": "Agent-08", "status": "processing", "task": "Block (3,4)", "progress": 55},
    ]
    
    # 统计信息
    total = len(agents)
    online = sum(1 for a in agents if a['status'] != 'offline')
    processing = sum(1 for a in agents if a['status'] == 'processing')
    idle = sum(1 for a in agents if a['status'] == 'idle')
    
    # 指标卡片
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("在线Agent", online, delta=f"{online}/{total}")
    with metric_cols[1]:
        st.metric("处理中", processing)
    with metric_cols[2]:
        st.metric("空闲", idle)
    with metric_cols[3]:
        st.metric("离线", total - online)
    
    # Agent详情表格
    with st.expander("查看Agent详情", expanded=True):
        for agent in agents:
            status_color = {
                "processing": "🟢",
                "idle": "⚪",
                "offline": "🔴"
            }.get(agent['status'], "⚪")
            
            status_text = {
                "processing": "处理中",
                "idle": "空闲",
                "offline": "离线"
            }.get(agent['status'], "未知")
            
            col1, col2, col3, col4 = st.columns([2, 2, 3, 4])
            with col1:
                st.text(f"{status_color} {agent['id']}")
            with col2:
                st.text(status_text)
            with col3:
                st.text(agent['task'])
            with col4:
                if agent['status'] == 'processing':
                    st.progress(agent['progress'] / 100, text=f"{agent['progress']}%")


def render_progress_panel():
    """渲染进度面板"""
    st.markdown("<h3 class='section-title'>📊 处理进度</h3>", unsafe_allow_html=True)
    
    # 获取或初始化进度
    if 'current_progress' not in st.session_state:
        st.session_state.current_progress = 0
    if 'processed_tiles' not in st.session_state:
        st.session_state.processed_tiles = 0
    if 'total_tiles' not in st.session_state:
        st.session_state.total_tiles = 25  # 默认值
    
    progress = st.session_state.current_progress
    
    # 整体进度条
    st.markdown("<h4>整体进度</h4>", unsafe_allow_html=True)
    st.progress(progress / 100, text=f"{progress}%")
    
    # 进度统计
    progress_cols = st.columns(3)
    with progress_cols[0]:
        st.metric("已完成块", f"{st.session_state.processed_tiles}/{st.session_state.total_tiles}")
    with progress_cols[1]:
        remaining = st.session_state.total_tiles - st.session_state.processed_tiles
        st.metric("剩余块", remaining)
    with progress_cols[2]:
        eta_min = int(remaining * 0.5)  # 假设每块30秒
        st.metric("预计剩余时间", f"{eta_min} 分钟")
    
    # 处理阶段
    stages = [
        ("图像分块", 100),
        ("并行处理", progress),
        ("图像融合", 0 if progress < 80 else (progress - 80) * 5),
        ("后处理", 0 if progress < 95 else (progress - 95) * 20),
        ("输出生成", 0 if progress < 99 else 100)
    ]
    
    st.markdown("<h4>处理阶段</h4>", unsafe_allow_html=True)
    for stage_name, stage_progress in stages:
        status_icon = "✅" if stage_progress == 100 else "🔄" if stage_progress > 0 else "⏳"
        st.progress(stage_progress / 100, text=f"{status_icon} {stage_name}")


def render_logs_panel():
    """渲染日志面板"""
    st.markdown("<h3 class='section-title'>📝 实时日志</h3>", unsafe_allow_html=True)
    
    # 日志控制
    log_control_col1, log_control_col2, log_control_col3 = st.columns([2, 2, 2])
    with log_control_col1:
        log_level = st.selectbox("日志级别", ["ALL", "INFO", "SUCCESS", "WARNING", "ERROR"], index=0)
    with log_control_col2:
        auto_scroll = st.toggle("自动滚动", value=True)
    with log_control_col3:
        if st.button("🗑️ 清空日志", use_container_width=True):
            st.session_state.logs = []
    
    # 日志显示区域
    log_container = st.container(height=300)
    
    with log_container:
        # 生成模拟日志
        if 'logs' not in st.session_state:
            st.session_state.logs = []
        
        # 添加新日志（模拟）
        if st.session_state.get('processing_started') and not st.session_state.get('processing_complete'):
            mock_logs = generate_mock_logs()
            log_index = min(len(st.session_state.logs), len(mock_logs) - 1)
            if log_index < len(mock_logs):
                level, message, msg_type = mock_logs[log_index]
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                st.session_state.logs.append({
                    'time': timestamp,
                    'level': level,
                    'message': message.format(st.session_state.total_tiles) if '{}' in message else message,
                    'type': msg_type
                })
        
        # 显示日志
        for log in st.session_state.logs:
            if log_level != "ALL" and log['level'] != log_level:
                continue
            
            level_color = {
                "INFO": "blue",
                "SUCCESS": "green",
                "WARNING": "orange",
                "ERROR": "red"
            }.get(log['level'], "gray")
            
            st.markdown(
                f"<span style='color: gray;'>[{log['time']}]</span> "
                f"<span style='color: {level_color}; font-weight: bold;'>[{log['level']}]</span> "
                f"{log['message']}",
                unsafe_allow_html=True
            )


def render_preview_panel():
    """渲染中间预览面板"""
    st.markdown("<h3 class='section-title'>👁️ 中间预览</h3>", unsafe_allow_html=True)
    
    # 预览选项
    preview_type = st.segmented_control(
        "预览类型",
        ["处理中块", "融合预览", "差异对比"],
        default="处理中块"
    )
    
    if preview_type == "处理中块":
        # 显示处理中的块缩略图网格
        st.markdown("<h4>处理中的块</h4>", unsafe_allow_html=True)
        
        # 模拟缩略图网格
        grid_cols = st.columns(4)
        for i in range(8):
            with grid_cols[i % 4]:
                # 创建占位符图像
                import numpy as np
                placeholder = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                
                status = random.choice(["✅", "🔄", "⏳"])
                st.image(placeholder, caption=f"Block ({i//4},{i%4}) {status}", use_container_width=True)
    
    elif preview_type == "融合预览":
        st.markdown("<h4>融合进度预览</h4>", unsafe_allow_html=True)
        
        # 模拟融合预览
        col1, col2 = st.columns(2)
        with col1:
            st.info("当前融合区域")
            import numpy as np
            preview = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
            st.image(preview, use_container_width=True)
        with col2:
            st.info("融合边界细节")
            boundary = np.random.randint(100, 150, (200, 200, 3), dtype=np.uint8)
            st.image(boundary, use_container_width=True)
    
    else:  # 差异对比
        st.markdown("<h4>差异对比</h4>", unsafe_allow_html=True)
        st.info("显示原始图像与处理结果的差异热力图")
        
        import numpy as np
        diff_map = np.random.randint(0, 50, (200, 200), dtype=np.uint8)
        st.image(diff_map, use_container_width=True, caption="差异热力图 (低差异区域为深色)")


def render_monitor_page():
    """渲染处理监控页面"""
    
    # 检查处理状态
    if not st.session_state.get('processing_started'):
        st.warning("⚠️ 尚未开始处理任务")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ 返回配置", use_container_width=True):
                st.session_state.current_page = "config"
                st.rerun()
        with col2:
            if st.button("🔄 模拟处理 (演示)", use_container_width=True, type="primary"):
                st.session_state.processing_started = True
                st.session_state.processing_complete = False
                st.session_state.current_progress = 0
                st.session_state.processed_tiles = 0
                st.session_state.total_tiles = 25
                st.session_state.logs = []
                st.rerun()
        return
    
    # 模拟进度更新
    if not st.session_state.get('processing_complete'):
        # 自动更新进度
        if st.session_state.current_progress < 100:
            increment = random.randint(2, 8)
            st.session_state.current_progress = min(100, st.session_state.current_progress + increment)
            st.session_state.processed_tiles = int(
                st.session_state.total_tiles * st.session_state.current_progress / 100
            )
            
            if st.session_state.current_progress >= 100:
                st.session_state.processing_complete = True
                st.session_state.processed_tiles = st.session_state.total_tiles
        
        # 自动刷新
        time.sleep(0.5)
        st.rerun()
    
    # 创建布局
    top_left, top_right = st.columns([1, 1])
    
    with top_left:
        render_progress_panel()
    
    with top_right:
        render_agent_status()
    
    st.divider()
    
    bottom_left, bottom_right = st.columns([1, 1])
    
    with bottom_left:
        render_logs_panel()
    
    with bottom_right:
        render_preview_panel()
    
    # 底部操作栏
    st.divider()
    
    action_col1, action_col2, action_col3 = st.columns([1, 1, 1])
    with action_col1:
        if not st.session_state.processing_complete:
            if st.button("⏸️ 暂停处理", use_container_width=True):
                st.session_state.processing_paused = True
                st.info("处理已暂停")
        else:
            if st.button("🔄 重新处理", use_container_width=True):
                st.session_state.processing_complete = False
                st.session_state.current_progress = 0
                st.session_state.processed_tiles = 0
                st.session_state.logs = []
                st.rerun()
    
    with action_col2:
        if st.button("⚙️ 调整参数", use_container_width=True):
            st.session_state.current_page = "config"
            st.rerun()
    
    with action_col3:
        if st.session_state.processing_complete:
            if st.button("➡️ 查看结果", use_container_width=True, type="primary"):
                st.session_state.current_page = "result"
                st.rerun()
        else:
            if st.button("⏹️ 取消处理", use_container_width=True, type="secondary"):
                st.session_state.processing_started = False
                st.session_state.current_progress = 0
                st.rerun()
