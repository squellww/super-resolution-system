#!/usr/bin/env python3
"""
Kimi Agent集群调度器 - 超高分辨率图像生成系统

该模块实现了用于超高分辨率图像生成系统的Kimi Agent集群调度器，
包含任务队列管理、负载均衡、结果聚合和故障恢复等核心功能。

技术规格:
- 使用asyncio实现异步任务调度
- 使用heapq实现优先级队列
- 使用asyncio.Semaphore控制并发
- 支持任务状态持久化和断点续传

作者: AI Assistant
版本: 1.0.0
"""

import asyncio
import heapq
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any, Tuple
from collections import deque
import hashlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = auto()      # 待处理
    PROCESSING = auto()   # 处理中
    SUCCESS = auto()      # 成功
    FAILED = auto()       # 失败
    RETRYING = auto()     # 重试中
    DEGRADED = auto()     # 降级处理中


class AgentStatus(Enum):
    """Agent状态枚举"""
    IDLE = auto()         # 空闲
    BUSY = auto()         # 忙碌
    OFFLINE = auto()      # 离线
    DEGRADED = auto()     # 降级模式


class VIPLevel(Enum):
    """VIP等级枚举 - 用于任务优先级排序"""
    NORMAL = 0
    SILVER = 1
    GOLD = 2
    PLATINUM = 3
    ENTERPRISE = 4


@dataclass(order=True)
class Task:
    """
    任务数据类
    
    用于表示一个超分辨率图像生成任务，包含所有必要的元数据和优先级信息。
    
    Attributes:
        task_id: 任务唯一标识符
        priority: 优先级分数（越小优先级越高）
        vip_level: VIP等级
        has_roi: 是否包含ROI区域
        has_edge_dependency: 是否有边缘依赖
        submit_time: 提交时间戳
        status: 当前任务状态
        retry_count: 重试次数
        max_retries: 最大重试次数
        input_path: 输入图像路径
        output_path: 输出图像路径
        scale_factor: 放大倍数
        target_resolution: 目标分辨率 (width, height)
        color_mode: 色彩模式 (RGB, RGBA, CMYK等)
        tile_config: 瓦片配置信息
        result_data: 结果数据
        error_message: 错误信息
        checkpoint_data: 断点数据
        assigned_agent: 分配的Agent ID
        processing_start_time: 处理开始时间
        processing_end_time: 处理结束时间
    """
    # 用于堆排序的字段
    priority: float = field(compare=True)
    
    # 任务基本信息
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    vip_level: VIPLevel = field(default=VIPLevel.NORMAL, compare=False)
    has_roi: bool = field(default=False, compare=False)
    has_edge_dependency: bool = field(default=False, compare=False)
    submit_time: float = field(default_factory=time.time, compare=False)
    
    # 任务状态
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    retry_count: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    
    # 图像处理参数
    input_path: str = field(default="", compare=False)
    output_path: str = field(default="", compare=False)
    scale_factor: float = field(default=2.0, compare=False)
    target_resolution: Tuple[int, int] = field(default_factory=lambda: (0, 0), compare=False)
    color_mode: str = field(default="RGB", compare=False)
    tile_config: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    # 结果和错误信息
    result_data: Optional[Dict[str, Any]] = field(default=None, compare=False)
    error_message: str = field(default="", compare=False)
    checkpoint_data: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    # 调度信息
    assigned_agent: Optional[str] = field(default=None, compare=False)
    processing_start_time: Optional[float] = field(default=None, compare=False)
    processing_end_time: Optional[float] = field(default=None, compare=False)
    
    @classmethod
    def calculate_priority(
        cls,
        vip_level: VIPLevel,
        has_roi: bool,
        has_edge_dependency: bool,
        submit_time: float
    ) -> float:
        """
        计算任务优先级分数
        
        优先级排序规则（从高到低）：
        1. VIP等级（权重10000）
        2. ROI包含（权重1000）
        3. 边缘依赖（权重100）
        4. FIFO（提交时间戳）
        
        Args:
            vip_level: VIP等级
            has_roi: 是否包含ROI
            has_edge_dependency: 是否有边缘依赖
            submit_time: 提交时间戳
            
        Returns:
            优先级分数（越小优先级越高）
        """
        priority = 0.0
        
        # VIP等级权重（最高优先级）
        priority -= vip_level.value * 10000
        
        # ROI包含权重
        if has_roi:
            priority -= 1000
        
        # 边缘依赖权重
        if has_edge_dependency:
            priority -= 100
        
        # FIFO原则 - 越早提交优先级越高
        priority += submit_time * 0.001
        
        return priority
    
    def to_dict(self) -> Dict[str, Any]:
        """将任务转换为字典格式，用于序列化"""
        data = asdict(self)
        # 转换枚举类型为字符串
        data['vip_level'] = self.vip_level.name
        data['status'] = self.status.name
        data['vip_level_enum'] = self.vip_level
        data['status_enum'] = self.status
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建任务对象"""
        # 恢复枚举类型
        if 'vip_level' in data and isinstance(data['vip_level'], str):
            data['vip_level'] = VIPLevel[data['vip_level']]
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = TaskStatus[data['status']]
        
        # 移除辅助字段
        data.pop('vip_level_enum', None)
        data.pop('status_enum', None)
        
        return cls(**data)
    
    def get_processing_duration(self) -> Optional[float]:
        """获取任务处理时长"""
        if self.processing_start_time is None:
            return None
        end_time = self.processing_end_time or time.time()
        return end_time - self.processing_start_time


@dataclass
class Agent:
    """
    Agent数据类
    
    用于表示一个Kimi Agent节点，包含状态、性能指标和负载信息。
    
    Attributes:
        agent_id: Agent唯一标识符
        status: 当前状态
        capacity: 处理能力（并发任务数）
        current_load: 当前负载
        pending_tasks: 待处理任务列表
        processed_tasks: 已处理任务计数
        avg_processing_time: 平均处理时间
        network_latency: 网络延迟（毫秒）
        weight: 权重分数
        last_heartbeat: 最后心跳时间
        capabilities: 支持的功能列表
        degradation_level: 降级级别（0=正常）
    """
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: AgentStatus = field(default=AgentStatus.IDLE)
    capacity: int = field(default=1)
    current_load: int = field(default=0)
    pending_tasks: List[str] = field(default_factory=list)
    processed_tasks: int = field(default=0)
    avg_processing_time: float = field(default=0.0)
    network_latency: float = field(default=0.0)
    weight: float = field(default=1.0)
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)
    degradation_level: int = field(default=0)
    
    def calculate_weight(self) -> float:
        """
        计算Agent权重
        
        权重因子：
        - 待处理任务数（负相关）
        - 历史处理时长（负相关）
        - 网络延迟（负相关）
        - 降级级别（负相关）
        
        Returns:
            权重分数（越高越好）
        """
        # 基础权重
        weight = 100.0
        
        # 负载因子（待处理任务越少权重越高）
        load_factor = max(0, self.capacity - len(self.pending_tasks))
        weight += load_factor * 10
        
        # 性能因子（处理时间越短权重越高）
        if self.avg_processing_time > 0:
            performance_factor = 1000.0 / (self.avg_processing_time + 1)
            weight += performance_factor
        
        # 网络延迟因子（延迟越低权重越高）
        if self.network_latency > 0:
            latency_factor = max(0, 100 - self.network_latency * 0.1)
            weight += latency_factor
        
        # 降级因子
        weight -= self.degradation_level * 50
        
        self.weight = weight
        return weight
    
    def is_available(self) -> bool:
        """检查Agent是否可用"""
        return (
            self.status in (AgentStatus.IDLE, AgentStatus.BUSY) and
            len(self.pending_tasks) < self.capacity and
            self.degradation_level < 3
        )
    
    def update_heartbeat(self):
        """更新心跳时间"""
        self.last_heartbeat = time.time()
    
    def check_health(self, timeout: float = 30.0) -> bool:
        """检查Agent健康状态"""
        return (time.time() - self.last_heartbeat) < timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """将Agent转换为字典格式"""
        data = asdict(self)
        data['status'] = self.status.name
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """从字典创建Agent对象"""
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = AgentStatus[data['status']]
        return cls(**data)


class AgentScheduler:
    """
    Kimi Agent集群调度器
    
    核心调度器类，负责任务队列管理、负载均衡、故障恢复和动态扩缩容。
    
    Attributes:
        max_agents: 最大Agent数量
        max_concurrent: 最大并发任务数
        task_queue: 优先级任务队列
        agents: Agent字典
        task_map: 任务映射表
        semaphore: 并发控制信号量
        checkpoint_dir: 检查点保存目录
        
    Example:
        >>> scheduler = AgentScheduler(max_agents=100, max_concurrent=60)
        >>> task = Task(priority=0, input_path="input.jpg", scale_factor=4.0)
        >>> task_id = await scheduler.submit_task(task)
        >>> result = await scheduler.get_task_result(task_id)
    """
    
    # 队列积压阈值配置
    QUEUE_DEPTH_LOW = 10      # 低水位线
    QUEUE_DEPTH_HIGH = 50     # 高水位线
    QUEUE_DEPTH_CRITICAL = 100  # 临界水位线
    
    # 扩缩容配置
    SCALE_UP_THRESHOLD = 0.8   # 扩容阈值（队列使用率）
    SCALE_DOWN_THRESHOLD = 0.2  # 缩容阈值
    MIN_AGENTS = 5
    MAX_AGENTS = 500
    
    def __init__(
        self,
        max_agents: int = 100,
        max_concurrent: int = 60,
        checkpoint_dir: str = "/tmp/agent_scheduler/checkpoints"
    ):
        """
        初始化调度器
        
        Args:
            max_agents: 最大Agent数量
            max_concurrent: 最大并发任务数
            checkpoint_dir: 检查点保存目录
        """
        self.max_agents = max_agents
        self.max_concurrent = max_concurrent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 任务队列（优先级堆）
        self._task_queue: List[Tuple[float, str, Task]] = []
        self._task_map: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, Task] = {}
        self._failed_tasks: Dict[str, Task] = {}
        
        # Agent管理
        self._agents: Dict[str, Agent] = {}
        self._agent_weights: Dict[str, float] = {}
        self._agent_index: int = 0  # 加权轮询索引
        
        # 并发控制
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_lock = asyncio.Lock()
        self._agent_lock = asyncio.Lock()
        
        # 统计信息
        self._stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_retried': 0,
            'scale_up_count': 0,
            'scale_down_count': 0,
            'start_time': time.time()
        }
        
        # 结果聚合回调
        self._result_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # 运行状态
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        logger.info(f"AgentScheduler initialized: max_agents={max_agents}, "
                   f"max_concurrent={max_concurrent}")
    
    async def start(self):
        """启动调度器"""
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("AgentScheduler started")
    
    async def stop(self):
        """停止调度器"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("AgentScheduler stopped")
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self._running:
            try:
                # 检查Agent健康状态
                await self._check_agent_health()
                
                # 动态扩缩容检查
                queue_depth = len(self._task_queue)
                await self.scale_agents(queue_depth)
                
                # 分配任务
                await self._dispatch_tasks()
                
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5)
    
    async def _check_agent_health(self):
        """检查所有Agent的健康状态"""
        async with self._agent_lock:
            for agent_id, agent in list(self._agents.items()):
                if not agent.check_health():
                    logger.warning(f"Agent {agent_id} health check failed")
                    agent.status = AgentStatus.OFFLINE
                    # 重新分配该Agent的任务
                    for task_id in agent.pending_tasks:
                        if task_id in self._task_map:
                            task = self._task_map[task_id]
                            await self.handle_failure(agent_id, task_id, 
                                "Agent health check failed")
    
    async def _dispatch_tasks(self):
        """分发任务到可用Agent"""
        async with self._queue_lock:
            while self._task_queue:
                # 获取优先级最高的任务
                _, task_id, task = heapq.heappop(self._task_queue)
                
                # 查找可用Agent
                agent_id = await self._select_agent()
                if agent_id is None:
                    # 没有可用Agent，任务重新入队
                    heapq.heappush(self._task_queue, (task.priority, task_id, task))
                    break
                
                # 分配任务
                success = await self.assign_to_agent(agent_id, task_id)
                if not success:
                    # 分配失败，重新入队
                    heapq.heappush(self._task_queue, (task.priority, task_id, task))
    
    async def _select_agent(self) -> Optional[str]:
        """
        选择最佳Agent（加权轮询）
        
        Returns:
            选中的Agent ID，如果没有可用Agent则返回None
        """
        async with self._agent_lock:
            available_agents = [
                (aid, agent) for aid, agent in self._agents.items()
                if agent.is_available()
            ]
            
            if not available_agents:
                return None
            
            # 计算权重
            weights = []
            for aid, agent in available_agents:
                weight = agent.calculate_weight()
                weights.append((aid, weight))
            
            # 加权轮询选择
            total_weight = sum(w for _, w in weights)
            if total_weight == 0:
                return available_agents[0][0]
            
            # 选择权重最高的Agent
            weights.sort(key=lambda x: x[1], reverse=True)
            return weights[0][0]
    
    async def submit_task(self, task: Task) -> str:
        """
        提交任务到调度队列
        
        Args:
            task: 要提交的任务对象
            
        Returns:
            任务ID
            
        Raises:
            ValueError: 如果任务参数无效
        """
        # 验证任务参数
        if not task.input_path:
            raise ValueError("Task input_path is required")
        
        # 计算优先级
        task.priority = Task.calculate_priority(
            task.vip_level,
            task.has_roi,
            task.has_edge_dependency,
            task.submit_time
        )
        
        async with self._queue_lock:
            # 添加到任务映射
            self._task_map[task.task_id] = task
            
            # 添加到优先级队列
            heapq.heappush(
                self._task_queue,
                (task.priority, task.task_id, task)
            )
            
            self._stats['total_submitted'] += 1
        
        logger.info(f"Task {task.task_id} submitted with priority {task.priority:.2f}")
        return task.task_id
    
    async def get_next_task(self) -> Optional[Task]:
        """
        获取下一个待处理任务
        
        Returns:
            下一个任务，如果没有则返回None
        """
        async with self._queue_lock:
            while self._task_queue:
                _, task_id, task = heapq.heappop(self._task_queue)
                
                # 检查任务状态
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.PROCESSING
                    task.processing_start_time = time.time()
                    return task
                
                # 已处理的任务，跳过
                if task.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
                    continue
                
                # 其他状态，重新入队
                heapq.heappush(self._task_queue, (task.priority, task_id, task))
            
            return None
    
    async def assign_to_agent(self, agent_id: str, task_id: str) -> bool:
        """
        将任务分配给指定Agent
        
        Args:
            agent_id: Agent ID
            task_id: 任务ID
            
        Returns:
            是否分配成功
        """
        async with self._agent_lock:
            if agent_id not in self._agents:
                logger.error(f"Agent {agent_id} not found")
                return False
            
            agent = self._agents[agent_id]
            
            if not agent.is_available():
                logger.warning(f"Agent {agent_id} is not available")
                return False
        
        async with self._queue_lock:
            if task_id not in self._task_map:
                logger.error(f"Task {task_id} not found")
                return False
            
            task = self._task_map[task_id]
            task.status = TaskStatus.PROCESSING
            task.assigned_agent = agent_id
            task.processing_start_time = time.time()
        
        async with self._agent_lock:
            agent.pending_tasks.append(task_id)
            agent.status = AgentStatus.BUSY
            agent.update_heartbeat()
        
        logger.info(f"Task {task_id} assigned to Agent {agent_id}")
        return True
    
    async def collect_result(
        self,
        agent_id: str,
        task_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        收集任务结果
        
        Args:
            agent_id: Agent ID
            task_id: 任务ID
            result: 结果数据
            
        Returns:
            是否收集成功
        """
        async with self._queue_lock:
            if task_id not in self._task_map:
                logger.error(f"Task {task_id} not found for result collection")
                return False
            
            task = self._task_map[task_id]
            
            # 完整性校验
            if not self._validate_result(result, task):
                logger.error(f"Result validation failed for task {task_id}")
                return False
            
            # 更新任务状态
            task.status = TaskStatus.SUCCESS
            task.processing_end_time = time.time()
            task.result_data = result
            
            # 移动到已完成任务
            self._completed_tasks[task_id] = task
            del self._task_map[task_id]
            
            self._stats['total_completed'] += 1
        
        async with self._agent_lock:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                if task_id in agent.pending_tasks:
                    agent.pending_tasks.remove(task_id)
                agent.processed_tasks += 1
                
                # 更新平均处理时间
                duration = task.get_processing_duration()
                if duration:
                    if agent.avg_processing_time == 0:
                        agent.avg_processing_time = duration
                    else:
                        agent.avg_processing_time = (
                            agent.avg_processing_time * 0.9 + duration * 0.1
                        )
                
                if not agent.pending_tasks:
                    agent.status = AgentStatus.IDLE
                
                agent.update_heartbeat()
        
        # 触发结果回调
        for callback in self._result_callbacks:
            try:
                callback(task_id, result)
            except Exception as e:
                logger.error(f"Result callback error: {e}")
        
        logger.info(f"Result collected for task {task_id} from Agent {agent_id}")
        return True
    
    def _validate_result(self, result: Dict[str, Any], task: Task) -> bool:
        """
        验证结果完整性
        
        校验项：
        - 分辨率
        - 色彩模式
        - 文件完整性
        
        Args:
            result: 结果数据
            task: 原始任务
            
        Returns:
            是否验证通过
        """
        # 检查必要字段
        required_fields = ['output_path', 'width', 'height', 'color_mode']
        for field in required_fields:
            if field not in result:
                logger.error(f"Missing required field: {field}")
                return False
        
        # 校验分辨率
        if task.target_resolution != (0, 0):
            expected_width, expected_height = task.target_resolution
            actual_width = result.get('width', 0)
            actual_height = result.get('height', 0)
            
            if actual_width != expected_width or actual_height != expected_height:
                logger.warning(
                    f"Resolution mismatch for task {task.task_id}: "
                    f"expected {expected_width}x{expected_height}, "
                    f"got {actual_width}x{actual_height}"
                )
                # 允许一定的误差范围
                tolerance = 0.05
                width_diff = abs(actual_width - expected_width) / max(expected_width, 1)
                height_diff = abs(actual_height - expected_height) / max(expected_height, 1)
                
                if width_diff > tolerance or height_diff > tolerance:
                    return False
        
        # 校验色彩模式
        if result.get('color_mode') != task.color_mode:
            logger.warning(
                f"Color mode mismatch for task {task.task_id}: "
                f"expected {task.color_mode}, got {result.get('color_mode')}"
            )
        
        # 校验文件完整性
        output_path = result.get('output_path')
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                logger.error(f"Output file is empty: {output_path}")
                return False
            
            # 校验文件哈希（如果提供）
            if 'file_hash' in result:
                expected_hash = result['file_hash']
                actual_hash = self._calculate_file_hash(output_path)
                if actual_hash != expected_hash:
                    logger.error(f"File hash mismatch for {output_path}")
                    return False
        
        return True
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def handle_failure(
        self,
        agent_id: str,
        task_id: str,
        error: str
    ) -> str:
        """
        处理任务失败
        
        故障恢复策略：
        1. 单Agent失败：自动重入队，最多3次
        2. 疑难块：降级策略（降低放大倍数、切换veImageX）
        3. 全集群故障：断点续传
        
        Args:
            agent_id: Agent ID
            task_id: 任务ID
            error: 错误信息
            
        Returns:
            处理结果描述
        """
        async with self._queue_lock:
            if task_id not in self._task_map:
                return "Task not found"
            
            task = self._task_map[task_id]
            task.error_message = error
            task.processing_end_time = time.time()
        
        async with self._agent_lock:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                if task_id in agent.pending_tasks:
                    agent.pending_tasks.remove(task_id)
                
                # 检查Agent是否需要降级
                recent_failures = sum(
                    1 for t in self._failed_tasks.values()
                    if t.assigned_agent == agent_id and
                    time.time() - t.processing_end_time < 300  # 5分钟内
                )
                
                if recent_failures >= 3:
                    agent.degradation_level += 1
                    agent.status = AgentStatus.DEGRADED
                    logger.warning(f"Agent {agent_id} degraded to level {agent.degradation_level}")
        
        # 决定处理方式
        if task.retry_count < task.max_retries:
            # 重试
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            task.assigned_agent = None
            task.processing_start_time = None
            task.processing_end_time = None
            
            # 重新计算优先级（降低优先级）
            task.priority = Task.calculate_priority(
                task.vip_level,
                task.has_roi,
                task.has_edge_dependency,
                task.submit_time
            ) + task.retry_count * 100  # 每次重试增加优先级惩罚
            
            async with self._queue_lock:
                heapq.heappush(
                    self._task_queue,
                    (task.priority, task_id, task)
                )
            
            self._stats['total_retried'] += 1
            logger.info(f"Task {task_id} requeued for retry ({task.retry_count}/{task.max_retries})")
            return f"Task requeued for retry ({task.retry_count}/{task.max_retries})"
        
        else:
            # 超过重试次数，应用降级策略
            return await self._apply_degradation(task)
    
    async def _apply_degradation(self, task: Task) -> str:
        """
        应用降级策略
        
        降级策略：
        1. 降低放大倍数
        2. 切换处理引擎
        3. 简化处理参数
        
        Args:
            task: 失败的任务
            
        Returns:
            处理结果描述
        """
        task.status = TaskStatus.DEGRADED
        
        # 策略1: 降低放大倍数
        if task.scale_factor > 1.5:
            original_scale = task.scale_factor
            task.scale_factor = max(1.5, task.scale_factor * 0.7)
            task.retry_count = 0  # 重置重试计数
            
            # 更新目标分辨率
            if task.target_resolution != (0, 0):
                width, height = task.target_resolution
                task.target_resolution = (
                    int(width * task.scale_factor / original_scale),
                    int(height * task.scale_factor / original_scale)
                )
            
            logger.info(f"Task {task.task_id} degraded: scale {original_scale} -> {task.scale_factor}")
        
        # 策略2: 简化瓦片配置
        if 'tile_size' in task.tile_config and task.tile_config['tile_size'] > 256:
            task.tile_config['tile_size'] = 256
            task.tile_config['overlap'] = 16
        
        # 策略3: 切换引擎（标记为需要切换）
        task.tile_config['use_fallback_engine'] = True
        
        # 重新入队
        task.priority = Task.calculate_priority(
            task.vip_level,
            task.has_roi,
            task.has_edge_dependency,
            task.submit_time
        ) + 500  # 降级任务优先级惩罚
        
        async with self._queue_lock:
            heapq.heappush(
                self._task_queue,
                (task.priority, task.task_id, task)
            )
        
        logger.info(f"Task {task.task_id} requeued with degradation strategy")
        return "Task requeued with degradation strategy"
    
    async def scale_agents(self, queue_depth: int) -> int:
        """
        动态扩缩容
        
        根据队列深度动态调整Agent数量：
        - queue_depth < QUEUE_DEPTH_LOW: 缩容
        - QUEUE_DEPTH_LOW <= queue_depth < QUEUE_DEPTH_HIGH: 维持
        - queue_depth >= QUEUE_DEPTH_HIGH: 扩容
        
        Args:
            queue_depth: 当前队列深度
            
        Returns:
            调整后的Agent数量
        """
        async with self._agent_lock:
            current_agents = len(self._agents)
            target_agents = current_agents
            
            # 计算队列使用率
            queue_capacity = self.max_concurrent
            queue_usage = queue_depth / max(queue_capacity, 1)
            
            # 扩容判断
            if queue_usage > self.SCALE_UP_THRESHOLD and queue_depth >= self.QUEUE_DEPTH_HIGH:
                # 需要扩容
                if queue_depth >= self.QUEUE_DEPTH_CRITICAL:
                    # 紧急扩容
                    target_agents = min(
                        current_agents + 20,
                        self.MAX_AGENTS,
                        self.max_agents
                    )
                else:
                    # 常规扩容
                    target_agents = min(
                        current_agents + 5,
                        self.MAX_AGENTS,
                        self.max_agents
                    )
                
                if target_agents > current_agents:
                    self._stats['scale_up_count'] += 1
                    logger.info(f"Scaling up: {current_agents} -> {target_agents}")
            
            # 缩容判断
            elif queue_usage < self.SCALE_DOWN_THRESHOLD and queue_depth < self.QUEUE_DEPTH_LOW:
                # 需要缩容
                idle_agents = sum(
                    1 for a in self._agents.values()
                    if a.status == AgentStatus.IDLE
                )
                
                if idle_agents > self.MIN_AGENTS:
                    target_agents = max(
                        current_agents - 3,
                        self.MIN_AGENTS
                    )
                    
                    if target_agents < current_agents:
                        self._stats['scale_down_count'] += 1
                        logger.info(f"Scaling down: {current_agents} -> {target_agents}")
            
            # 执行扩缩容
            if target_agents > current_agents:
                for _ in range(target_agents - current_agents):
                    await self._add_agent()
            elif target_agents < current_agents:
                await self._remove_idle_agents(current_agents - target_agents)
            
            return len(self._agents)
    
    async def _add_agent(self) -> str:
        """添加新Agent"""
        agent = Agent(
            capacity=1,
            capabilities=['super_resolution', 'tile_processing']
        )
        self._agents[agent.agent_id] = agent
        logger.debug(f"Added new Agent {agent.agent_id}")
        return agent.agent_id
    
    async def _remove_idle_agents(self, count: int):
        """移除空闲Agent"""
        removed = 0
        for agent_id, agent in list(self._agents.items()):
            if agent.status == AgentStatus.IDLE and removed < count:
                del self._agents[agent_id]
                removed += 1
                logger.debug(f"Removed idle Agent {agent_id}")
    
    async def register_agent(self, agent: Agent) -> str:
        """
        注册Agent到调度器
        
        Args:
            agent: Agent对象
            
        Returns:
            Agent ID
        """
        async with self._agent_lock:
            self._agents[agent.agent_id] = agent
            logger.info(f"Agent {agent.agent_id} registered")
            return agent.agent_id
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        从调度器注销Agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功注销
        """
        async with self._agent_lock:
            if agent_id not in self._agents:
                return False
            
            agent = self._agents[agent_id]
            
            # 重新分配该Agent的任务
            for task_id in agent.pending_tasks:
                if task_id in self._task_map:
                    await self.handle_failure(agent_id, task_id, "Agent unregistered")
            
            del self._agents[agent_id]
            logger.info(f"Agent {agent_id} unregistered")
            return True
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态字典，如果不存在则返回None
        """
        task = None
        
        if task_id in self._task_map:
            task = self._task_map[task_id]
        elif task_id in self._completed_tasks:
            task = self._completed_tasks[task_id]
        elif task_id in self._failed_tasks:
            task = self._failed_tasks[task_id]
        
        if task:
            return {
                'task_id': task_id,
                'status': task.status.name,
                'retry_count': task.retry_count,
                'assigned_agent': task.assigned_agent,
                'submit_time': task.submit_time,
                'processing_duration': task.get_processing_duration(),
                'error_message': task.error_message
            }
        
        return None
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务结果字典，如果不存在则返回None
        """
        if task_id in self._completed_tasks:
            task = self._completed_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status.name,
                'result': task.result_data,
                'processing_duration': task.get_processing_duration()
            }
        return None
    
    def add_result_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """添加结果回调函数"""
        self._result_callbacks.append(callback)
    
    async def save_checkpoint(self) -> str:
        """
        保存检查点（断点续传）
        
        保存当前调度器状态到文件，包括：
        - 所有任务状态
        - Agent状态
        - 统计信息
        
        Returns:
            检查点文件路径
        """
        checkpoint = {
            'timestamp': time.time(),
            'tasks': {},
            'agents': {},
            'stats': self._stats.copy()
        }
        
        # 保存所有任务
        for task_id, task in self._task_map.items():
            checkpoint['tasks'][task_id] = task.to_dict()
        
        for task_id, task in self._completed_tasks.items():
            checkpoint['tasks'][task_id] = task.to_dict()
        
        for task_id, task in self._failed_tasks.items():
            checkpoint['tasks'][task_id] = task.to_dict()
        
        # 保存所有Agent
        async with self._agent_lock:
            for agent_id, agent in self._agents.items():
                checkpoint['agents'][agent_id] = agent.to_dict()
        
        # 生成检查点文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.json"
        
        # 保存到文件
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)
    
    async def restore_checkpoint(self, checkpoint_path: str) -> bool:
        """
        从检查点恢复（断点续传）
        
        从检查点文件恢复调度器状态，包括：
        - 待处理任务重新入队
        - 处理中任务标记为重试
        - Agent状态恢复
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            是否恢复成功
        """
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            # 恢复任务
            restored_count = 0
            for task_id, task_data in checkpoint.get('tasks', {}).items():
                task = Task.from_dict(task_data)
                
                # 处理中任务标记为重试
                if task.status == TaskStatus.PROCESSING:
                    task.status = TaskStatus.RETRYING
                    task.retry_count += 1
                    task.assigned_agent = None
                
                # 待处理和重试任务重新入队
                if task.status in (TaskStatus.PENDING, TaskStatus.RETRYING):
                    self._task_map[task_id] = task
                    heapq.heappush(
                        self._task_queue,
                        (task.priority, task_id, task)
                    )
                    restored_count += 1
                
                # 已完成任务
                elif task.status == TaskStatus.SUCCESS:
                    self._completed_tasks[task_id] = task
                
                # 失败任务
                elif task.status == TaskStatus.FAILED:
                    self._failed_tasks[task_id] = task
            
            # 恢复Agent
            async with self._agent_lock:
                for agent_id, agent_data in checkpoint.get('agents', {}).items():
                    agent = Agent.from_dict(agent_data)
                    # 重置Agent状态
                    agent.status = AgentStatus.IDLE
                    agent.pending_tasks = []
                    agent.update_heartbeat()
                    self._agents[agent_id] = agent
            
            # 恢复统计信息
            self._stats.update(checkpoint.get('stats', {}))
            
            logger.info(f"Checkpoint restored from {checkpoint_path}: "
                       f"{restored_count} tasks requeued")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        获取调度器统计信息
        
        Returns:
            统计信息字典
        """
        async with self._agent_lock:
            agent_stats = {
                'total': len(self._agents),
                'idle': sum(1 for a in self._agents.values() if a.status == AgentStatus.IDLE),
                'busy': sum(1 for a in self._agents.values() if a.status == AgentStatus.BUSY),
                'offline': sum(1 for a in self._agents.values() if a.status == AgentStatus.OFFLINE),
                'degraded': sum(1 for a in self._agents.values() if a.status == AgentStatus.DEGRADED)
            }
        
        async with self._queue_lock:
            queue_stats = {
                'pending': len(self._task_queue),
                'processing': sum(1 for t in self._task_map.values() if t.status == TaskStatus.PROCESSING),
                'retrying': sum(1 for t in self._task_map.values() if t.status == TaskStatus.RETRYING),
                'degraded': sum(1 for t in self._task_map.values() if t.status == TaskStatus.DEGRADED)
            }
        
        return {
            'agents': agent_stats,
            'queue': queue_stats,
            'tasks': {
                'submitted': self._stats['total_submitted'],
                'completed': self._stats['total_completed'],
                'failed': self._stats['total_failed'],
                'retried': self._stats['total_retried'],
                'completion_rate': (
                    self._stats['total_completed'] / max(self._stats['total_submitted'], 1)
                )
            },
            'scaling': {
                'scale_up_count': self._stats['scale_up_count'],
                'scale_down_count': self._stats['scale_down_count']
            },
            'uptime': time.time() - self._stats['start_time']
        }


# =============================================================================
# 使用示例
# =============================================================================

async def demo_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("Kimi Agent集群调度器 - 基本使用示例")
    print("=" * 60)
    
    # 创建调度器
    scheduler = AgentScheduler(max_agents=10, max_concurrent=5)
    await scheduler.start()
    
    # 注册一些Agent
    for i in range(5):
        agent = Agent(
            capacity=2,
            capabilities=['super_resolution', 'tile_processing']
        )
        await scheduler.register_agent(agent)
    
    # 创建并提交任务
    tasks = []
    for i in range(10):
        task = Task(
            priority=0,  # 会自动计算
            vip_level=VIPLevel.GOLD if i < 3 else VIPLevel.NORMAL,
            has_roi=(i % 3 == 0),
            has_edge_dependency=(i % 2 == 0),
            input_path=f"/data/input/image_{i}.jpg",
            output_path=f"/data/output/image_{i}_sr.jpg",
            scale_factor=4.0,
            target_resolution=(4096, 4096),
            color_mode="RGB"
        )
        task_id = await scheduler.submit_task(task)
        tasks.append(task_id)
        print(f"Submitted task {task_id} (VIP: {task.vip_level.name})")
    
    # 等待一段时间
    await asyncio.sleep(2)
    
    # 获取统计信息
    stats = await scheduler.get_statistics()
    print(f"\n调度器统计:")
    print(f"  Agent数量: {stats['agents']}")
    print(f"  队列状态: {stats['queue']}")
    print(f"  任务统计: {stats['tasks']}")
    
    # 停止调度器
    await scheduler.stop()
    print("\n调度器已停止")


async def demo_priority_queue():
    """优先级队列示例"""
    print("\n" + "=" * 60)
    print("优先级队列示例")
    print("=" * 60)
    
    scheduler = AgentScheduler(max_agents=5, max_concurrent=3)
    
    # 按不同优先级提交任务
    test_cases = [
        (VIPLevel.NORMAL, False, False, "普通用户任务"),
        (VIPLevel.PLATINUM, True, True, "企业VIP+ROI+边缘依赖"),
        (VIPLevel.GOLD, True, False, "金牌VIP+ROI"),
        (VIPLevel.NORMAL, False, True, "普通用户+边缘依赖"),
        (VIPLevel.ENTERPRISE, False, False, "企业VIP"),
    ]
    
    for vip, roi, edge, desc in test_cases:
        task = Task(
            priority=0,
            vip_level=vip,
            has_roi=roi,
            has_edge_dependency=edge,
            input_path=f"/data/input/{desc}.jpg",
            scale_factor=2.0
        )
        task_id = await scheduler.submit_task(task)
        print(f"提交: {desc} -> 优先级: {task.priority:.2f}")
    
    # 按优先级获取任务
    print("\n按优先级顺序获取任务:")
    for i in range(5):
        task = await scheduler.get_next_task()
        if task:
            print(f"  {i+1}. VIP: {task.vip_level.name}, ROI: {task.has_roi}, "
                  f"边缘: {task.has_edge_dependency}, 优先级: {task.priority:.2f}")


async def demo_failure_recovery():
    """故障恢复示例"""
    print("\n" + "=" * 60)
    print("故障恢复示例")
    print("=" * 60)
    
    scheduler = AgentScheduler(max_agents=3, max_concurrent=2)
    await scheduler.start()
    
    # 注册Agent
    for i in range(3):
        agent = Agent(capacity=1)
        await scheduler.register_agent(agent)
    
    # 创建任务
    task = Task(
        priority=0,
        input_path="/data/input/test.jpg",
        scale_factor=4.0,
        max_retries=3
    )
    task_id = await scheduler.submit_task(task)
    print(f"任务 {task_id} 已提交")
    
    # 模拟任务分配
    agent_id = list(scheduler._agents.keys())[0]
    await scheduler.assign_to_agent(agent_id, task_id)
    print(f"任务分配给 Agent {agent_id}")
    
    # 模拟多次失败和重试
    for i in range(3):
        result = await scheduler.handle_failure(agent_id, task_id, f"模拟错误 #{i+1}")
        print(f"  失败处理 #{i+1}: {result}")
        
        # 检查任务状态
        status = await scheduler.get_task_status(task_id)
        if status:
            print(f"    状态: {status['status']}, 重试次数: {status['retry_count']}")
    
    # 第四次失败（触发降级）
    result = await scheduler.handle_failure(agent_id, task_id, "第四次失败")
    print(f"  第四次失败: {result}")
    
    # 检查降级后的任务
    status = await scheduler.get_task_status(task_id)
    if status:
        print(f"    降级后状态: {status['status']}")
    
    await scheduler.stop()


async def demo_checkpoint():
    """断点续传示例"""
    print("\n" + "=" * 60)
    print("断点续传示例")
    print("=" * 60)
    
    # 创建调度器并添加任务
    scheduler1 = AgentScheduler(max_agents=5, max_concurrent=3)
    await scheduler1.start()
    
    # 注册Agent
    for i in range(3):
        agent = Agent(capacity=2)
        await scheduler1.register_agent(agent)
    
    # 添加各种状态的任务
    for i in range(5):
        task = Task(
            priority=0,
            input_path=f"/data/input/checkpoint_{i}.jpg",
            scale_factor=2.0
        )
        await scheduler1.submit_task(task)
    
    # 保存检查点
    checkpoint_path = await scheduler1.save_checkpoint()
    print(f"检查点已保存: {checkpoint_path}")
    
    # 获取保存前的统计
    stats_before = await scheduler1.get_statistics()
    print(f"保存前任务数: {stats_before['tasks']['submitted']}")
    
    await scheduler1.stop()
    
    # 创建新调度器并恢复
    print("\n创建新调度器并恢复...")
    scheduler2 = AgentScheduler(max_agents=5, max_concurrent=3)
    
    success = await scheduler2.restore_checkpoint(checkpoint_path)
    print(f"恢复结果: {'成功' if success else '失败'}")
    
    # 获取恢复后的统计
    stats_after = await scheduler2.get_statistics()
    print(f"恢复后队列深度: {stats_after['queue']['pending']}")
    print(f"恢复后Agent数: {stats_after['agents']['total']}")
    
    # 清理检查点文件
    os.remove(checkpoint_path)
    print(f"已清理检查点文件")


async def demo_load_balancing():
    """负载均衡示例"""
    print("\n" + "=" * 60)
    print("负载均衡示例")
    print("=" * 60)
    
    scheduler = AgentScheduler(max_agents=10, max_concurrent=5)
    
    # 注册不同性能的Agent
    agents_config = [
        {'capacity': 3, 'latency': 10, 'name': '高性能Agent-1'},
        {'capacity': 3, 'latency': 15, 'name': '高性能Agent-2'},
        {'capacity': 1, 'latency': 50, 'name': '低性能Agent-1'},
        {'capacity': 1, 'latency': 60, 'name': '低性能Agent-2'},
    ]
    
    for config in agents_config:
        agent = Agent(
            capacity=config['capacity'],
            network_latency=config['latency'],
            capabilities=['super_resolution']
        )
        await scheduler.register_agent(agent)
        print(f"注册 {config['name']}: 容量={config['capacity']}, 延迟={config['latency']}ms")
    
    # 提交多个任务，观察分配
    print("\n提交任务并观察Agent选择:")
    for i in range(8):
        task = Task(
            priority=0,
            input_path=f"/data/input/lb_{i}.jpg",
            scale_factor=2.0
        )
        task_id = await scheduler.submit_task(task)
        
        # 手动分配任务（模拟调度）
        agent_id = await scheduler._select_agent()
        if agent_id:
            agent = scheduler._agents[agent_id]
            await scheduler.assign_to_agent(agent_id, task_id)
            print(f"  任务 {i+1} -> Agent (负载:{len(agent.pending_tasks)}, "
                  f"权重:{agent.weight:.1f})")


async def demo_dynamic_scaling():
    """动态扩缩容示例"""
    print("\n" + "=" * 60)
    print("动态扩缩容示例")
    print("=" * 60)
    
    scheduler = AgentScheduler(max_agents=50, max_concurrent=20)
    await scheduler.start()
    
    # 初始Agent数量
    print(f"初始Agent数量: {len(scheduler._agents)}")
    
    # 模拟队列积压，触发扩容
    test_scenarios = [
        (5, "低负载"),
        (30, "中等负载"),
        (120, "高负载（紧急扩容）"),
        (8, "负载下降（缩容）"),
    ]
    
    for queue_depth, desc in test_scenarios:
        # 模拟队列深度
        actual_agents = await scheduler.scale_agents(queue_depth)
        print(f"  {desc} (队列深度={queue_depth}): Agent数量={actual_agents}")
    
    await scheduler.stop()


async def main():
    """主函数 - 运行所有示例"""
    print("\n" + "=" * 70)
    print("Kimi Agent集群调度器 - 完整演示")
    print("=" * 70 + "\n")
    
    # 运行所有示例
    await demo_basic_usage()
    await demo_priority_queue()
    await demo_failure_recovery()
    await demo_checkpoint()
    await demo_load_balancing()
    await demo_dynamic_scaling()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
