#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超高分辨率图像生成系统 - 主入口
Super Resolution Image Generation System - Main Entry

该系统实现从720p输入到1亿-2亿像素输出的超高分辨率图像生成，
采用分块-并行-融合的分布式架构。

作者: AI Assistant
版本: 1.0.0
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('super_resolution.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 导入各模块
try:
    from tiling_module import TilingModule, Tile, TileMetadata
    from super_resolution_module import SuperResolutionModule, UpscaleResult
    from blending_module import BlendingModule, TileInfo
    from quality_assessment_module import QualityAssessmentModule
    from agent_scheduler import AgentScheduler, Task, TaskPriority
except ImportError as e:
    logger.error(f"模块导入失败: {e}")
    logger.error("请确保所有模块文件在同一目录下")
    sys.exit(1)


@dataclass
class PipelineConfig:
    """Pipeline配置类"""
    # 分块参数
    block_size: int = 2048
    overlap_ratio: float = 0.2
    padding_mode: str = 'mirror'
    
    # 超分参数
    target_resolution: str = "100MP"  # 100MP, 150MP, 200MP
    seedream_strength: float = 0.5
    seedream_steps: int = 50
    
    # 融合参数
    blend_method: str = 'laplacian'  # laplacian, poisson, weighted
    num_pyramid_levels: int = 6
    
    # 调度参数
    max_agents: int = 60
    max_concurrent: int = 30
    
    # 质量评估
    enable_qa: bool = True
    qa_device: str = 'cpu'
    
    # API配置
    volc_ak: str = ""
    volc_sk: str = ""
    volc_region: str = "cn-beijing"


@dataclass
class PipelineResult:
    """Pipeline结果类"""
    success: bool
    output_path: Optional[str]
    processing_time: float
    total_blocks: int
    successful_blocks: int
    failed_blocks: int
    quality_score: Optional[float]
    quality_report: Optional[Dict[str, Any]]
    error_message: Optional[str]


class SuperResolutionPipeline:
    """
    超高分辨率图像生成Pipeline
    
    五阶段处理流程：
    1. 图像分块 (Tiling)
    2. 并行超分 (Super-Resolution)
    3. 智能融合 (Blending)
    4. 质量评估 (Quality Assessment)
    5. 结果输出 (Output)
    """
    
    def __init__(self, config: PipelineConfig):
        """
        初始化Pipeline
        
        Args:
            config: Pipeline配置对象
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化各模块
        self.tiling_module = TilingModule(
            block_size=config.block_size,
            overlap_ratio=config.overlap_ratio,
            padding_mode=config.padding_mode
        )
        
        self.blending_module = BlendingModule(
            method=config.blend_method,
            num_levels=config.num_pyramid_levels
        )
        
        self.quality_module = QualityAssessmentModule(
            device=config.qa_device
        )
        
        # 超分模块和调度器在异步上下文中初始化
        self.sr_module: Optional[SuperResolutionModule] = None
        self.scheduler: Optional[AgentScheduler] = None
        
        self.logger.info("Pipeline初始化完成")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.sr_module = SuperResolutionModule(
            ak=self.config.volc_ak,
            sk=self.config.volc_sk,
            region=self.config.volc_region
        )
        await self.sr_module.__aenter__()
        
        self.scheduler = AgentScheduler(
            max_agents=self.config.max_agents,
            max_concurrent=self.config.max_concurrent
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.sr_module:
            await self.sr_module.__aexit__(exc_type, exc_val, exc_tb)
    
    def _calculate_target_size(self, original_size: tuple, target_resolution: str) -> tuple:
        """
        根据目标分辨率计算输出尺寸
        
        Args:
            original_size: 原始图像尺寸 (width, height)
            target_resolution: 目标分辨率标识
            
        Returns:
            目标尺寸 (width, height)
        """
        width, height = original_size
        aspect_ratio = width / height
        
        resolution_map = {
            "100MP": (12245, 8163),  # 3:2画幅
            "150MP": (15000, 10000),
            "200MP": (17320, 11547),
        }
        
        if target_resolution in resolution_map:
            target_w, target_h = resolution_map[target_resolution]
            # 保持原始宽高比
            if aspect_ratio > target_w / target_h:
                target_h = int(target_w / aspect_ratio)
            else:
                target_w = int(target_h * aspect_ratio)
            return (target_w, target_h)
        else:
            # 自定义解析，如 "10000x10000"
            try:
                w, h = map(int, target_resolution.split('x'))
                return (w, h)
            except:
                self.logger.warning(f"无法解析目标分辨率: {target_resolution}，使用默认100MP")
                return (12245, 8163)
    
    async def _process_single_tile(self, tile: Tile, prompt: str) -> Optional[UpscaleResult]:
        """
        处理单个分块
        
        Args:
            tile: 分块对象
            prompt: 生成提示词
            
        Returns:
            超分结果或None
        """
        try:
            # 将分块图像转换为bytes
            import io
            img_bytes = io.BytesIO()
            tile.image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # 调用Seedream进行超分
            result = await self.sr_module.upscale_seedream(
                image=img_bytes,
                prompt=prompt,
                strength=self.config.seedream_strength,
                size=f"{self.config.block_size * 2}x{self.config.block_size * 2}"
            )
            
            return result
        except Exception as e:
            self.logger.error(f"分块 {tile.metadata.block_id} 处理失败: {e}")
            return None
    
    async def _parallel_upscale(self, tiles: List[Tile], prompt: str) -> List[Optional[UpscaleResult]]:
        """
        并行超分处理
        
        Args:
            tiles: 分块列表
            prompt: 生成提示词
            
        Returns:
            超分结果列表
        """
        self.logger.info(f"开始并行超分处理，共 {len(tiles)} 个分块")
        
        # 创建任务
        tasks = []
        for tile in tiles:
            task = Task(
                task_id=tile.metadata.block_id,
                priority=TaskPriority.NORMAL,
                data={
                    'tile': tile,
                    'prompt': prompt,
                    'block_size': self.config.block_size
                }
            )
            tasks.append(task)
            self.scheduler.submit_task(task)
        
        # 并行处理
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_limit(tile: Tile) -> Optional[UpscaleResult]:
            async with semaphore:
                return await self._process_single_tile(tile, prompt)
        
        results = await asyncio.gather(*[
            process_with_limit(tile) for tile in tiles
        ])
        
        successful = sum(1 for r in results if r is not None)
        self.logger.info(f"并行超分完成: {successful}/{len(tiles)} 成功")
        
        return results
    
    async def process(self, 
                      input_path: str, 
                      output_path: str,
                      prompt: str,
                      roi_regions: Optional[List[Dict]] = None) -> PipelineResult:
        """
        执行完整的超分辨率处理Pipeline
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            prompt: 生成提示词
            roi_regions: ROI区域列表（可选）
            
        Returns:
            Pipeline结果
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"开始处理: {input_path}")
        self.logger.info(f"目标分辨率: {self.config.target_resolution}")
        
        try:
            # ========== Stage 1: 图像分块 ==========
            self.logger.info("=== Stage 1: 图像分块 ===")
            tiles = self.tiling_module.split_image(input_path)
            self.logger.info(f"分块完成: 共 {len(tiles)} 个分块")
            
            # 检查缓存
            image_hash = self.tiling_module._compute_image_hash(input_path)
            cached_result = self.tiling_module.restore_from_cache(image_hash)
            
            if cached_result:
                self.logger.info("命中缓存，跳过超分阶段")
                # TODO: 从缓存恢复
            
            # ========== Stage 2: 并行超分 ==========
            self.logger.info("=== Stage 2: 并行超分 ===")
            upscale_results = await self._parallel_upscale(tiles, prompt)
            
            successful_results = [r for r in upscale_results if r is not None]
            failed_count = len(upscale_results) - len(successful_results)
            
            if len(successful_results) == 0:
                raise RuntimeError("所有分块处理失败")
            
            # 将超分结果转换为TileInfo
            tile_infos = []
            for i, (tile, result) in enumerate(zip(tiles, upscale_results)):
                if result:
                    tile_info = TileInfo(
                        image=result.image,
                        position=(tile.metadata.global_x * 2, tile.metadata.global_y * 2),
                        index=i
                    )
                    tile_infos.append(tile_info)
            
            # ========== Stage 3: 智能融合 ==========
            self.logger.info("=== Stage 3: 智能融合 ===")
            
            # 构建重叠映射
            overlap_map = {}
            for i, tile in enumerate(tiles):
                if upscale_results[i]:
                    overlap_map[i] = {
                        'overlap_pixels': int(self.config.block_size * self.config.overlap_ratio),
                        'neighbors': []
                    }
                    # 查找邻居
                    for j, other_tile in enumerate(tiles):
                        if i != j and upscale_results[j]:
                            # 简化的邻居检测逻辑
                            dx = abs(tile.metadata.global_x - other_tile.metadata.global_x)
                            dy = abs(tile.metadata.global_y - other_tile.metadata.global_y)
                            if dx <= self.config.block_size and dy <= self.config.block_size:
                                overlap_map[i]['neighbors'].append(j)
            
            fused_image = self.blending_module.laplacian_fusion(
                tile_infos, 
                overlap_map
            )
            
            self.logger.info(f"融合完成: 输出尺寸 {fused_image.size}")
            
            # ========== Stage 4: 质量评估 ==========
            quality_report = None
            quality_score = None
            
            if self.config.enable_qa:
                self.logger.info("=== Stage 4: 质量评估 ===")
                
                # 加载原始图像
                from PIL import Image
                original_image = Image.open(input_path)
                
                # 计算缩放因子
                scale_factor = fused_image.width / original_image.width
                
                # 全参考评估
                qa_result = self.quality_module.evaluate_full_reference(
                    original=original_image,
                    upscaled=fused_image,
                    scale_factor=scale_factor
                )
                
                # 商业广告专项评估
                commercial_result = self.quality_module.evaluate_commercial(
                    image=fused_image,
                    roi_regions=roi_regions or []
                )
                
                # 生成报告
                quality_report = {
                    'full_reference': qa_result,
                    'commercial': commercial_result,
                    'timestamp': datetime.now().isoformat()
                }
                
                quality_score = qa_result.get('overall_score', 0)
                self.logger.info(f"质量评分: {quality_score:.2f}/100")
            
            # ========== Stage 5: 结果输出 ==========
            self.logger.info("=== Stage 5: 结果输出 ===")
            
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存图像
            if output_path.lower().endswith('.tiff') or output_path.lower().endswith('.tif'):
                fused_image.save(output_path, format='TIFF', compression='lzw')
            elif output_path.lower().endswith('.png'):
                fused_image.save(output_path, format='PNG', compress_level=3)
            else:
                fused_image.save(output_path, quality=95)
            
            # 保存质量报告
            if quality_report:
                report_path = output_path.rsplit('.', 1)[0] + '_qa_report.json'
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(quality_report, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            self.logger.info(f"处理完成，耗时: {processing_time:.2f}秒")
            
            return PipelineResult(
                success=True,
                output_path=output_path,
                processing_time=processing_time,
                total_blocks=len(tiles),
                successful_blocks=len(successful_results),
                failed_blocks=failed_count,
                quality_score=quality_score,
                quality_report=quality_report,
                error_message=None
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline执行失败: {e}", exc_info=True)
            processing_time = time.time() - start_time
            
            return PipelineResult(
                success=False,
                output_path=None,
                processing_time=processing_time,
                total_blocks=0,
                successful_blocks=0,
                failed_blocks=0,
                quality_score=None,
                quality_report=None,
                error_message=str(e)
            )


async def main():
    """主函数示例"""
    # 配置
    config = PipelineConfig(
        block_size=2048,
        overlap_ratio=0.2,
        target_resolution="100MP",
        seedream_strength=0.5,
        blend_method='laplacian',
        max_agents=60,
        max_concurrent=30,
        enable_qa=True,
        volc_ak="your_access_key",
        volc_sk="your_secret_key"
    )
    
    # 执行Pipeline
    async with SuperResolutionPipeline(config) as pipeline:
        result = await pipeline.process(
            input_path="input.jpg",
            output_path="output.tiff",
            prompt="高端商业摄影，8K超高清，专业广告品质，细腻质感，锐利边缘"
        )
        
        if result.success:
            print(f"✅ 处理成功!")
            print(f"   输出文件: {result.output_path}")
            print(f"   处理时间: {result.processing_time:.2f}秒")
            print(f"   分块统计: {result.successful_blocks}/{result.total_blocks} 成功")
            if result.quality_score:
                print(f"   质量评分: {result.quality_score:.2f}/100")
        else:
            print(f"❌ 处理失败: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
