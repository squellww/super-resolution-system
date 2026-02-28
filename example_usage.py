#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超分辨率模块使用示例

本文件演示如何使用SuperResolutionModule进行各种超分操作。
"""

import asyncio
from pathlib import Path

from super_resolution_module import (
    SuperResolutionModule,
    UpscaleConfig,
    UpscaleProvider,
    VeImageXTemplate,
    PromptTemplateManager,
)


# 配置（请替换为您的实际AK/SK）
AK = "your_access_key"
SK = "your_secret_key"


async def demo_basic_seedream():
    """基础Seedream超分演示"""
    print("=" * 60)
    print("演示1: 基础Seedream超分")
    print("=" * 60)
    
    sr = SuperResolutionModule(ak=AK, sk=SK)
    
    try:
        # 构建美妆类Prompt
        prompt = sr.build_prompt(
            category="beauty",
            custom_desc="luxury face cream jar with gold lid"
        )
        print(f"生成的Prompt: {prompt[:100]}...")
        
        # 执行超分
        # result = await sr.upscale_seedream(
        #     image="test_input.jpg",
        #     prompt=prompt,
        #     strength=0.4,
        #     size="4096x4096",
        #     num_inference_steps=35
        # )
        
        # print(f"✓ 超分完成!")
        # print(f"  原始尺寸: {result.original_size}")
        # print(f"  超分尺寸: {result.upscaled_size}")
        # print(f"  放大倍数: {result.scale_factor:.2f}×")
        # print(f"  处理时间: {result.processing_time:.2f}秒")
        
        # result.image.save("output_seedream.png", quality=95)
        
    finally:
        await sr.close()


async def demo_veimagex_fast():
    """veImageX快速超分演示"""
    print("\n" + "=" * 60)
    print("演示2: veImageX快速超分")
    print("=" * 60)
    
    sr = SuperResolutionModule(ak=AK, sk=SK)
    
    try:
        # result = await sr.upscale_veimagex(
        #     image="test_input.jpg",
        #     template_id=VeImageXTemplate.AI_SUPER_RESOLUTION.value,
        #     scale_factor=2.0
        # )
        
        # print(f"✓ veImageX超分完成!")
        # print(f"  处理时间: {result.processing_time:.2f}秒")
        # print(f"  提供商: {result.provider}")
        
        # result.image.save("output_veimagex.png", quality=95)
        
    finally:
        await sr.close()


async def demo_hybrid_strategy():
    """混合策略超分演示"""
    print("\n" + "=" * 60)
    print("演示3: 混合策略超分 (4×)")
    print("=" * 60)
    
    sr = SuperResolutionModule(ak=AK, sk=SK)
    
    try:
        # result = await sr.hybrid_upscale(
        #     image="test_input.jpg",
        #     target_scale=4.0,
        #     category="3c",
        #     custom_desc="premium smartphone"
        # )
        
        # print(f"✓ 混合超分完成!")
        # print(f"  原始尺寸: {result.original_size}")
        # print(f"  超分尺寸: {result.upscaled_size}")
        # print(f"  实际放大倍数: {result.scale_factor:.2f}×")
        # print(f"  总处理时间: {result.processing_time:.2f}秒")
        
        # # 打印处理历史
        # history = result.metadata.get('processing_history', [])
        # print(f"  处理阶段:")
        # for stage in history:
        #     print(f"    - {stage['stage']}: {stage['provider']} ({stage['time']:.2f}s)")
        
        # result.image.save("output_hybrid.png", quality=95)
        
    finally:
        await sr.close()


async def demo_retry_mechanism():
    """重试机制演示"""
    print("\n" + "=" * 60)
    print("演示4: 指数退避重试机制")
    print("=" * 60)
    
    sr = SuperResolutionModule(ak=AK, sk=SK)
    
    try:
        print("使用retry_with_backoff包装超分函数...")
        print("重试策略: 1s → 2s → 4s")
        
        # result = await sr.retry_with_backoff(
        #     sr.upscale_seedream,
        #     "test_input.jpg",
        #     "high quality product photo",
        #     0.5,
        #     "4096x4096",
        #     max_retries=3
        # )
        
        # print(f"✓ 重试后超分成功!")
        # result.image.save("output_retry.png", quality=95)
        
    except Exception as e:
        print(f"✗ 所有重试均失败: {e}")
        
    finally:
        await sr.close()


async def demo_prompt_templates():
    """Prompt模板系统演示"""
    print("\n" + "=" * 60)
    print("演示5: Prompt模板系统")
    print("=" * 60)
    
    # 列出所有可用模板
    categories = PromptTemplateManager.list_categories()
    print(f"可用行业类别: {categories}")
    
    # 演示各类别Prompt构建
    test_categories = ["beauty", "3c", "food", "fashion", "jewelry"]
    
    for category in test_categories:
        prompt = PromptTemplateManager.build_prompt(
            category=category,
            custom_desc=f"sample {category} product"
        )
        print(f"\n[{category}]")
        print(f"  {prompt[:120]}...")


async def demo_config_based():
    """基于配置的超分演示"""
    print("\n" + "=" * 60)
    print("演示6: 基于配置的超分")
    print("=" * 60)
    
    sr = SuperResolutionModule(ak=AK, sk=SK)
    
    try:
        # 创建不同的配置
        configs = [
            UpscaleConfig(
                provider=UpscaleProvider.SEEDREAM,
                target_scale=4.0,
                strength=0.4,
                num_inference_steps=35
            ),
            UpscaleConfig(
                provider=UpscaleProvider.VEIMAGEX,
                target_scale=2.0
            ),
            UpscaleConfig(
                provider=UpscaleProvider.HYBRID,
                target_scale=4.0,
                strength=0.5
            )
        ]
        
        for i, config in enumerate(configs):
            print(f"\n配置 {i+1}: {config.provider.value}")
            print(f"  目标放大倍数: {config.target_scale}×")
            print(f"  强度: {config.strength}")
            print(f"  推理步数: {config.num_inference_steps}")
            
            # result = await sr.upscale("test_input.jpg", config)
            # print(f"✓ 超分完成: {result.upscaled_size}")
            
    finally:
        await sr.close()


async def demo_context_manager():
    """上下文管理器演示"""
    print("\n" + "=" * 60)
    print("演示7: 异步上下文管理器")
    print("=" * 60)
    
    async with SuperResolutionModule(ak=AK, sk=SK) as sr:
        print("使用上下文管理器自动管理资源...")
        
        # result = await sr.upscale_seedream(
        #     image="test_input.jpg",
        #     prompt="high quality image",
        #     strength=0.5
        # )
        
        # print(f"✓ 超分完成!")
    
    print("上下文退出，资源已自动释放")


def main():
    """主函数"""
    print("超分辨率模块使用示例")
    print("=" * 60)
    print(f"注意: 请先在代码中设置您的AK/SK后再运行")
    print("=" * 60)
    
    # 运行所有演示
    demos = [
        demo_basic_seedream,
        demo_veimagex_fast,
        demo_hybrid_strategy,
        demo_retry_mechanism,
        demo_prompt_templates,
        demo_config_based,
        demo_context_manager,
    ]
    
    for demo in demos:
        try:
            asyncio.run(demo())
        except Exception as e:
            print(f"演示失败: {e}")


if __name__ == "__main__":
    main()
