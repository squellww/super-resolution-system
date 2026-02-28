# 图像融合模块 (Blending Module)

## 概述

图像融合模块是超高分辨率图像生成系统的核心组件，提供多种图像融合算法，用于无缝拼接瓦片图像。

## 功能特性

### 1. 拉普拉斯金字塔融合（推荐方案）
- 多分辨率分解与重建
- 距离衰减权重（S型余弦过渡）
- 复杂度：O(N²·4/3)
- 适用于高质量无缝融合

### 2. 泊松融合（备选方案）
- OpenCV seamlessClone封装
- 三种模式：NORMAL, MIXED, MONOCHROME_TRANSFER
- 梯度域求解
- 适用于对象插入

### 3. 加权平均融合（快速方案）
- 线性衰减：α(x) = 1 - x
- 余弦窗：α(x) = 0.5·(1 + cos(π·x))
- 适用于实时预览

### 4. 质量控制
- 接缝检测：SSIM局部计算（窗口16×16，阈值0.95）
- 分级修复：自动选择修复策略
- 色彩一致性：全局直方图匹配 + 局部引导滤波

## 快速开始

### 安装依赖

```bash
pip install numpy opencv-python
```

### 基本使用

```python
from blending_module import BlendingModule, TileInfo
import numpy as np

# 创建融合模块
blender = BlendingModule(method='laplacian', num_levels=6)

# 准备瓦片
tiles = [
    TileInfo(image=tile1, x=0, y=0, row=0, col=0),
    TileInfo(image=tile2, x=400, y=0, row=0, col=1),
    TileInfo(image=tile3, x=0, y=400, row=1, col=0),
    TileInfo(image=tile4, x=400, y=400, row=1, col=1),
]

# 执行融合
result = blender.laplacian_fusion(tiles, output_shape=(800, 800))

# 检测接缝
seams = blender.detect_seams(result, tiles)
print(f"Detected {len(seams)} seams")

# 色彩校正
corrected = blender.color_correction(result, reference_tile=tiles[0].image)
```

## API参考

### BlendingModule类

#### 初始化
```python
BlendingModule(
    method: str = 'laplacian',
    num_levels: int = 6,
    ssim_threshold: float = 0.95,
    use_cuda: bool = False
)
```

#### 主要方法

##### laplacian_fusion
```python
laplacian_fusion(
    tiles: List[Union[np.ndarray, TileInfo]],
    overlap_map: Optional[List[OverlapRegion]] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    weight_type: WeightType = WeightType.COSINE
) -> np.ndarray
```

##### poisson_fusion
```python
poisson_fusion(
    src: np.ndarray,
    dst: np.ndarray,
    mask: Optional[np.ndarray] = None,
    center: Optional[Tuple[int, int]] = None,
    mode: PoissonMode = PoissonMode.NORMAL
) -> np.ndarray
```

##### weighted_average_fusion
```python
weighted_average_fusion(
    tiles: List[Union[np.ndarray, TileInfo]],
    weights: Optional[List[np.ndarray]] = None,
    weight_type: WeightType = WeightType.COSINE,
    output_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray
```

##### detect_seams
```python
detect_seams(
    result: np.ndarray,
    tiles: List[Union[np.ndarray, TileInfo]],
    window_size: int = 16,
    stride: int = 8
) -> List[Seam]
```

##### color_correction
```python
color_correction(
    image: np.ndarray,
    reference_tile: np.ndarray,
    method: str = "histogram",
    local_filter: bool = True
) -> np.ndarray
```

## 性能优化

### CUDA加速
```python
blender = BlendingModule(use_cuda=True)
```

### 并行处理
```python
from blending_module import ParallelBlender

parallel_blender = ParallelBlender(num_workers=4)
results = parallel_blender.blend_tiles_parallel(blender, tile_groups, output_shape)
```

## 算法对比

| 方法 | 速度 | 质量 | 适用场景 |
|------|------|------|----------|
| 拉普拉斯金字塔 | 中等 | 最高 | 高质量输出 |
| 泊松融合 | 较慢 | 高 | 对象插入 |
| 加权平均 | 最快 | 中等 | 实时预览 |

## 示例代码

运行示例：
```bash
python blending_module.py
```

## 技术细节

### 拉普拉斯金字塔融合流程
1. 构建高斯金字塔（下采样）
2. 构建拉普拉斯金字塔（差分）
3. 每层应用距离权重
4. 重建图像（上采样叠加）

### 接缝检测算法
1. 滑动窗口计算局部SSIM
2. 合并相邻低SSIM区域
3. 分级标记严重程度

### 色彩校正流程
1. 全局直方图匹配
2. 局部引导滤波平滑

## 参考文献

1. Burt, P. J., & Adelson, E. H. (1983). A multiresolution spline with application to image mosaics. ACM TOG, 2(4), 217-236.
2. Perez, P., Gangnet, M., & Blake, A. (2003). Poisson image editing. ACM TOG, 22(3), 313-318.
3. He, K., Sun, J., & Tang, X. (2013). Guided image filtering. IEEE TPAMI, 35(6), 1397-1409.

## 许可证

MIT License
