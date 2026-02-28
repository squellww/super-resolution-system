# 超分辨率图像质量评估模块 (Quality Assessment Module)

## 模块概述

本模块实现超高分辨率图像生成系统的完整质量评估体系，支持全参考、无参考和商业广告专项评估。

## 功能特性

### 1. 降采样对比法
- Bicubic降采样（OpenCV INTER_CUBIC）
- 多尺度对比：0.1×（结构色彩）、0.2×（中频）、0.4×（高频）

### 2. 全参考指标
| 指标 | 优良阈值 | 说明 |
|------|---------|------|
| PSNR | >35dB | 峰值信噪比 |
| SSIM | >0.98 | 结构相似性 |
| MS-SSIM | >0.98 | 多尺度SSIM |
| LPIPS | <0.05 | 感知相似度 |

### 3. 无参考指标
| 指标 | 优良阈值 | 说明 |
|------|---------|------|
| NIQE | <3.0 | 自然图像质量评估 |
| BRISQUE | <20.0 | 盲/无参考图像质量评估 |

### 4. 商业广告专项评估
- **细节保真度**：文字清晰度、产品纹理
- **色彩准确性**：品牌色ΔE、肤色自然度
- **视觉舒适度**：无过度锐化、无伪影

## 快速开始

### 安装依赖

```bash
pip install torch torchvision
pip install opencv-python-headless
pip install scikit-image
pip install lpips
pip install numpy
```

### 基本使用

```python
from quality_assessment_module import QualityAssessmentModule
import numpy as np
import cv2

# 初始化评估模块
qam = QualityAssessmentModule(device='cpu')

# 加载图像
original = cv2.imread('original.png')
upscaled = cv2.imread('upscaled.png')

# 全参考评估
metrics = qam.evaluate_full_reference(original, upscaled, scale_factor=4)
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"MS-SSIM: {metrics['ms_ssim']:.4f}")
print(f"LPIPS: {metrics['lpips_vgg']:.4f}")

# 无参考评估
no_ref_metrics = qam.evaluate_no_reference(upscaled)
print(f"NIQE: {no_ref_metrics['niqe']:.2f}")
print(f"BRISQUE: {no_ref_metrics['brisque']:.2f}")

# 商业专项评估
roi_regions = [
    {'type': 'text', 'bbox': [100, 100, 200, 50]},
    {'type': 'product', 'bbox': [300, 200, 150, 150]},
    {'type': 'brand', 'bbox': [50, 400, 100, 80], 'reference_color': (255, 0, 0)}
]
commercial_metrics = qam.evaluate_commercial(upscaled, roi_regions)
print(f"商业综合评分: {commercial_metrics['commercial_score']:.2f}/100")

# 生成报告
all_metrics = {**metrics, **no_ref_metrics, **commercial_metrics}
report = qam.generate_report(all_metrics, report_type='full', output_path='report.txt')
print(report)
```

## API文档

### QualityAssessmentModule

#### `__init__(self, device='cpu', thresholds=None, scale_config=None)`
初始化质量评估模块。

**参数：**
- `device`: 计算设备 ('cpu' 或 'cuda')
- `thresholds`: 自定义质量阈值配置 (QualityThresholds)
- `scale_config`: 自定义多尺度配置 (ScaleConfig)

#### `evaluate_full_reference(original, upscaled, scale_factor=4) -> Dict[str, float]`
全参考质量评估。

**参数：**
- `original`: 原始高分辨率图像 (numpy array)
- `upscaled`: 超分辨率输出图像 (numpy array)
- `scale_factor`: 超分辨率放大倍数

**返回：**
包含PSNR、SSIM、MS-SSIM、LPIPS等指标的字典

#### `evaluate_no_reference(image) -> Dict[str, float]`
无参考质量评估。

**参数：**
- `image`: 待评估图像 (numpy array)

**返回：**
包含NIQE、BRISQUE、锐度、对比度等指标的字典

#### `evaluate_commercial(image, roi_regions=None) -> Dict[str, float]`
商业广告专项评估。

**参数：**
- `image`: 待评估图像 (numpy array)
- `roi_regions`: 感兴趣区域列表，每个区域包含:
  - `type`: 'text' | 'product' | 'brand' | 'face'
  - `bbox`: [x, y, w, h]
  - `reference_color`: 可选的品牌色参考 (R, G, B)

**返回：**
包含商业专项评估指标的字典

#### `generate_report(metrics, report_type='full', output_path=None) -> str`
生成质量评估报告。

**参数：**
- `metrics`: 评估指标字典
- `report_type`: 报告类型 ('full', 'summary', 'json')
- `output_path`: 输出文件路径

**返回：**
格式化的报告字符串

#### `batch_evaluate(image_pairs, scale_factor=4) -> List[Dict[str, float]]`
批量评估。

**参数：**
- `image_pairs`: (原始图像, 超分辨率图像) 元组列表
- `scale_factor`: 超分辨率放大倍数

**返回：**
评估结果列表

## 质量阈值配置

### 默认阈值

```python
from quality_assessment_module import QualityThresholds

thresholds = QualityThresholds(
    PSNR_EXCELLENT=40.0,    # PSNR优秀阈值
    PSNR_GOOD=35.0,         # PSNR良好阈值
    PSNR_FAIR=30.0,         # PSNR一般阈值
    SSIM_EXCELLENT=0.98,    # SSIM优秀阈值
    SSIM_GOOD=0.95,         # SSIM良好阈值
    SSIM_FAIR=0.90,         # SSIM一般阈值
    LPIPS_EXCELLENT=0.02,   # LPIPS优秀阈值
    LPIPS_GOOD=0.05,        # LPIPS良好阈值
    LPIPS_FAIR=0.10,        # LPIPS一般阈值
)

qam = QualityAssessmentModule(thresholds=thresholds)
```

## 报告示例

### 完整报告

```
======================================================================
超分辨率图像质量评估报告
======================================================================
生成时间: 2024-01-15 10:30:00

----------------------------------------------------------------------
【全参考质量指标】
----------------------------------------------------------------------
PSNR:           38.47 dB    [good]
SSIM:           0.9991
MS-SSIM:        0.9991    [excellent]
LPIPS (VGG):    0.0016    [excellent]
LPIPS (Alex):   0.0009

----------------------------------------------------------------------
【降采样对比法 (多尺度)】
----------------------------------------------------------------------
  structure_color (0.1x):
    PSNR: 39.58 dB
    SSIM: 0.9992
  mid_frequency (0.2x):
    PSNR: 39.55 dB
    SSIM: 0.9991
  high_frequency (0.4x):
    PSNR: 39.59 dB
    SSIM: 0.9991

----------------------------------------------------------------------
【无参考质量指标】
----------------------------------------------------------------------
NIQE:           4.77    [good]
BRISQUE:        25.00    [fair]
锐度:           49107.87
对比度:         49.59
色彩丰富度:     62.83

----------------------------------------------------------------------
【商业广告专项评估】
----------------------------------------------------------------------
商业综合评分:   75.00/100

  细节保真度:
    全局锐度:   49107.87
    高频比例:   0.7997

  视觉舒适度:
    锐化评分:   85.00/100
    伪影评分:   90.00/100
    噪声水平:   15.66
    亮度均匀性: 95.74/100

----------------------------------------------------------------------
【综合评估】
----------------------------------------------------------------------
综合质量评分:   85.83/100

----------------------------------------------------------------------
【质量等级说明】
----------------------------------------------------------------------
  excellent - 优秀
  good      - 良好
  fair      - 一般
  poor      - 较差

======================================================================
报告生成完成
======================================================================
```

## 文件结构

```
super_resolution_system/
├── quality_assessment_module.py    # 质量评估模块主代码
├── quality_report.txt              # 示例报告（文本格式）
├── quality_report.json             # 示例报告（JSON格式）
└── README_QualityAssessment.md     # 本文档
```

## 技术细节

### 降采样对比法

降采样对比法是验证超分辨率算法有效性的重要方法：

1. 将超分辨率结果降采样回低分辨率空间
2. 与原始图像的降采样版本进行对比
3. 多尺度对比可以评估不同频率成分的重建质量

### PSNR计算

```
PSNR = 10 * log10(MAX^2 / MSE)
```

其中MAX是像素最大值（通常为255），MSE是均方误差。

### SSIM计算

SSIM从三个维度评估图像相似性：
- **亮度**：图像平均亮度
- **对比度**：图像标准差
- **结构**：归一化后的协方差

### LPIPS计算

LPIPS使用预训练的深度网络（VGG/AlexNet）提取特征，计算特征空间的距离：

```
LPIPS = ||φ(x) - φ(y)||^2
```

其中φ是深度网络的特征提取函数。

## 注意事项

1. **图像格式**：输入图像应为numpy array，值范围为[0, 255]或[0, 1]
2. **设备选择**：GPU加速可显著提升LPIPS计算速度
3. **内存使用**：大尺寸图像可能需要较大的内存
4. **pyiqa库**：如需更精确的NIQE/BRISQUE，可安装pyiqa库

## 许可证

MIT License
