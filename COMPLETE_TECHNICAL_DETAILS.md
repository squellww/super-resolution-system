# 超分辨率系统 - 完整技术细节文档

> 版本: 1.0.0  
> 创建日期: 2026-02-28  
> 用途: 供DeepResearch进行深度优化

---

## 📁 一、项目文件结构

```
super_resolution_system/
├── 核心模块 (必须理解)
│   ├── tiling_module.py          (3,695行) - 图像分块
│   ├── super_resolution_module.py (1,152行) - 超分API
│   ├── blending_module.py         (1,698行) - 图像融合
│   └── main.py                     (483行) - Pipeline主入口
│
├── 辅助模块
│   ├── agent_scheduler.py        (1,365行) - 任务调度
│   ├── quality_assessment_module.py (1,284行) - 质量评估
│   ├── ark_api_module.py           (90行) - ARK API封装
│   └── config.py                  (300行) - 配置
│
├── Web界面
│   ├── app.py                     (152行) - Streamlit主入口
│   └── pages/                     - 子页面
│
└── 文档
    ├── README.md
    ├── TECHNICAL_ARCHITECTURE.md
    └── VOLCANO_ENGINE_INTEGRATION.md
```

---

## 🔧 二、核心模块详解

### 2.1 TilingModule (图像分块模块)

**文件**: `tiling_module.py`  
**功能**: 将大图像分割为多个小块，每块可独立处理

#### 核心数据结构

```python
@dataclass
class TileMetadata:
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    global_x: int = 0           # 在原始图像中的X坐标 (关键!)
    global_y: int = 0           # 在原始图像中的Y坐标 (关键!)
    input_w: int = 2048         # 输入块宽度
    input_h: int = 2048         # 输入块高度
    output_w: int = 4096        # 输出块宽度 (Seedream限制)
    output_h: int = 4096        # 输出块高度
    overlap_top: int = 0        # 顶部重叠像素
    overlap_bottom: int = 0     # 底部重叠像素
    overlap_left: int = 0       # 左侧重叠像素
    overlap_right: int = 0      # 右侧重叠像素
    status: TileStatus = TileStatus.PENDING

@dataclass
class Tile:
    metadata: TileMetadata
    data: np.ndarray            # 图像数据 (H, W, C)
    mask: Optional[np.ndarray] = None
    cache_path: Optional[str] = None
```

#### 关键方法

**split_image()** - 图像分块主方法
```python
def split_image(self, image_path: str, save_metadata: bool = True) -> List[Tile]:
    """
    将图像分割为多个重叠的块
    
    流程:
    1. 用OpenCV加载图像 (BGR格式)
    2. 转换为RGB (cv2.COLOR_BGR2RGB) [已修复]
    3. 计算分块位置 (_calculate_tile_positions)
    4. 对每个位置提取图像块
    5. 添加填充使所有块大小一致
    6. 创建Tile对象，设置global_x/y [已修复]
    7. 建立邻居关系 (_build_neighbor_relationships)
    8. 返回Tile列表
    """
```

**_calculate_tile_positions()** - 计算分块位置
```python
def _calculate_tile_positions(self, image_width: int, image_height: int) -> List[Tuple[int, int, int, int]]:
    """
    计算所有分块的位置
    
    算法:
    - step = block_size - overlap_pixels
    - 从(0,0)开始，每隔step像素一个块
    - 边界块可能小于标准大小
    
    返回: [(x, y, w, h), ...]
    """
```

#### 当前配置
```python
block_size = 1024          # 输入块大小 (像素)
overlap_ratio = 0.2        # 20%重叠
overlap_pixels = 204       # 1024 * 0.2
padding_mode = 'mirror'    # 边缘填充模式
output_scale = 2.0         # 默认输出缩放 (2x)
```

#### 已修复的问题
1. ✅ BGR到RGB颜色转换
2. ✅ global_x/y设置到TileMetadata

---

### 2.2 SuperResolutionModule (超分辨率模块)

**文件**: `super_resolution_module.py`  
**功能**: 调用外部API进行图像超分

#### 支持的提供商

```python
class UpscaleProvider(Enum):
    SEEDREAM = "seedream"    # 火山引擎Seedream 4.0 (AI生成)
    VEIMAGEX = "veimagex"    # 火山引擎veImageX (CNN超分)
    HYBRID = "hybrid"        # 混合策略
```

#### 核心配置

```python
# 火山引擎ARK API配置 (当前提供的)
ARK_API_KEY = "sk-xIr6z0QlYiu498lwe406xbeuxXeIE6Mp6neFxkhABigECvQ9"
ARK_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
ARK_MODEL = "ep-20260228221135-66v8k"

# Seedream配置 (config.py中)
seedream_endpoint = "https://operator.las.cn-beijing.volces.com/api/v1/online/images/generations"
seedream_model = "doubao-seedream-4-0-250828"
```

#### 核心方法

**upscale_seedream()** - Seedream超分
```python
async def upscale_seedream(
    self,
    image: Union[str, Image.Image],
    prompt: str,
    strength: float = 0.5,
    size: str = "4096x4096",
    num_inference_steps: int = 50,
    seed: Optional[int] = None
) -> SuperResolutionResult:
    """
    使用Seedream 4.0进行AI超分
    
    参数:
    - image: 输入图像路径或PIL Image
    - prompt: 描述图像内容和质量要求的提示词
    - strength: 生成强度 0.0-1.0 (0.5平衡，0.8创意)
    - size: 输出尺寸 ("4096x4096", "2K", "1024x1024")
    - num_inference_steps: 推理步数 (50-100，越多细节越好)
    - seed: 随机种子 (保持一致性)
    
    返回:
    - SuperResolutionResult: 包含超分后的图像和元数据
    
    当前问题: API认证失败 (401 Unauthorized)
    """
```

**build_prompt()** - 构建提示词
```python
def build_prompt(
    self,
    category: str = "general",
    custom_desc: str = "",
    quality_level: str = "8K"
) -> str:
    """
    根据类别构建专业的超分提示词
    
    支持的类别:
    - beauty: 美妆/化妆品
    - 3c: 数码产品
    - food: 美食
    - fashion: 时尚服装
    - jewelry: 珠宝
    - furniture: 家具
    - automotive: 汽车
    - general: 通用
    
    返回示例:
    "高端化妆品商业摄影，柔光棚拍，8K超高清，
     细腻肤质，专业广告品质，锐利边缘，印刷级精度"
    """
```

#### Prompt模板 (config.py)

```python
prompt_templates = {
    'beauty': {
        'subject': '高端化妆品商业摄影，柔光棚拍',
        'style': '8K超高清，细腻肤质，专业广告品质，无噪点',
        'quality': '锐利边缘，精确色彩还原，印刷级精度',
        'negative': '模糊，变形，多余元素，色彩偏移，压缩伪影'
    },
    '3c': {
        'subject': '精密数码产品摄影，科技感十足',
        'style': '金属光泽，精密工艺，未来感设计',
        'quality': '超高清细节，材质真实感，专业灯光',
        'negative': '模糊，反光过曝，塑料感，低质量'
    },
    # ... 更多类别
}
```

#### 当前限制
- ❌ API认证失败 (401)
- ❌ 无法实现真正的AI超分
- ❌ 当前只能用传统插值 (Lanczos/Bicubic)

---

### 2.3 BlendingModule (图像融合模块)

**文件**: `blending_module.py`  
**功能**: 将多个超分后的块无缝融合为完整图像

#### 融合算法

```python
class BlendMethod(Enum):
    LAPLACIAN = "laplacian"    # 拉普拉斯金字塔 (推荐)
    POISSON = "poisson"        # 泊松融合
    WEIGHTED = "weighted"      # 加权融合
    DIRECT = "direct"          # 直接粘贴 (当前使用)
```

#### 拉普拉斯金字塔融合 (待完善)

```python
def laplacian_blend(
    self,
    tiles: List[TileInfo],
    output_size: Tuple[int, int]
) -> Image.Image:
    """
    拉普拉斯金字塔融合算法
    
    步骤:
    1. 为每个块创建高斯金字塔 (多层模糊)
    2. 创建拉普拉斯金字塔 (每层 = 该层高斯 - 下层高斯上采样)
    3. 在每一层创建权重图 (距离中心越近权重越高)
    4. 每层融合: 加权平均所有块的拉普拉斯金字塔
    5. 从上到下重建图像
    
    优点:
    - 无缝过渡
    - 保留高频细节
    - 颜色平滑
    """
```

#### 当前实现 (简化版)

测试脚本中使用的是简单粘贴:
```python
output_img = Image.new('RGB', (output_width, output_height))

for tile_info in upscaled_tiles:
    tile_img = tile_info['image']
    out_x = int(tile_info['global_x'] * scale_factor)
    out_y = int(tile_info['global_y'] * scale_factor)
    output_img.paste(tile_img, (out_x, out_y))  # 简单覆盖
```

**问题**: 重叠区域直接覆盖，没有渐变融合，可能有接缝

---

### 2.4 Main Pipeline (主流程)

**文件**: `main.py`

#### Pipeline执行流程

```python
class SuperResolutionPipeline:
    def __init__(self, config: PipelineConfig):
        self.tiling_module = TilingModule(...)
        self.blending_module = BlendingModule(...)
        self.quality_module = QualityAssessmentModule(...)
        # sr_module在async上下文初始化
    
    async def process(self, input_path, output_path, prompt) -> PipelineResult:
        """
        完整处理流程:
        
        1. 图像分块
           tiles = tiling_module.split_image(input_path)
        
        2. 并行超分 (通过AgentScheduler)
           for tile in tiles:
               task = Task(tile, prompt)
               scheduler.submit(task)
           results = await scheduler.wait_all()
        
        3. 图像融合
           output = blending_module.laplacian_blend(results)
        
        4. 质量评估
           metrics = quality_module.compute_metrics(input, output)
        
        5. 保存结果
           output.save(output_path)
        
        return PipelineResult(...)
        """
```

---

## 🔄 三、数据流详解

### 3.1 输入到输出的完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: 用户图像 (如 1920x1080 PNG/JPG)                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: TilingModule.split_image()                             │
│  - 加载图像 (OpenCV BGR → RGB转换)                               │
│  - 计算分块位置 (考虑重叠)                                        │
│  - 提取每个块 (numpy array HxWx3)                                │
│  - 创建Tile对象 (包含global_x/y位置信息)                         │
│                                                                 │
│  OUTPUT: List[Tile]                                              │
│  例如: 1920x1080 → 6个块 (2x3网格)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: SuperResolutionModule.upscale_seedream()               │
│  - 对每个Tile构建Prompt                                          │
│  - 调用Seedream API (当前: 用传统插值代替)                         │
│  - 超分到目标尺寸 (如 4096x4096)                                  │
│                                                                 │
│  OUTPUT: List[SuperResolutionResult]                             │
│  每个块从 1024x1024 → 4096x4096                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: BlendingModule.laplacian_blend()                       │
│  - 创建输出画布 (目标尺寸)                                        │
│  - 对每个超分块计算在画布中的位置                                  │
│  - 融合重叠区域 (拉普拉斯金字塔/泊松/加权)                         │
│                                                                 │
│  OUTPUT: PIL Image                                               │
│  例如: 合并6个块为完整图像 12245x8163                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: QualityAssessmentModule.compute_metrics()              │
│  - 计算PSNR (峰值信噪比)                                         │
│  - 计算SSIM (结构相似性)                                         │
│  - 计算LPIPS (感知相似度)                                        │
│                                                                 │
│  OUTPUT: Dict[str, float]                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: TIFF文件 (100MP-200MP)                                  │
│  - 保存为TIFF格式 (支持LZW压缩)                                  │
│  - 保存预览图 (JPEG)                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 关键数据转换

| 阶段 | 数据格式 | 尺寸变化 | 备注 |
|------|---------|---------|------|
| 输入 | PIL Image | 1920x1080 | RGB格式 |
| 分块 | np.ndarray | 1024x1024 | uint8, HWC |
| 超分 | PIL Image | 4096x4096 | RGB格式 |
| 融合 | PIL Image | 12245x8163 | 最终输出 |

---

## 🐛 四、当前缺陷和限制

### 4.1 严重问题

| 问题 | 状态 | 影响 | 原因 |
|------|------|------|------|
| API认证失败 | ❌ 未解决 | 无法使用AI超分 | API Key格式或配额问题 |
| 无真正AI超分 | ❌ 未解决 | 只是插值放大 | API不可用 |
| 融合算法简化 | ⚠️ 部分 | 可能有接缝 | 使用简单粘贴 |

### 4.2 性能问题

| 问题 | 描述 |
|------|------|
| 单线程处理 | 未实现真正的并行超分 |
| 内存占用高 | 大图像 (100MP) 占用大量内存 |
| 速度慢 | 传统插值算法较慢 |

### 4.3 功能缺失

| 功能 | 优先级 | 说明 |
|------|-------|------|
| 真正的AI超分 | 🔴 高 | 需要修复API或换用其他服务 |
| 多级超分 | 🔴 高 | 2x→2x→2x 渐进式放大 |
| 风格一致性 | 🟡 中 | 所有块使用相同seed |
| GPU加速 | 🟢 低 | 使用CUDA加速处理 |

---

## 🎯 五、期望的优化方向

### 5.1 核心目标

实现真正的 **AI驱动超分辨率**，不只是像素放大，而是细节重建。

### 5.2 具体优化点

#### 1. 修复API集成
```python
# 当前: API调用失败
# 期望: 成功调用Seedream 4.0

async def upscale_with_seedream(image, prompt):
    response = await http.post(
        ARK_ENDPOINT,
        headers={"Authorization": f"Bearer {ARK_API_KEY}"},
        json={
            "model": ARK_MODEL,
            "prompt": prompt,
            "size": "4096x4096",
            "seed": 42  # 保持一致性
        }
    )
    return response.data[0].url
```

#### 2. 多级超分策略
```
输入 1920x1080
    ↓ 2x Seedream
  3840x2160
    ↓ 2x Seedream  
  7680x4320
    ↓ 1.6x 插值/AI
  12245x8163 (100MP)
```

#### 3. 完善的融合算法
```python
# 实现真正的拉普拉斯金字塔融合
def laplacian_blend(tiles):
    # 1. 为每个块创建高斯金字塔 (6层)
    # 2. 创建拉普拉斯金字塔
    # 3. 每层创建权重图 (距离中心权重高)
    # 4. 逐层融合
    # 5. 重建图像
```

#### 4. 内容感知优化
```python
# 分析图像内容，选择最佳策略
content_type = analyze_content(image)  # beauty/3c/food/...
prompt = build_prompt(content_type)    # 使用对应模板
strength = get_optimal_strength(content_type)  # 动态调整强度
```

---

## 📋 六、关键代码片段

### 6.1 分块逻辑 (tiling_module.py)

```python
def _calculate_tile_positions(self, image_width, image_height):
    positions = []
    step = self.block_size - self.overlap_pixels  # 1024 - 204 = 820
    
    num_tiles_x = max(1, int(np.ceil((image_width - self.overlap_pixels) / step)))
    num_tiles_y = max(1, int(np.ceil((image_height - self.overlap_pixels) / step)))
    
    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            x = tile_x * step
            y = tile_y * step
            w = min(self.block_size, image_width - x)
            h = min(self.block_size, image_height - y)
            positions.append((x, y, w, h))
    
    return positions
```

### 6.2 当前超分逻辑 (简化版)

```python
def smart_upscale(image, scale_factor):
    # 1. Bicubic插值
    upscaled = image.resize(new_size, Image.Resampling.BICUBIC)
    
    # 2. 边缘检测 + 自适应锐化
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = cv2.GaussianBlur(edges, (5, 5), 0)
    
    # 3. 根据边缘强度混合锐化
    sharpened = cv2.filter2D(img_array, -1, kernel)
    result = original * (1 - edge_mask) + sharpened * edge_mask
    
    return result
```

### 6.3 融合逻辑 (简化版)

```python
def simple_blend(tiles, output_size):
    output = Image.new('RGB', output_size)
    
    for tile in tiles:
        x = int(tile.global_x * scale_factor)
        y = int(tile.global_y * scale_factor)
        output.paste(tile.image, (x, y))  # 简单粘贴
    
    return output
```

---

## 🔑 七、API配置信息

### 火山引擎配置

```python
# ARK API (图像生成)
ARK_API_KEY = "sk-xIr6z0QlYiu498lwe406xbeuxXeIE6Mp6neFxkhABigECvQ9"
ARK_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
ARK_MODEL = "ep-20260228221135-66v8k"

# Seedream (超分)
SEEDREAM_ENDPOINT = "https://operator.las.cn-beijing.volces.com/api/v1/online/images/generations"
SEEDREAM_MODEL = "doubao-seedream-4-0-250828"

# veImageX (快速超分)
VEIMAGEX_ENDPOINT = "https://imagex.volcengineapi.com"
```

### API调用示例

```bash
curl -X POST https://ark.cn-beijing.volces.com/api/v3/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -d '{
    "model": "ep-20260228221135-66v8k",
    "prompt": "高端产品摄影，8K超高清，专业广告品质",
    "size": "4096x4096",
    "seed": 42,
    "watermark": false
  }'
```

---

## 📊 八、测试数据

### 测试图片
- **路径**: `C:\Users\squel\Pictures\donut base08.png`
- **尺寸**: 1920x1080 (2.07MP)
- **格式**: PNG

### 预期输出
- **尺寸**: 12245x8163 (100MP)
- **格式**: TIFF
- **用途**: 广告级印刷

---

## 🎓 九、关键概念解释

### 9.1 超分辨率 vs 图像放大

| 特性 | 传统放大 | AI超分 |
|------|---------|--------|
| 原理 | 插值算法 (双线性/双三次/Lanczos) | 深度学习生成 |
| 细节 | 平滑、模糊 | 锐利、真实 |
| 纹理 | 丢失 | 重建 |
| 速度 | 快 | 慢 |
| 质量 | 低-中 | 高 |

### 9.2 分块-融合策略

**为什么分块?**
- Seedream 4.0最大输出4096x4096
- 要达到100MP需要多块拼接

**为什么重叠?**
- 消除块间接缝
- 提供融合缓冲区

**为什么融合?**
- 简单拼接有明显边界
- 融合使过渡自然

---

## 🚀 十、给DeepResearch的优化建议

### 优先级1 (必须)
1. **修复API认证** - 确保可以调用Seedream
2. **实现多级超分** - 2x→2x→2x渐进放大
3. **完善融合算法** - 真正的拉普拉斯金字塔

### 优先级2 (重要)
4. **风格一致性** - 所有块使用相同seed
5. **内容感知** - 自动选择最佳Prompt
6. **并行处理** - 多线程/多进程加速

### 优先级3 (可选)
7. **GPU加速** - 使用CUDA
8. **批量处理** - 多图并行
9. **智能分块** - 基于内容自适应分块

---

## 📞 十一、联系信息

**GitHub**: https://github.com/squellww/super-resolution-system  
**邮箱**: squellwww@me.com

---

*本文档包含所有技术细节，供DeepResearch进行深度优化*
