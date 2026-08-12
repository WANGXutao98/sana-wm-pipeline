# Stage 3 中 UniMatch 和 DOVER 的工作原理分析

## 📋 概述

Stage 3 是 SANA-WM 论文（arXiv:2605.15178v1）中的**视觉质量评估阶段**，使用两个核心模型：
1. **UniMatch**：光流计算（运动幅度评估）
2. **DOVER**：视频质量评估

**目标**：根据论文 Table 6 的阈值筛选出高质量视频。

---

## 🎯 UniMatch：光流计算

### 输入
```python
# 输入：两帧 RGB 图像
frame1: np.ndarray  # shape (H, W, 3), dtype uint8
frame2: np.ndarray  # shape (H, W, 3), dtype uint8
```

### 输出
```python
# 输出：光流场
flow: np.ndarray  # shape (H, W, 2), dtype float32
# flow[:,:,0] = x 方向位移
# flow[:,:,1] = y 方向位移
```

### 工作原理

**1. 采样策略（论文固定）**

根据代码注释和实现（`visual_metrics.py:17-20`）：
```python
UNIMATCH_SAMPLE_EVERY_S = 0.5  # 每 0.5 秒采样一对帧
UNIMATCH_WINDOW_S = 60         # 只在前 60 秒窗口内采样
```

**采样逻辑**：
```python
def enumerate_unimatch_pairs(n_frames, fps=16, window_s=60):
    """
    在前 60 秒内，每隔 0.5 秒采样一对相邻帧
    
    例如：16 fps，60 秒 = 960 帧
    采样：(0,1), (8,9), (16,17), ..., (952,953)
    总共：120 对帧
    """
    max_frames = min(n_frames, window_s * fps)
    step = int(fps * 0.5)  # 8 帧
    
    pairs = []
    for i in range(0, max_frames - 1, step):
        pairs.append((i, i + 1))  # 相邻帧对
    return pairs
```

**2. 光流计算**

```python
def unimatch_flow_magnitude(frames_rgb, flow_fn, fps=16):
    """
    计算所有采样帧对的光流幅度，返回平均值
    """
    pairs = enumerate_unimatch_pairs(len(frames_rgb), fps)
    
    magnitudes = []
    for i, j in pairs:
        # 调用 UniMatch 模型计算光流
        flow = flow_fn(frames_rgb[i], frames_rgb[j])
        
        # 计算光流幅度（欧氏距离）
        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        
        # 取平均
        avg_magnitude = magnitude.mean()
        magnitudes.append(avg_magnitude)
    
    # 返回所有帧对的平均光流幅度
    return np.mean(magnitudes)
```

**3. 物理意义**

```
光流幅度 = 像素的平均移动距离（单位：像素/帧）

高光流（> 80）：
- 快速运动（跑步、飞行、赛车）
- 大幅度相机运动（快速摇镜）

中等光流（3-80）：
- 正常步行
- 平稳相机移动

低光流（< 3）：
- 静态场景
- 微小运动
- 可能是"boring"视频（论文要求过滤）
```

---

## 🎯 DOVER：视频质量评估

### 输入
```python
# 输入：一段视频片段（chunk）
frames_rgb: np.ndarray  # shape (T, H, W, 3), dtype uint8
# T = 80 帧（5 秒 @ 16fps）
```

### 输出
```python
# 输出：质量分数
dover_score: float  # range [0, 1], 越高越好
```

### 工作原理

**1. 分块策略（论文固定）**

根据代码（`visual_metrics.py:20`）：
```python
DOVER_CHUNK_S = 5  # 5 秒为一个 chunk
```

**分块逻辑**：
```python
def dover_score(frames_rgb, dover_fn, fps=16, chunk_s=5):
    """
    将视频分成非重叠的 5 秒 chunk，分别评估后取平均
    """
    chunk_size = fps * chunk_s  # 80 帧
    chunks = []
    
    for i in range(0, len(frames_rgb), chunk_size):
        end = min(i + chunk_size, len(frames_rgb))
        if end - i >= chunk_size:  # 只处理完整的 chunk
            chunks.append((i, end))
    
    scores = []
    for start, end in chunks:
        chunk = frames_rgb[start:end]
        score = dover_fn(chunk)  # 调用 DOVER 模型
        scores.append(score)
    
    # 返回所有 chunk 的平均分数
    return np.mean(scores)
```

**2. DOVER 模型内部**

DOVER 模型包含两个评估维度：

```python
def dover_fn(frames_rgb):
    """
    DOVER 模型评估两个维度：
    
    1. Technical Quality（技术质量）：
       - 压缩伪影（块效应、蚊式噪声）
       - 运动模糊
       - 噪声水平
       - 编码质量
    
    2. Aesthetic Quality（美学质量）：
       - 构图
       - 光照
       - 色彩和谐
       - 视觉吸引力
    
    最终分数 = (technical_score + aesthetic_score) / 2
    """
    # 输入：(T, H, W, 3) uint8
    # 转换：(1, 3, T, H, W) float32, normalized [0,1]
    
    views = {
        "technical": input_tensor,
        "aesthetic": input_tensor,
    }
    
    results = model(views)
    # results = [technical_score, aesthetic_score]
    
    return (results[0] + results[1]) / 2
```

**3. 分数解释**

根据 Table 6 的阈值：

```
DOVER 分数范围：[0, 1]

高质量（> 0.65）：
- 清晰无噪声
- 良好的编码质量
- 视觉吸引力强

中等质量（0.35-0.65）：
- 轻微压缩伪影
- 可接受的视觉质量

低质量（< 0.35）：
- 严重压缩伪影
- 模糊或噪声过多
- 构图差
- 论文要求过滤
```

---

## 📊 Table 6 筛选逻辑

### 整体流程

```python
def evaluate(source: str, scores: dict, cfg: dict):
    """
    根据论文 Table 6 的阈值判断视频是否接受
    
    规则：所有适用的过滤器都必须通过
    """
    rules = cfg["per_source"][source]
    reasons = []
    
    # 检查每个指标
    for metric_name, threshold in rules.items():
        if threshold is None:  # "—" in paper，不检查
            continue
        
        value = scores.get(metric_name)
        if not in_range(value, threshold):
            reasons.append(f"{metric_name} out of range")
    
    return {
        "accepted": len(reasons) == 0,  # 必须全部通过
        "reasons": reasons
    }
```

### 具体阈值（基于 Table 6）

根据 `table6_thresholds.yaml` 的实际配置：

#### 1. OmniWorld（游戏场景）

```yaml
vmaf_motion:      [0.5, 100]   # 运动幅度检查
unimatch_flow:    [3, 100]     # 光流范围：允许大运动
dover:            [0.35, 1.0]  # 质量要求：中等
color_saturation: null         # 不检查饱和度
scene_cuts_max:   null         # 允许场景切换
vlm_entity:       [0, 10]      # 实体数量：≤10
vlm_quality:      [0.5, 1.5]   # VLM 质量检查
```

**解释**：
- 游戏场景运动幅度大，允许高光流（最高 100）
- 质量要求适中（DOVER ≥ 0.35）
- 需要 VLM 检查实体和质量

#### 2. Sekai_Game_Drone（无人机视角）

```yaml
vmaf_motion:      [1.0, 50]    # 运动幅度：中高
unimatch_flow:    [5, 100]     # 光流：中高（飞行视角）
dover:            [0.65, 1.0]  # 质量要求：高
color_saturation: [30, 180]    # 饱和度检查
scene_cuts_max:   0            # 不允许场景切换
vlm_entity:       null         # 不检查实体
vlm_quality:      null         # 不用 VLM
```

**解释**：
- 无人机视角要求高质量（DOVER ≥ 0.65）
- 不允许场景切换（连续飞行）
- 饱和度要求（避免过暗/过曝）

#### 3. DL3DV（室内 3D 扫描）

```yaml
vmaf_motion:      [0.5, 50]
unimatch_flow:    [3, 80]
dover:            [0.40, 1.0]  # 质量要求：中等
color_saturation: [0, 180]
scene_cuts_max:   1            # 最多 1 次切换
vlm_entity:       null
vlm_quality:      null
```

**解释**：
- 室内场景运动较慢
- 质量要求适中
- 允许少量场景切换

---

## 🔬 实际案例分析

### 案例 1：高质量游戏视频（接受）

```python
source = "OmniWorld"
scores = {
    "vmaf_motion": 25.3,        # ✅ in [0.5, 100]
    "unimatch_flow": 45.2,      # ✅ in [3, 100]
    "dover": 0.68,              # ✅ in [0.35, 1.0]
    "color_saturation": 85.4,   # ✅ null（不检查）
    "scene_cuts": 0,            # ✅ null（不检查）
    "vlm_entity_count": 5,      # ✅ in [0, 10]
    "vlm_quality": 0.8,         # ✅ in [0.5, 1.5]
}

result = evaluate(source, scores, table6_cfg)
# result = {"accepted": True, "reasons": []}
```

**结论**：所有指标通过，视频被接受。

---

### 案例 2：低质量视频（拒绝）

```python
source = "Sekai_Game_Drone"
scores = {
    "vmaf_motion": 3.2,         # ✅ in [1.0, 50]
    "unimatch_flow": 8.5,       # ✅ in [5, 100]
    "dover": 0.32,              # ❌ NOT in [0.65, 1.0]
    "color_saturation": 45.3,   # ✅ in [30, 180]
    "scene_cuts": 0,            # ✅ ≤ 0
}

result = evaluate(source, scores, table6_cfg)
# result = {
#     "accepted": False, 
#     "reasons": ["dover=0.32 not in [0.65, 1.0]"]
# }
```

**结论**：DOVER 分数过低（0.32 < 0.65），视频被拒绝。

---

### 案例 3：静态场景（拒绝）

```python
source = "OmniWorld"
scores = {
    "vmaf_motion": 0.3,         # ❌ NOT in [0.5, 100]
    "unimatch_flow": 1.2,       # ❌ NOT in [3, 100]
    "dover": 0.85,              # ✅ in [0.35, 1.0]
    "color_saturation": 120.0,  # ✅ null
    "scene_cuts": 0,            # ✅ null
    "vlm_entity_count": 3,      # ✅ in [0, 10]
    "vlm_quality": 0.9,         # ✅ in [0.5, 1.5]
}

result = evaluate(source, scores, table6_cfg)
# result = {
#     "accepted": False,
#     "reasons": [
#         "vmaf_motion=0.3 not in [0.5, 100]",
#         "unimatch_flow=1.2 not in [3, 100]"
#     ]
# }
```

**结论**：运动幅度过小（静态或"boring"场景），即使质量高也被拒绝。

---

## 📐 数学细节

### UniMatch 光流幅度计算

```python
# 1. 计算光流（UniMatch 模型输出）
flow = unimatch_model(frame1, frame2)
# flow.shape = (H, W, 2)

# 2. 计算每个像素的位移幅度
magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
# magnitude.shape = (H, W)

# 3. 取全图平均
avg_magnitude = magnitude.mean()

# 4. 对所有采样帧对重复上述过程，最后取平均
final_score = np.mean([avg_mag_1, avg_mag_2, ..., avg_mag_N])
```

**示例**：
```
帧对 1：平均光流 = 45.3 像素/帧
帧对 2：平均光流 = 48.1 像素/帧
...
帧对 120：平均光流 = 42.8 像素/帧

最终分数 = (45.3 + 48.1 + ... + 42.8) / 120 = 45.2
```

---

### DOVER 质量评分

```python
# 1. 输入预处理
frames_rgb = frames_rgb.astype(np.float32) / 255.0  # 归一化到 [0,1]
input_tensor = to_tensor(frames_rgb)  # (1, 3, T, H, W)

# 2. DOVER 模型推理（双分支）
technical_score = model.technical_head(features)  # [0, 1]
aesthetic_score = model.aesthetic_head(features)  # [0, 1]

# 3. 取平均
final_score = (technical_score + aesthetic_score) / 2

# 4. 对所有 chunk 重复，最后取平均
dover_score = np.mean([chunk1_score, chunk2_score, ...])
```

**示例**：
```
Chunk 1（0-5秒）：
  - Technical: 0.72
  - Aesthetic: 0.68
  - Average: 0.70

Chunk 2（5-10秒）：
  - Technical: 0.65
  - Aesthetic: 0.63
  - Average: 0.64

最终 DOVER 分数 = (0.70 + 0.64) / 2 = 0.67
```

---

## 🎯 筛选决策树

```
视频样本
    ↓
1. 计算 UniMatch 光流
    ↓
    光流 < 3？ ──YES─→ ❌ 拒绝（静态/boring）
    ↓ NO
    光流 > 100？ ──YES─→ ❌ 拒绝（过度运动）
    ↓ NO
2. 计算 DOVER 质量
    ↓
    DOVER < 阈值？ ──YES─→ ❌ 拒绝（低质量）
    ↓ NO
3. 检查其他指标
    ↓
    色彩饱和度 OK？ ──NO─→ ❌ 拒绝
    ↓ YES
    场景切换 OK？ ──NO─→ ❌ 拒绝
    ↓ YES
4. VLM 检查（如果需要）
    ↓
    实体数量 OK？ ──NO─→ ❌ 拒绝
    ↓ YES
    VLM 质量 OK？ ──NO─→ ❌ 拒绝
    ↓ YES
    ✅ 接受
```

---

## 📊 统计数据（基于代码）

### 采样密度

```
UniMatch：
- 采样间隔：0.5 秒
- 采样窗口：60 秒
- 16 fps 视频：120 对帧
- 计算量：120 次光流计算

DOVER：
- chunk 大小：5 秒（80 帧）
- 非重叠分块
- 10 秒视频：2 个 chunk
- 60 秒视频：12 个 chunk
- 计算量：N_chunks 次质量评估
```

### 性能开销

基于之前的 profile 数据：

```
单样本（1068 帧 = 66.75 秒 @ 16fps）：

1. UniMatch 光流：
   - 采样：120 对帧（前 60 秒）
   - 耗时：~6.8 秒
   - 占比：~40%

2. DOVER 质量：
   - 分块：13 个 chunk（66.75 秒）
   - 耗时：~6.0 秒（FP16 GPU + 480p）
   - 占比：~35%

3. 其他（饱和度、VLM）：
   - 耗时：~4.2 秒
   - 占比：~25%

总计：~17 秒（优化后）
```

---

## 🔬 论文依据（基于代码注释）

根据 `visual_metrics.py` 的注释：

```python
# Paper-fixed constants (arXiv:2605.15178v1, App. B.3)
UNIMATCH_SAMPLE_EVERY_S: float = 0.5
UNIMATCH_WINDOW_S: int = 60
DOVER_CHUNK_S: int = 5
```

**关键点**：
1. 这些是**论文固定的常量**，不应随意修改
2. 来源：arXiv:2605.15178v1，附录 B.3
3. 修改这些参数需要重新阅读论文并理解其设计原理

---

## 💡 设计原理推测

### 为什么 UniMatch 采样间隔是 0.5 秒？

**推测**：
- 太密集（如每帧）：计算量大，相邻帧光流变化小
- 太稀疏（如 2 秒）：可能错过重要的运动模式
- 0.5 秒（8 帧 @ 16fps）：平衡计算量和运动捕捉

### 为什么 DOVER chunk 是 5 秒？

**推测**：
- 太短（如 1 秒）：难以评估视频连续性和整体质量
- 太长（如 30 秒）：显存占用大，难以捕捉局部质量变化
- 5 秒（80 帧）：足够评估质量，显存可控

### 为什么 UniMatch 只看前 60 秒？

**推测**：
- 训练视频的有效内容可能集中在前 60 秒
- 减少计算量（对于长视频）
- 早期运动模式足以代表整个视频

---

## ✅ 总结

### UniMatch（光流）

| 属性 | 值 |
|------|-----|
| **输入** | 两帧 RGB 图像 (H, W, 3) |
| **输出** | 平均光流幅度（像素/帧） |
| **采样** | 每 0.5 秒，前 60 秒 |
| **作用** | 过滤静态/boring 场景和过度运动 |
| **阈值** | 通常 [3, 80] 或 [3, 100] |

### DOVER（质量）

| 属性 | 值 |
|------|-----|
| **输入** | 视频 chunk (T, H, W, 3), T=80 |
| **输出** | 质量分数 [0, 1] |
| **分块** | 5 秒非重叠 chunk |
| **维度** | Technical + Aesthetic |
| **作用** | 过滤低质量视频 |
| **阈值** | 通常 [0.35, 1.0] 或 [0.65, 1.0] |

### Table 6 筛选

- **规则**：所有适用过滤器必须全部通过
- **灵活性**：不同数据源（source）有不同阈值
- **严格性**：一项不通过即拒绝（AND 逻辑）

---

**生成时间**：2026-08-09  
**基于**：代码实现 + 注释 + Table 6 配置  
**状态**：✅ 完整（待补充论文细节）
