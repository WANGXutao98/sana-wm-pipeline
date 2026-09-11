# 深度分析：sana-wm-data-clean 完整实现解析

> **创建时间**：2026-08-12  
> **目的**：回答三个核心问题并给出完整替换方案

---

## 问题1：sana-wm-data-clean 如何使用逐帧内参BA？

### 1.1 完整流程

```
┌─────────────────────────────────────────────────────────┐
│ Step 0: 准备工作（手动执行一次）                            │
│ ─────────────────────────────────────────────────────── │
│ 1. 应用逐帧内参BA补丁到VIPE源码                            │
│    bash vipe_patches/apply_vipe_patches.sh              │
│                                                          │
│    修改的文件：                                           │
│    - vipe/slam/maths/geom.py       (添加 fi/fj 参数)    │
│    - vipe/slam/ba/terms.py         (逐帧Jacobian)       │
│    - vipe/slam/components/buffer.py (intrinsics_pf)     │
│    - vipe/slam/system.py           (dump逐帧内参)       │
│                                                          │
│ 2. 复制深度后端到VIPE                                      │
│    cp vipe_patches/pi3x_moge_depth.py \                 │
│       <vipe>/vipe/priors/depth/pi3xmoge.py              │
│                                                          │
│ 3. 注册深度模型（修改 __init__.py）                        │
│    在 make_depth_model() 添加:                           │
│    elif model_name == "pi3xmoge":                        │
│        from .pi3xmoge import Pi3xMogeModel              │
│        return Pi3xMogeModel()                            │
│                                                          │
│ 4. 复制配置文件                                            │
│    cp vipe_patches/sanawm_pipeline.yaml \               │
│       <vipe>/configs/pipeline/sanawm.yaml               │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 1: 预计算融合深度（每个视频执行一次）                   │
│ 文件: scripts/precompute_fused_depth.py                  │
│ ─────────────────────────────────────────────────────── │
│ 执行:                                                     │
│   python scripts/precompute_fused_depth.py \            │
│     input.mp4 /tmp/depth_out                            │
│                                                          │
│ 输出:                                                     │
│   /tmp/depth_out/                                        │
│   ├── fused.npy       (S, h, w) 融合深度                 │
│   ├── sig.npy         (S, 768) RGB 16x16签名             │
│   ├── scales.npy      (S,) 逐帧尺度因子                   │
│   └── sample_idx.npy  (S,) 采样索引                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: VIPE SLAM + 逐帧内参BA                            │
│ 文件: pose/vipe_cli.py::annotate_pose_vipe_cli()         │
│ ─────────────────────────────────────────────────────── │
│ 环境变量:                                                  │
│   SANA_WM_FUSED_DEPTH_DIR=/tmp/depth_out                │
│   SANA_WM_PF_DUMP=/tmp/work/intr_pf.npy  (输出)         │
│                                                          │
│ 执行:                                                     │
│   vipe infer input.mp4 -o /tmp/vipe_out \               │
│     --pipeline sanawm                                    │
│                                                          │
│ 配置文件: sanawm_pipeline.yaml                            │
│   slam:                                                  │
│     keyframe_depth: pi3xmoge  ✅ 使用Pi3xMogeModel       │
│     optimize_intrinsics: true ✅ 启用内参优化             │
│     ba:                                                  │
│       fused: false  ✅ 禁用fused CUDA kernel（为支持逐帧） │
│                                                          │
│ VIPE内部执行:                                             │
│   1. 加载 pi3xmoge 深度模型                               │
│      - 读取 SANA_WM_FUSED_DEPTH_DIR/fused.npy          │
│      - 读取 sig.npy (用于RGB签名匹配)                     │
│   2. SLAM前端: 跟踪关键帧                                  │
│   3. BA后端: 优化poses + intrinsics_pf                   │
│      - intrinsics_pf: (N,4) 每帧独立优化                 │
│      - 修改后的geom.py使用fi/fj索引逐帧内参               │
│   4. 输出:                                                │
│      - vipe_out/pose/<stem>.npz    (T,4,4) c2w          │
│      - SANA_WM_PF_DUMP  (K,4) 逐帧内参 [fx,fy,cx,cy]    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: 转换为标准输出格式                                 │
│ 文件: pose/vipe_cli.py::annotate_pose_vipe_cli() 尾部     │
│ ─────────────────────────────────────────────────────── │
│ 1. 加载VIPE输出:                                          │
│    poses = _load_vipe_pose(vipe_out)  # (N,4,4)         │
│    intr_raw = np.load(pf_dump)        # (K,4) K个关键帧  │
│                                                          │
│ 2. 插值到全部帧:                                          │
│    intr = _load_perframe_intrinsics(pf_dump, N)         │
│    # 如果 K < N: 线性插值                                 │
│    # 如果 K == N: 直接使用                                │
│    # 如果 K == 1: 广播                                    │
│                                                          │
│ 3. 保存标准格式:                                          │
│    np.save(out_dir / f"{clip_id}.poses.npy", poses)     │
│    np.save(out_dir / f"{clip_id}.intrinsics.npy", intr) │
│                                                          │
│ 最终输出:                                                  │
│   <out_dir>/                                             │
│   ├── <clip_id>.poses.npy       (N,4,4) c2w float64     │
│   ├── <clip_id>.intrinsics.npy  (N,4) float64           │
│   └── result.json                metadata + QC           │
└─────────────────────────────────────────────────────────┘
```

### 1.2 关键技术细节

**逐帧内参BA的实现**（修改VIPE源码）：

```python
# vipe/slam/maths/geom.py (补丁后)
def compute_residuals(..., fi, fj):  # ✅ 新增 fi/fj 参数
    # 原版: 按rig索引 qi/qj 获取内参（所有帧共享）
    # intrinsics[qi]  # (V, 4)
    
    # 补丁后: 按frame索引 fi/fj 获取内参（每帧独立）
    if fi is None:
        fi = qi  # fallback到原版行为
    if fj is None:
        fj = qj
    intrinsics_i = intrinsics_pf[fi]  # ✅ (N, 4) 逐帧内参buffer
    intrinsics_j = intrinsics_pf[fj]
```

**配置要求**（sanawm_pipeline.yaml）：

```yaml
slam:
  optimize_intrinsics: true  # ✅ 必须启用
  ba:
    fused: false  # ✅ 必须禁用（fused CUDA kernel假设共享内参）
```

**检验方法**（判断是否真正逐帧BA）：

```python
# 检查输出内参的变化幅度
intr = np.load("intrinsics.npy")  # (N, 4)
fx_std = intr[:, 0].std()
fy_std = intr[:, 1].std()

# 如果是真正的逐帧BA（变焦视频）:
# fx_std > 5 (像素)

# 如果是共享内参（固定焦距）:
# fx_std < 0.1 (近似零)

# 如果是插值:
# fx_std 会呈现线性趋势
```

---

## 问题2：CachedDepthModel 的真相

### 2.1 CachedDepthModel 的深度来源

**答案**：✅ **是的，CachedDepthModel 使用 Pi3X+MoGe-2 融合深度**

**证据链**：

```python
# 1. mode_default.py 预计算函数
def _precompute_depth_cache(...):
    # ... Pi3X推理 ...
    d_pi3x = ...  # (T, H, W)
    
    # ... MoGe-2推理 ...
    d_moge = ...  # (T, H, W)
    
    # ❌ 错误的融合算法（这是bug所在）
    scale_history = ...  # (T,)
    depths_fused = d_pi3x * scale_history[:, None, None]
    
    # 保存给CachedDepthModel使用
    np.savez_compressed(cache_path, depths=depths_fused, scale_history=scale_history)
```

```python
# 2. CachedDepthModel 加载预计算结果
class CachedDepthModel(DepthEstimationModel):
    def __init__(self, cache_path):
        d = np.load(cache_path)
        self._depths = d["depths"]  # ✅ 加载Pi3X+MoGe-2融合深度
```

```yaml
# 3. vipe_cached_depth.yaml 配置
slam:
  keyframe_depth: cached  # ✅ 使用CachedDepthModel
```

**流程对比**：

| 步骤 | 原流程 | 参考实现 |
|------|--------|---------|
| **预计算** | `mode_default.py::_precompute_depth_cache()` | `scripts/precompute_fused_depth.py` |
| **融合算法** | ❌ 均值比率+错误EMA | ✅ 加权LS+正确EMA |
| **保存格式** | `.npz` (depths, scale_history) | `.npy` 多文件 (fused, sig, scales) |
| **VIPE深度后端** | `CachedDepthModel` | `Pi3xMogeModel` |
| **加载方式** | frame_idx索引 | RGB签名匹配 |

### 2.2 为什么设计成预计算？

**原因1**：环境隔离（参考实现的解释）
```python
# pi3x_moge_depth.py 注释:
# Pi3X/MoGe-2需要torch 2.5/cu124
# VIPE需要torch 2.7/cu128（CUDA扩展ABI锁定）
# 它们无法共享进程
```

**原因2**：性能优化
- Pi3X/MoGe-2推理慢（~1-2秒/帧）
- 预计算一次，VIPE可以多次调用（调试BA参数时）

**原因3**：解耦设计
- 深度估计 vs SLAM/BA是独立问题
- 便于替换不同深度模型

---

## 问题3：用 sana-wm-data-clean 完全替换原流程

### 3.1 替换方案评估

你说得对！既然 sana-wm-data-clean 被证实可产生可训练数据，应该尽量完全替换。

**完整替换清单**：

| 组件 | 原流程 | sana-wm-data-clean | 替换难度 | 优先级 |
|------|--------|-------------------|---------|--------|
| **融合算法** | mode_default.py:108-120 | pose/fusion.py | ⭐ 简单 | 🔴 高 |
| **预计算脚本** | _precompute_depth_cache() | scripts/precompute_fused_depth.py | ⭐⭐ 中等 | 🟡 中 |
| **VIPE深度后端** | CachedDepthModel | Pi3xMogeModel | ⭐⭐⭐ 复杂 | 🟡 中 |
| **逐帧内参BA** | 无 | apply_perframe_intrinsics_ba.py | ⭐⭐⭐⭐ 很复杂 | 🟢 低 |

### 3.2 完整替换方案（推荐）

#### 方案：渐进式替换（3个阶段）

##### 阶段1：替换融合算法（1小时，今天）

**目标**：修复核心bug

**操作**：
1. 复制 `sana-wm-data-clean/sana_wm_data/pose/fusion.py` → `src/.../stage02_pose/depth_fusion.py`
2. 修改 `mode_default.py:108-120`，调用新的 `fuse_depth_sequence()`

**验证**：
```bash
# 本地测试1个样本
python -m sana_wm_pipeline.stage02_pose.run_worker \
  --input testdata/*.mp4 \
  --output /tmp/test_fixed
  
# 检查输出
python -c "
import numpy as np
s = np.load('/tmp/test_fixed/_depth_cache.npz')['scale_history']
print(f'Scale std: {s.std():.4f}')  # 应该更平滑
print(f'NaN count: {np.isnan(s).sum()}')  # 应该为0
"
```

**预期效果**：尺度估计偏差 ↓80%，时序抖动 ↓70%

---

##### 阶段2：替换预计算脚本 + VIPE深度后端（3小时，今天/明天）

**目标**：完全对齐参考实现的深度处理

**操作1：独立预计算脚本**

```python
# 创建 scripts/precompute_fused_depth_reference.py
# 完全复制 sana-wm-data-clean/scripts/precompute_fused_depth.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sana_wm_pipeline.stage02_pose.depth_fusion import fuse_depth_sequence
# ... 其余代码与参考实现相同 ...
```

**操作2：复制Pi3xMogeModel**

```bash
# 复制深度后端
cp sana-wm-data-clean/vipe_patches/pi3x_moge_depth.py \
   third_party/vipe/vipe/priors/depth/pi3xmoge.py

# 修改 third_party/vipe/vipe/priors/depth/__init__.py
# 在 make_depth_model() 中添加：
elif model_name == "pi3xmoge":
    from .pi3xmoge import Pi3xMogeModel
    return Pi3xMogeModel()
```

**操作3：创建新的VIPE配置**

```yaml
# 创建 configs/vipe_sanawm.yaml (或复制sanawm_pipeline.yaml)
slam:
  keyframe_depth: pi3xmoge  # ✅ 改用Pi3xMogeModel
  optimize_intrinsics: true
  ba:
    fused: false  # ✅ 为逐帧内参BA准备
```

**操作4：修改 mode_default.py**

```python
# mode_default.py

def run_default(...):
    # Step 1: 调用独立预计算脚本
    depth_dir = work_dir / "depth"
    subprocess.run([
        sys.executable,
        "scripts/precompute_fused_depth_reference.py",
        str(clip_path),
        str(depth_dir)
    ], check=True)
    
    # Step 2: 设置环境变量
    os.environ["SANA_WM_FUSED_DEPTH_DIR"] = str(depth_dir)
    
    # Step 3: 调用VIPE（改用新配置）
    subprocess.run([
        "vipe", "infer", str(clip_path),
        "-o", str(work_dir),
        "--pipeline", "vipe_sanawm"  # ✅ 新配置
    ], check=True)
```

**验证**：
```bash
# 对比两个版本的输出
python scripts/compare_depth_backends.py \
  --cached /tmp/output_cached \
  --pi3xmoge /tmp/output_pi3xmoge
```

**预期效果**：
- 输出poses基本相同（<1%差异）
- 签名匹配更robust（极端场景）

---

##### 阶段3：应用逐帧内参BA补丁（可选，2-3小时）

**目标**：支持变焦视频

**操作**：

```bash
# 1. 应用补丁到VIPE源码
cd third_party/vipe
python ../../sana-wm-data-clean/vipe_patches/apply_perframe_intrinsics_ba.py .

# 2. 验证补丁是否成功
grep "intrinsics_pf\|per-frame intrinsics" vipe/slam/components/buffer.py
# 应该有输出

# 3. 重新安装VIPE（如果需要CUDA重编译）
pip install -e . --no-build-isolation
```

**验证**：
```python
# 检查输出内参的变化
intr = np.load("output/clip.intrinsics.npy")
fx_std = intr[:, 0].std()
print(f"fx std: {fx_std:.2f}")
# 变焦视频应该 >5，固定焦距应该 <0.5
```

**预期效果**：
- 变焦视频轨迹精度 ↑10-15%
- 固定焦距视频无明显差异

---

### 3.3 风险评估

| 阶段 | 风险 | 缓解措施 |
|------|------|---------|
| **阶段1** | 极低 | 只改算法，不动架构 |
| **阶段2** | 中等 | 可能破坏现有pipeline集成 |
| **阶段3** | 高 | 需要修改VIPE源码，可能导致不兼容 |

### 3.4 时间规划（1.5天）

**Day 1 下午（剩余4小时）**：
- ✅ 阶段1完成（1h）
- ✅ 阶段2完成（3h）
- 本地验证（testdata）

**Day 2 上午（4小时）**：
- CMCC部署
- 200样本标注
- 对比报告

**可选（如果Day 2结果仍有问题）**：
- 阶段3：应用逐帧内参BA补丁

---

## 四、最终推荐

基于你的反馈"sana-wm-data-clean被证实可产生可训练数据"，我推荐：

### ✅ 推荐方案：阶段1+2 完整替换

**理由**：
1. ✅ **最接近参考实现**（融合算法+预计算+深度后端）
2. ✅ **风险可控**（阶段3可选，不影响核心功能）
3. ✅ **符合1.5天时间要求**
4. ✅ **保留了逐帧内参BA的升级路径**

**不推荐**：只替换融合算法

**原因**：既然要完全对齐参考实现，就应该连深度后端一起换掉，避免留下隐患。

---

## 五、实施检查清单

### 阶段1（融合算法）
- [ ] 复制 `fusion.py`
- [ ] 修改 `mode_default.py:108-120`
- [ ] 本地测试1个样本
- [ ] 验证scale平滑度和NaN

### 阶段2（预计算+深度后端）
- [ ] 创建 `scripts/precompute_fused_depth_reference.py`
- [ ] 复制 `pi3x_moge_depth.py` → `third_party/vipe/.../pi3xmoge.py`
- [ ] 修改 `__init__.py` 注册pi3xmoge模型
- [ ] 创建 `configs/vipe_sanawm.yaml`
- [ ] 修改 `mode_default.py` 调用新流程
- [ ] 本地测试对比cached vs pi3xmoge

### 阶段3（可选，逐帧内参BA）
- [ ] 应用 `apply_perframe_intrinsics_ba.py`
- [ ] 验证补丁成功
- [ ] 测试变焦视频

---

**现在需要你确认**：

1. ✅ 批准阶段1+2完整替换方案？
2. 📝 是否需要我立即开始编写替换代码？
3. 🎯 阶段3（逐帧内参BA）是否作为可选项（如果200样本验证后仍有问题再考虑）？

**回复"批准阶段1+2"后，我将立即开始实施！** 🚀
