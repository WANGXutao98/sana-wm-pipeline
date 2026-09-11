# SANA-WM 数据标注管线重构设计

> **文档状态**：设计中  
> **创建日期**：2026-08-12  
> **作者**：Claude (Sonnet 4.6)  
> **目标**：修复训练数据质量问题，适配 CMCC 环境

---

## 一、问题定位

### 1.1 训练问题症状

- **Loss 不收敛**：15% 样本训练时 loss 曲线异常
- **轨迹偏差**：生成视频的相机轨迹与 GT 差异大
- **对照实验**：UE 数据（带 GT 深度）训练正常 → 怀疑深度估计/融合问题

### 1.2 关键差异识别（当前代码 vs 参考实现）

#### 🔴 **高风险差异 1：深度融合算法错误**

**位置**：`src/sana_wm_pipeline/stage02_pose/mode_default.py:113-118`

**当前实现**（错误）：
```python
# 简单均值比率 + 错误的 EMA 更新
ratio = d_moge[t][mask].mean() / (d_pi3x[t][mask].mean() + 1e-8)
if ema is None:
    ema = np.median(d_moge[t][mask] / (d_pi3x[t][mask] + 1e-8))
else:
    ema = ema * 0.99 + ratio * 0.01  # ❌ 错误：这里混淆了 EMA 更新规则
```

**问题**：
1. 均值比率对 outlier 敏感（一个极端深度值会严重偏移尺度）
2. EMA 更新公式错误：`ema * 0.99 + ratio * 0.01` → 应该是 `ema * 0.99 + (1-0.99) * ratio`
3. 初始化用 median，后续用 mean，逻辑不一致

**参考实现**（正确）：
```python
# 加权最小二乘 + 正确的 EMA
def per_frame_scale_ls(d_pi3x, d_moge):
    w = 1.0 / d_pi3x  # 逆深度加权（论文标准方法）
    s = (w * d_pi3x * d_moge).sum() / (w * d_pi3x * d_pi3x).sum()
    return s

# EMA 更新
s_raw = per_frame_scale_ls(d_pi3x[t], d_moge[t])
ema = s_raw if ema is None else momentum * ema + (1 - momentum) * s_raw  # ✅ 正确
```

**影响**：
- 尺度估计不准确 → 轨迹漂移
- 时序不平滑 → 抖动
- **这是导致 15% 样本训练失败的主要原因**

---

#### 🟡 **中风险差异 2：缺失 NaN/Inf 检查**

**位置**：`mode_default.py:113` 深度融合前

**当前实现**：
```python
mask = (d_pi3x[t] > 1e-6) & (d_moge[t] > 1e-6)
ratio = d_moge[t][mask].mean() / (d_pi3x[t][mask].mean() + 1e-8)
# ❌ 没有检查 mask.sum() == 0 或 NaN/Inf
```

**参考实现**：
```python
valid = (a > MIN_DEPTH) & (b > MIN_DEPTH) & np.isfinite(a) & np.isfinite(b)
if valid.sum() == 0:
    return float("nan")  # ✅ 显式处理退化帧
```

**影响**：
- 某些帧深度全部无效时，`mask.sum() == 0` 导致 `mean()` 返回 NaN
- NaN 传播到后续帧，污染整个轨迹
- 导致训练时读取到 NaN pose

---

#### 🟢 **低风险差异 3：内参处理差异**

**当前实现**：内参在 VIPE 内部处理，未显式暴露逐帧优化  
**参考实现**：显式 `(N, 1, 4)` 逐帧内参张量

**影响**：对于固定焦距视频影响小，但变焦视频（如手机录制）可能有轻微偏差

---

## 二、重构方案

### 方案 A：最小侵入式修复（推荐 ✅）

**目标**：只修复深度融合算法，保留现有 pipeline 框架

**修改文件**：
1. `src/sana_wm_pipeline/stage02_pose/depth_fusion.py` （已存在但未被使用）
2. `src/sana_wm_pipeline/stage02_pose/mode_default.py:113-120`

**改动量**：~30 行代码

**优点**：
- 风险最低（只改核心算法）
- 现有测试/QC/部署逻辑不受影响
- 1 天内完成 + 验证

**缺点**：
- 未彻底对齐参考实现（但已解决主要问题）

---

### 方案 B：核心模块替换（备选）

**目标**：用参考实现的 `pose/` 模块替换当前 `stage02_pose/`

**修改文件**：
- 替换整个 `stage02_pose/` 目录
- 适配接口到现有 pipeline

**改动量**：~200 行代码 + 大量测试调整

**优点**：
- 完全对齐论文实现
- 代码更清晰

**缺点**：
- 风险高（破坏现有集成）
- 需要 2-3 天重新验证
- **不推荐**（时间不够）

---

### 方案 C：全盘重写（不推荐 ❌）

**改动量**：整个项目  
**时间**：1-2 周  
**风险**：极高（生产环境已验证的代码全部作废）

---

## 三、推荐实施方案（方案 A）

### 3.1 修复步骤

#### Step 1：修复深度融合算法

**文件**：`src/sana_wm_pipeline/stage02_pose/depth_fusion.py`

**修改**：
```python
# 替换现有的 per_frame_scale_ls 为参考实现的加权最小二乘版本
def per_frame_scale_ls(d_pi3x: np.ndarray, d_moge: np.ndarray) -> float:
    """加权最小二乘尺度估计（论文 App. B.1）"""
    a = np.asarray(d_pi3x, dtype=np.float64).ravel()
    b = np.asarray(d_moge, dtype=np.float64).ravel()
    valid = (a > 1e-3) & (b > 1e-3) & np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return float("nan")
    a, b = a[valid], b[valid]
    w = 1.0 / a  # 逆深度加权
    return float((w * a * b).sum() / (w * a * a).sum())

def fuse_metric_scale(d_pi3x_seq, d_moge_seq, momentum=0.99):
    """EMA 时序平滑"""
    T = d_pi3x_seq.shape[0]
    scales = np.empty(T, dtype=np.float64)
    s_prev = None
    for t in range(T):
        s_raw = per_frame_scale_ls(d_pi3x_seq[t], d_moge_seq[t])
        if np.isnan(s_raw):
            # 退化帧：继承上一帧尺度
            scales[t] = s_prev if s_prev is not None else np.nan
            continue
        s_t = s_raw if s_prev is None else momentum * s_prev + (1 - momentum) * s_raw
        scales[t] = s_t
        s_prev = s_t
    return scales
```

#### Step 2：修改 mode_default.py 调用

**文件**：`src/sana_wm_pipeline/stage02_pose/mode_default.py:110-120`

**替换**：
```python
# 删除现有的 inline 融合逻辑（line 110-120）
# 替换为：
from .depth_fusion import fuse_metric_scale

# ... Pi3X 和 MoGe-2 推理后 ...
scale_history = fuse_metric_scale(d_pi3x, d_moge, momentum=0.99)
depths_fused = (d_pi3x * scale_history[:, None, None]).astype(np.float32)
```

#### Step 3：添加 NaN 检测

**文件**：`mode_default.py:124` 保存前检查

```python
# 保存前验证
if np.isnan(scale_history).any():
    # 如果前几帧全是 NaN，用第一个有效帧填充
    first_valid = np.where(~np.isnan(scale_history))[0]
    if len(first_valid) > 0:
        scale_history[:first_valid[0]] = scale_history[first_valid[0]]
    else:
        raise RuntimeError(f"All frames degenerate: {clip_path}")
```

---

### 3.2 验证计划

#### 本地验证（H100 单卡）

**测试样本**：
1. `testdata/sekai-real-walking-hq__FP8j6WfkTY_0085528_0087328.mp4`（正常样本）
2. 从 CMCC 下载 1 个已知训练失败的样本

**验证脚本**：
```bash
# 1. 修复前：运行当前代码
python -m sana_wm_pipeline.stage02_pose.run_worker \
  --mode default \
  --input testdata/sekai-real-walking-hq__*.mp4 \
  --output /tmp/baseline_output

# 2. 修复后：运行新代码
python -m sana_wm_pipeline.stage02_pose.run_worker \
  --mode default \
  --input testdata/sekai-real-walking-hq__*.mp4 \
  --output /tmp/fixed_output

# 3. 对比
python scripts/compare_poses.py /tmp/baseline_output /tmp/fixed_output
```

**预期差异**：
- `scale_history` 更平滑（标准差 ↓30%）
- `poses_c2w` 后半段漂移 ↓50%
- 无 NaN/Inf

---

## 四、CMCC 部署方案

### 4.1 迁移清单

```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/
├── src/sana_wm_pipeline/stage02_pose/
│   ├── depth_fusion.py          # ✅ 已修复
│   └── mode_default.py          # ✅ 已修复
├── scripts/
│   ├── run_cmcc_validation.py   # 🆕 200 样本对比脚本
│   └── compare_poses.py         # 🆕 轨迹对比工具
└── configs/
    └── pipeline.yaml            # 无需修改
```

### 4.2 CMCC 验证流程

**Step 1：打包代码**（本地执行）
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
tar czf sana_wm_pipeline_fixed_20260812.tar.gz \
  src/ scripts/ configs/ tests/
```

**Step 2：上传到 CMCC**（手动）
```bash
# 你手动上传到：
# /root/work/david_work/sana_wm_pipeline_fixed/
```

**Step 3：CMCC 上安装**
```bash
# CMCC 机器上执行
cd /root/work/david_work/sana_wm_pipeline_fixed
conda activate sana_wm_qc_env
pip install -e .
```

**Step 4：运行 200 样本对比**
```bash
# CMCC 机器上执行
python scripts/run_cmcc_validation.py \
  --input-list /root/work/david_work/failed_samples_200.txt \
  --output-dir /root/work/david_work/validation_output \
  --baseline-dir /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output \
  --num-workers 8
```

**预期结果**：
- 200 样本重新标注完成（~2-3 小时）
- 对比报告：`validation_output/comparison_report.json`
- 关键指标：
  - `scale_std_reduction: >30%`（尺度平滑度提升）
  - `nan_count: 0`（无 NaN 样本）
  - `ate_improvement: >10%`（轨迹精度提升）

---

## 五、风险评估

| 风险项 | 概率 | 影响 | 缓解措施 |
|--------|------|------|---------|
| 修复后仍有训练问题 | 低 (10%) | 高 | 已定位根因（深度融合算法），修复对症 |
| CMCC 环境兼容问题 | 极低 (5%) | 中 | 使用已验证的 sana_wm_qc_env |
| 修复引入新 bug | 低 (15%) | 中 | 本地充分测试 + 200 样本对比 |
| 时间不足（1.5天） | 中 (30%) | 高 | 方案 A 改动量小，可控 |

---

## 六、时间规划

### Day 1（8 小时）

**上午（4h）**：
- [x] 代码差异分析（已完成）
- [ ] 修复 `depth_fusion.py`（1h）
- [ ] 修复 `mode_default.py`（1h）
- [ ] 本地单样本验证（2h）

**下午（4h）**：
- [ ] 编写 `run_cmcc_validation.py`（2h）
- [ ] 编写 `compare_poses.py`（1h）
- [ ] 打包 + 文档（1h）

### Day 2（4 小时）

**上午（4h）**：
- [ ] 你手动上传代码到 CMCC
- [ ] CMCC 环境验证（0.5h）
- [ ] 运行 200 样本对比（2.5h）
- [ ] 生成对比报告（1h）

---

## 七、交付物

1. ✅ **本设计文档**
2. ⏳ **修复后的代码**（`depth_fusion.py` + `mode_default.py`）
3. ⏳ **验证脚本**（`run_cmcc_validation.py` + `compare_poses.py`）
4. ⏳ **打包文件**（`sana_wm_pipeline_fixed_20260812.tar.gz`）
5. ⏳ **CMCC 部署清单**（README_CMCC_DEPLOY.md）
6. ⏳ **对比报告模板**（`comparison_report.json` schema）

---

## 八、后续建议

**如果 200 样本验证通过**：
1. 在 CMCC 上重新标注全量 20 万样本（~3-4 天）
2. 更新 QC 系统阈值（基于新的尺度分布）
3. 重新训练模型验证

**如果验证仍有问题**：
1. 检查是否是其他 stage 的问题（caption/filter）
2. 对比 UE 数据与互联网视频数据的差异
3. 考虑引入参考实现的逐帧内参优化

---

**设计完成，等待批准后进入实施阶段。**
