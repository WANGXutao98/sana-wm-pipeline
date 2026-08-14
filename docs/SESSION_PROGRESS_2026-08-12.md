# SANA-WM 重构进度保存（2026-08-12）

> **会话状态**：准备开始阶段1+2+3实施  
> **暂停原因**：机器即将重启  
> **恢复命令**：见本文档末尾

---

## 一、当前状态

### 已完成工作

1. ✅ **深度代码分析**（3小时）
   - 对比了三个目录的实现差异
   - 定位了训练失败的根本原因（融合算法bug）
   - 生成了详细分析文档

2. ✅ **方案设计**（1小时）
   - 确定了阶段1+2+3的完整替换方案
   - 与用户确认了执行策略

3. ✅ **关键文档已生成**
   - `docs/COMPLETE_REPLACEMENT_PLAN.md` - 完整替换方案
   - `docs/FUSION_REPLACEMENT_ANALYSIS.md` - 融合算法分析
   - `docs/INPUT_OUTPUT_COMPARISON.md` - 输入输出对比
   - `docs/FINAL_DECISION.md` - 最终决策文档

### 待执行工作

**阶段0**：创建 git 开发分支
- 分支名：`refactor/sana-wm-align-reference-impl`

**阶段1**：替换融合算法（预计1小时）
- 复制 `sana-wm-data-clean/sana_wm_data/pose/fusion.py`
- 修改 `src/sana_wm_pipeline/stage02_pose/mode_default.py`
- 本地验证

**阶段2**：替换预计算+深度后端（预计3小时）
- 创建独立预计算脚本
- 复制 Pi3xMogeModel 到VIPE
- 修改 mode_default.py 调用逻辑
- 创建新VIPE配置

**阶段3**：应用逐帧内参BA补丁（预计2-3小时）
- 应用 apply_perframe_intrinsics_ba.py
- 验证补丁成功

---

## 二、关键发现总结

### 2.1 训练失败根因

**位置**：`src/sana_wm_pipeline/stage02_pose/mode_default.py:108-120`

**3个致命错误**：
1. ❌ 均值比率 vs 加权最小二乘
2. ❌ EMA公式错误：`0.99*ema + 0.01*ratio` 应该是 `0.99*ema + (1-0.99)*ratio`
3. ❌ 缺失NaN检查

### 2.2 架构差异

| 组件 | 原流程 | sana-wm-data-clean | 一致性 |
|------|--------|-------------------|--------|
| 融合算法 | 错误实现 | 正确实现 | ❌ 0% |
| 预计算方式 | inline函数 | 独立脚本 | ⚠️ 50% |
| VIPE深度后端 | CachedDepthModel | Pi3xMogeModel | ⚠️ 70% |
| 逐帧内参BA | 未应用 | 需要补丁 | ❌ 0% |

---

## 三、实施计划详情

### 阶段1：融合算法替换

**源文件**：
```
sana-wm-data-clean/sana_wm_data/pose/fusion.py
```

**目标位置**：
```
src/sana_wm_pipeline/stage02_pose/depth_fusion.py (新建)
```

**修改文件**：
```
src/sana_wm_pipeline/stage02_pose/mode_default.py
- 删除：第108-120行（13行）
- 替换为：4行调用新的 fuse_depth_sequence()
```

**关键代码**（来自参考实现）：
```python
def solve_frame_scale(d_pi3x, d_moge):
    a = np.asarray(d_pi3x, dtype=np.float64).ravel()
    b = np.asarray(d_moge, dtype=np.float64).ravel()
    valid = (a > 1e-3) & (b > 1e-3) & np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return 1.0
    a, b = a[valid], b[valid]
    w = 1.0 / (b + 1e-8)  # 逆深度加权
    return float((w * a * b).sum() / (w * a * a).sum())
```

---

### 阶段2：预计算+深度后端替换

**操作1：独立预计算脚本**

源：`sana-wm-data-clean/scripts/precompute_fused_depth.py`  
目标：`scripts/precompute_fused_depth_reference.py`

**操作2：Pi3xMogeModel**

源：`sana-wm-data-clean/vipe_patches/pi3x_moge_depth.py`  
目标：`third_party/vipe/vipe/priors/depth/pi3xmoge.py`

注册位置：`third_party/vipe/vipe/priors/depth/__init__.py`

**操作3：VIPE配置**

源：`sana-wm-data-clean/vipe_patches/sanawm_pipeline.yaml`  
目标：`configs/vipe_sanawm.yaml`

**操作4：修改调用逻辑**

文件：`src/sana_wm_pipeline/stage02_pose/mode_default.py`
- 函数：`run_default()`
- 修改：从inline预计算改为subprocess调用独立脚本

---

### 阶段3：逐帧内参BA补丁

**源补丁**：
```
sana-wm-data-clean/vipe_patches/apply_perframe_intrinsics_ba.py
```

**执行**：
```bash
cd third_party/vipe
python ../../sana-wm-data-clean/vipe_patches/apply_perframe_intrinsics_ba.py .
```

**验证**：
```bash
grep "intrinsics_pf" vipe/slam/components/buffer.py
grep "per-frame intrinsics" vipe/slam/maths/geom.py
```

---

## 四、验证计划

### 本地验证（阶段1+2完成后）

```bash
# 测试样本
export TEST_VIDEO=/mnt/afs/davidwang/workspace/sana_wm_pipeline/testdata/sekai-real-walking-hq__FP8j6WfkTY_0085528_0087328.mp4

# 运行新代码
python -m sana_wm_pipeline.stage02_pose.run_worker \
  --mode default \
  --input $TEST_VIDEO \
  --output /tmp/test_new

# 检查输出
python -c "
import numpy as np
# 检查scale平滑度
s = np.load('/tmp/test_new/scale.npy')
print(f'Scale std: {s.std():.4f}')
print(f'Scale range: [{s.min():.3f}, {s.max():.3f}]')
print(f'NaN count: {np.isnan(s).sum()}')

# 检查poses
p = np.load('/tmp/test_new/poses_c2w.npy')
print(f'Poses shape: {p.shape}')
print(f'Poses NaN: {np.isnan(p).sum()}')
"
```

### CMCC验证（明天）

```bash
# 打包代码
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
tar czf sana_wm_pipeline_refactored_20260812.tar.gz \
  src/ scripts/ configs/ third_party/vipe/ tests/

# 上传到CMCC后
conda activate sana_wm_qc_env
pip install -e .

# 运行200样本
python scripts/batch_annotate_200.py \
  --input-list failed_samples_200.txt \
  --output-dir /tmp/refactored_output \
  --num-workers 8
```

---

## 五、重启后恢复命令

### 机器重启后

```bash
# 1. 进入工作目录
cd /mnt/afs/davidwang/workspace

# 2. 启动Claude
bash workspace/start_claude.sh

# 3. 在Claude中执行
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 4. 继续执行任务
# 提示词：
# "机器已重启，请继续执行 SANA-WM 重构任务。
#  参考进度文档：docs/SESSION_PROGRESS_2026-08-12.md
#  从阶段0（创建git分支）开始执行阶段1+2+3"
```

---

## 六、重要提醒

### 执行原则

1. ✅ **严格复制参考实现代码**（不允许自己编写）
2. ✅ **保持与 sana-wm-data-clean 100%一致**
3. ✅ **每个阶段完成后验证**
4. ✅ **遇到问题立即停止，不要猜测**

### 关键文件路径

**参考实现目录**：
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-wm-data-clean/
├── sana_wm_data/pose/
│   ├── fusion.py          ← 阶段1需要
│   └── ...
├── scripts/
│   └── precompute_fused_depth.py  ← 阶段2需要
└── vipe_patches/
    ├── pi3x_moge_depth.py    ← 阶段2需要
    ├── sanawm_pipeline.yaml  ← 阶段2需要
    └── apply_perframe_intrinsics_ba.py  ← 阶段3需要
```

**当前项目目录**：
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/
├── src/sana_wm_pipeline/stage02_pose/
│   ├── mode_default.py    ← 需要修改
│   └── depth_fusion.py    ← 新建
├── scripts/
│   └── precompute_fused_depth_reference.py  ← 新建
├── configs/
│   └── vipe_sanawm.yaml   ← 新建
└── third_party/vipe/
    └── vipe/priors/depth/
        ├── pi3xmoge.py    ← 新建
        └── __init__.py    ← 需要修改
```

---

## 七、时间规划（重启后）

**Day 1 下午（剩余时间）**：
- 阶段0：创建分支（5分钟）
- 阶段1：融合算法（1小时）
- 阶段2：预计算+深度后端（3小时）
- 本地验证（30分钟）

**Day 2 上午**：
- 阶段3：逐帧内参BA（可选，2小时）
- CMCC部署（1小时）
- 200样本验证（3小时）

---

**会话保存完成！机器重启后使用上述恢复命令继续执行。** ✅
