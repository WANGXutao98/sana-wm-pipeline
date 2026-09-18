# Pose Stage 代码对齐与测试流程方法论

**目标读者**: 用于指导后续 Pose Stage 其他模式（gt-depth, default）的代码对齐和测试  
**参考案例**: GT-Pose 模式完整对齐过程  
**适用原则**: Ponytail 原则 - 官方代码是 Ground Truth

---

## 📋 整体流程框架

### 阶段 1: 代码对齐（Code Alignment）
### 阶段 2: 单元测试（Unit Testing）
### 阶段 3: 冒烟测试（Smoke Testing）
### 阶段 4: 结果验证（Result Validation）

---

## 🎯 阶段 1: 代码对齐

### 1.1 逐行对比官方代码

**官方参考**: `sana-wm-data-clean/sana_wm_data/pose/stage.py`

**对比维度**:
1. **算法逻辑**: 核心算法是否一致
2. **数据流**: 输入 → 处理 → 输出
3. **参数配置**: 默认值、环境变量
4. **接口签名**: 函数参数、返回值
5. **执行顺序**: 步骤是否按相同顺序

**对比方法**:
```python
# 官方代码 (stage.py)
if mode == "gt_pose":
    pred_pos = adapters.run_pi3x_trajectory(..., n_scale, ...)
    n_scale = pred_pos.shape[0]
    gt_poses = _load_gt_poses(rec, N)
    if gt_poses is not None:
        gt_sub = gt_poses[adapters.even_indices(..., n_scale)]
        s = recover_metric_scale(pred_pos, gt_sub[:, :3, 3], ...)
        poses = gt_poses
    # ...

# 本地实现 (mode_gtpose.py)
# 逐行对照，确保逻辑一致
```

**输出文档**: 
- `{MODE}_CODE_ALIGNMENT_ANALYSIS.md`
- 包含逐行对比、差异说明、对齐度评估

---

### 1.2 关键对齐点清单

**GT-Pose 示例**:
- [x] Pi3X 调用方式: `_real.pi3_infer()` vs subprocess
- [x] 帧子采样逻辑: `even_indices(N, n_scale)`
- [x] Umeyama 算法: 80% inlier filter
- [x] 内参加载优先级: GT → seed fallback
- [x] 输出帧数: 完整 N 帧（不截断）
- [x] Scale 广播: `[s] * m` vs `np.full(m, s)`

**通用模板**:
```markdown
## 关键对齐点

| 维度 | 官方 | 本地 | 对齐度 | 备注 |
|------|------|------|--------|------|
| 模型调用 | ... | ... | ✅/❌ | ... |
| 数据加载 | ... | ... | ✅/❌ | ... |
| 算法实现 | ... | ... | ✅/❌ | ... |
| 输出格式 | ... | ... | ✅/❌ | ... |
```

---

### 1.3 修复策略

**常见问题类型**:
1. **接口不一致**: 添加兼容参数
2. **路径推断失败**: 添加显式路径参数
3. **帧数处理**: 确保输入/输出帧数正确
4. **数据类型**: 统一 dtype (float32/float64)

**GT-Pose 案例**:
```python
# 问题: GT intrinsics 无法加载（路径推断失败）
# 修复: 添加 gt_intrinsics_path 参数
def run_gtpose(
    clip_path: Path,
    gt_poses_path: Path,
    work_dir: Path,
    inlier_percentile: float = DEFAULT_INLIER_PERCENTILE,
    gt_intrinsics_path: Path | None = None,  # ← 新增
) -> PoseArtifact:
    # 优先使用显式路径
    if gt_intrinsics_path is not None:
        intr = _load_gt_intrinsics(gt_intrinsics_path, m)
    else:
        # Fallback 原有逻辑
        intr = _load_gt_intrinsics(gt_poses_path, m)
```

---

## 🧪 阶段 2: 单元测试

### 2.1 测试覆盖点

**GT-Pose 示例**:
1. ✅ 导入路径对齐
2. ✅ `even_indices()` 函数
3. ✅ `_load_gt_poses()` 各种格式
4. ✅ `_positions_to_poses()` (dry-run)
5. ✅ 接口兼容性
6. ✅ 与官方逻辑对比

**测试脚本**: `verify_{mode}_alignment.py`

```python
def test_core_function():
    # 单元测试：验证核心函数
    result = even_indices(303, 64)
    assert len(result) == 64
    assert result[0] == 0
    assert result[-1] == 302
```

---

### 2.2 数据结构验证

**DL3DV 案例**:
```python
# 验证输入数据格式
camera_data = np.load("sample.camera.npz")
assert 'c2w' in camera_data
assert 'K_px' in camera_data

c2w = camera_data['c2w']
assert c2w.shape == (N, 4, 4)
assert c2w.dtype == np.float32

K_px = camera_data['K_px']
assert K_px.shape == (N, 4)
assert K_px.dtype == np.float32
```

**输出文档**: `{DATASET}_DATA_VERIFICATION_REPORT.md`

---

## 🔥 阶段 3: 冒烟测试

### 3.1 测试脚本设计

**目录结构**:
```
experiments/data_production_smoke/
├── smoke_{mode}_{dataset}.sh       # Shell 入口
└── SMOKE_{MODE}_{DATASET}_REVIEW.md # 审阅文档

scripts/
└── smoke_batch_{mode}.py           # Python 批处理
```

---

### 3.2 Shell 脚本设计模式

**核心功能**:
1. ✅ 环境配置（路径、模型权重、环境变量）
2. ✅ 自动扫描数据目录
3. ✅ 样本验证（文件存在性、完整性）
4. ✅ 调用 Python 批处理脚本
5. ✅ 结果报告

**模板**: `experiments/data_production_smoke/smoke_{mode}_{dataset}.sh`

```bash
#!/bin/bash
set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────────
export PROJ_DIR="/path/to/sana_wm_pipeline"
export DATA_DIR="/path/to/test_data/{dataset}"
export OUT_BASE="/path/to/smoke_result_{mode}_{dataset}"

# ── 模型权重 ──────────────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="/path/to/pi3x"
export SANA_WM_MOGE2_WEIGHTS="/path/to/moge2"

# ── 性能配置 ──────────────────────────────────────────────────
export SANA_WM_MAX_FRAMES=64

# ── 环境激活 ──────────────────────────────────────────────────
source "$ENV_DIR/bin/activate"
cd "$PROJ_DIR"

# ── 自动扫描样本 ──────────────────────────────────────────────
SAMPLES_FILE="$OUT_BASE/samples.tsv"
mkdir -p "$OUT_BASE"

for video_file in "$DATA_DIR"/*.mp4; do
    [[ -f "$video_file" ]] || continue
    
    sample_id=$(basename "$video_file" .mp4)
    
    # 检查必需文件
    # GT-Pose: .mp4 + .camera.npz
    # GT-Depth: .mp4 + .camera.npz + depth maps
    # Default: .mp4
    
    if [[ -f "必需文件" ]]; then
        # 自动读取帧数
        n_frames=$(python3 -c "读取元数据")
        echo "$sample_id	$n_frames" >> samples.tsv
    fi
done

# ── 验证样本文件 ──────────────────────────────────────────────
# 检查完整性、显示统计信息

# ── 执行批处理 ────────────────────────────────────────────────
python "$PROJ_DIR/scripts/smoke_batch_{mode}.py" \
    --samples "$SAMPLES_FILE" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_BASE"
```

---

### 3.3 Python 批处理脚本设计模式

**核心功能**:
1. ✅ 单进程批量处理（利用 @lru_cache）
2. ✅ 从数据目录加载输入
3. ✅ 调用核心处理函数
4. ✅ 保存结果到输出目录
5. ✅ 生成 JSON 摘要

**模板**: `scripts/smoke_batch_{mode}.py`

```python
#!/usr/bin/env python3
"""
{MODE} 模式批量冒烟测试

用法:
    python scripts/smoke_batch_{mode}.py \
        --samples /path/to/samples.tsv \
        --data-dir /path/to/data \
        --output-dir /path/to/output
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sana_wm_pipeline.stage02_pose.mode_{mode} import run_{mode}


def process_sample(sample_id: str, n_frames: int, data_dir: Path, output_dir: Path) -> dict:
    """处理单个样本"""
    print(f"\n{'='*60}")
    print(f"样本: {sample_id}")
    print(f"{'='*60}")
    
    # 输入文件
    video_path = data_dir / f"{sample_id}.mp4"
    # 根据模式加载其他必需文件
    
    # 工作目录
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 开始计时
    start_time = time.time()
    
    try:
        # 加载必需数据（根据模式不同）
        # GT-Pose: camera.npz → c2w, K_px
        # GT-Depth: camera.npz + depth maps
        # Default: 无 GT
        
        # 调用核心处理函数
        artifact = run_{mode}(
            clip_path=video_path,
            # 模式特定参数
            work_dir=sample_dir,
        )
        
        elapsed = time.time() - start_time
        
        # 保存结果
        np.save(sample_dir / "poses.npy", artifact.poses_c2w)
        np.save(sample_dir / "intrinsics.npy", artifact.intrinsics)
        np.save(sample_dir / "scale_per_frame.npy", artifact.scale_per_frame)
        
        # 模式特定输出
        if artifact.depth_downsampled is not None:
            np.save(sample_dir / "depth.npy", artifact.depth_downsampled)
        
        # 生成报告
        result = {
            "sample_id": sample_id,
            "status": "success",
            "elapsed_time": f"{elapsed:.2f}s",
            "output_poses_shape": str(artifact.poses_c2w.shape),
            "output_intrinsics_shape": str(artifact.intrinsics.shape),
            # 模式特定指标
        }
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "sample_id": sample_id,
            "status": "failed",
            "error": str(e),
            "elapsed_time": f"{elapsed:.2f}s",
        }


def main():
    parser = argparse.ArgumentParser(description="{MODE} 模式批量冒烟测试")
    parser.add_argument("--samples", required=True, help="样本列表 (TSV)")
    parser.add_argument("--data-dir", required=True, help="数据目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    args = parser.parse_args()
    
    # 读取样本列表
    samples = []
    with open(args.samples) as f:
        for line in f:
            if line.strip():
                sample_id, n_frames = line.strip().split('\t')
                samples.append((sample_id, int(n_frames)))
    
    # 批量处理
    results = []
    for sample_id, n_frames in samples:
        result = process_sample(sample_id, n_frames, Path(args.data_dir), Path(args.output_dir))
        results.append(result)
    
    # 保存摘要
    summary_file = Path(args.output_dir) / "smoke_test_summary.json"
    with summary_file.open('w') as f:
        json.dump({
            "total_samples": len(samples),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "results": results,
        }, f, indent=2)
    
    # 打印总结
    success_count = sum(1 for r in results if r["status"] == "success")
    if success_count == len(samples):
        print(f"\n🎉 所有样本测试通过！")
        sys.exit(0)
    else:
        print(f"\n❌ 有 {len(samples) - success_count} 个样本失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

### 3.4 样本选择策略

**原则**:
1. ✅ 覆盖不同文件大小（小/中/大）
2. ✅ 覆盖不同帧数（如果数据集有变化）
3. ✅ 覆盖不同场景类型（室内/室外）
4. ✅ 数量适中（5-10 个样本）

**GT-Pose 示例**:
```
样本 1: 5.6M,  303帧 (小文件，快速验证)
样本 2: 6.5M,  303帧 (小文件，稳定性)
样本 3: 19M,   301帧 (中等大小，不同帧数)
样本 4: 28M,   303帧 (大文件，压力测试)
样本 5: 48M,   301帧 (最大文件，极限测试)
```

---

### 3.5 鲁棒性设计

**错误处理清单**:
- [x] 数据目录不存在
- [x] 无有效样本文件
- [x] 必需文件缺失（.mp4, .camera.npz 等）
- [x] 元数据读取失败
- [x] 处理过程异常
- [x] 磁盘空间不足

**实现模式**:
```bash
# Shell 脚本
if [[ ! -d "$DATA_DIR" ]]; then
    echo "❌ 错误：数据目录不存在"
    exit 1
fi

if [[ ! -s "$SAMPLES_FILE" ]]; then
    echo "❌ 错误：未找到有效样本"
    exit 1
fi
```

```python
# Python 脚本
try:
    artifact = run_{mode}(...)
    return {"status": "success", ...}
except Exception as e:
    return {"status": "failed", "error": str(e)}
```

---

## ✅ 阶段 4: 结果验证

### 4.1 验证维度

**基础验证**:
1. ✅ 成功率: 100% (N/N)
2. ✅ 输出文件: 所有必需文件存在
3. ✅ Shape: 所有 shape 正确
4. ✅ Dtype: 所有 dtype 正确

**数据验证**:
1. ✅ GT 数据识别: 哪些是 GT，哪些是计算
2. ✅ 数值范围: 是否合理
3. ✅ 数据一致性: 相关字段是否一致

**逻辑验证**:
1. ✅ 与官方对比: 关键中间结果
2. ✅ 物理意义: 是否符合预期
3. ✅ 边界情况: 特殊输入的处理

---

### 4.2 验证方法

#### 方法 1: 直接对比 GT

```python
# 验证输出是否使用了 GT
output_poses = np.load("poses.npy")
gt_poses = np.load("camera.npz")['c2w']

if np.allclose(output_poses, gt_poses):
    print("✅ poses.npy = GT (未修改)")
else:
    print("❌ poses.npy ≠ GT")
```

#### 方法 2: 数值合理性检查

```python
# 验证 scale 范围
scale = np.load("scale_per_frame.npy")
scale_mean = scale.mean()

if 0.1 < scale_mean < 100:
    print(f"✅ Scale {scale_mean:.2f} 在合理范围")
else:
    print(f"⚠️  Scale {scale_mean:.2f} 可疑")
```

#### 方法 3: 内部一致性检查

```python
# 验证帧数一致性
poses_frames = poses.shape[0]
intrinsics_frames = intrinsics.shape[0]
scale_frames = scale.shape[0]

if poses_frames == intrinsics_frames == scale_frames:
    print(f"✅ 所有输出帧数一致: {poses_frames}")
else:
    print(f"❌ 帧数不一致")
```

---

### 4.3 评估报告模板

**文档**: `{MODE}_{DATASET}_SMOKE_TEST_EVALUATION.md`

```markdown
# {MODE} 模式 {DATASET} 数据集冒烟测试评估

## 测试结果总览
- 总样本数: N
- 成功: N
- 失败: 0
- 成功率: 100%

## ✅ 确认正确的部分
1. **输出 X**: 验证方法 + 结论
2. **输出 Y**: 验证方法 + 结论
3. **指标 Z**: 合理范围 + 实际值

## ⚠️ 可疑的部分
1. **现象**: 描述
2. **分析**: 原因
3. **结论**: 是否正常

## 最终结论
- 核心功能: ✅/❌
- 与官方对齐度: X%
- 是否可进入下一阶段: ✅/❌
```

---

## 📚 文档体系

### 必需文档清单

**代码对齐阶段**:
- [x] `{MODE}_CODE_ALIGNMENT_ANALYSIS.md` - 逐行对比
- [x] `{MODE}_FLOW_COMPARISON.md` - 流程对比
- [x] `{DATASET}_DATA_VERIFICATION_REPORT.md` - 数据结构验证

**测试阶段**:
- [x] `verify_{mode}_alignment.py` - 单元测试脚本
- [x] `smoke_{mode}_{dataset}.sh` - 冒烟测试脚本
- [x] `smoke_batch_{mode}.py` - Python 批处理
- [x] `SMOKE_{MODE}_{DATASET}_REVIEW.md` - 测试审阅

**验证阶段**:
- [x] `{MODE}_{DATASET}_SMOKE_TEST_EVALUATION.md` - 结果评估
- [x] `{MODE}_SMOKE_TEST_FLOW.md` - 流程图
- [x] `{MODE}_OUTPUT_ANALYSIS.md` - 输出数据分析

---

## 🔧 工具和脚本

### 通用工具

**1. 数据扫描脚本**:
```bash
# 自动扫描数据目录生成 samples.tsv
for file in *.mp4; do
    sample_id=$(basename "$file" .mp4)
    n_frames=$(获取帧数)
    echo "$sample_id	$n_frames"
done > samples.tsv
```

**2. 结果验证脚本**:
```python
# 批量验证输出文件
def verify_outputs(output_dir):
    for sample_dir in output_dir.iterdir():
        assert (sample_dir / "poses.npy").exists()
        assert (sample_dir / "intrinsics.npy").exists()
        # ...
```

**3. 数据对比工具**:
```python
# 对比两个目录的输出
def compare_outputs(dir1, dir2):
    for sample in samples:
        poses1 = np.load(dir1 / sample / "poses.npy")
        poses2 = np.load(dir2 / sample / "poses.npy")
        diff = np.abs(poses1 - poses2).max()
        print(f"{sample}: max_diff = {diff}")
```

---

## 🎯 关键经验教训（GT-Pose 案例）

### 1. 路径推断问题

**问题**: GT intrinsics 无法加载  
**原因**: `gt_poses_path` 指向输出目录临时文件  
**修复**: 添加 `gt_intrinsics_path` 显式参数  
**教训**: 不要依赖路径推断，提供显式参数

### 2. 接口设计

**问题**: `run_gtpose()` 需要文件路径，不能传 numpy array  
**解决**: 保存临时文件 `gt_poses.npy`  
**教训**: 保持接口一致性，使用临时文件适配

### 3. 帧数处理

**问题**: 混淆 Pi3X 处理帧数 (64) vs 输出帧数 (303)  
**关键**: Pi3X 子采样节省内存，输出保持完整帧数  
**教训**: 明确区分 `n_scale` (处理) 和 `N` (输出)

### 4. Scale 单值

**问题**: 为什么所有 scale 都一样？  
**原因**: Umeyama Sim(3) 返回单个全局 scale  
**教训**: 理解算法的数学含义，单值广播是正确行为

### 5. 数据验证

**方法**: 
- 从实际数据出发（不虚构）
- 逐项验证（GT vs 计算）
- 检查物理意义（数值范围、一致性）

---

## 📋 检查清单（Checklist）

### 代码对齐 ✅
- [ ] 逐行对比官方代码
- [ ] 所有关键对齐点 100% 一致
- [ ] 单元测试全部通过
- [ ] 文档完整（对比分析、流程图）

### 冒烟测试 ✅
- [ ] Shell 脚本支持自动扫描
- [ ] Python 脚本鲁棒性强
- [ ] 样本选择合理（覆盖不同情况）
- [ ] 错误处理完善
- [ ] 审阅文档清晰

### 结果验证 ✅
- [ ] 100% 成功率
- [ ] 输出文件完整
- [ ] Shape/Dtype 正确
- [ ] GT 数据正确识别
- [ ] 计算数据合理
- [ ] 评估报告详细

### 文档体系 ✅
- [ ] 所有必需文档完成
- [ ] 流程图清晰
- [ ] 可追溯性强
- [ ] 便于后续参考

---

## 🚀 后续模式适配指南

### GT-Depth 模式

**数据要求**:
- ✅ `.mp4` (视频)
- ✅ `.camera.npz` (c2w, K_px)
- ✅ depth maps (GT 深度)

**核心差异**:
- GT depth 用于 SLAM
- MoGe-2 恢复 metric scale
- 需要深度融合逻辑

**参考文档**:
- 官方: `stage.py` gt_depth 分支
- 本地: 参考 GT-Pose 流程

---

### Default 模式

**数据要求**:
- ✅ `.mp4` (仅视频，无 GT)

**核心差异**:
- Pi3X + MoGe-2 融合深度
- VIPE SLAM + per-frame BA
- 完全无 GT，算法最复杂

**参考文档**:
- 官方: `stage.py` default 分支
- 需要理解 VIPE 流程

---

## 📖 总结

### 核心原则

1. **Ponytail 原则**: 官方代码是 Ground Truth
2. **逐行对比**: 不放过任何差异
3. **实际数据**: 所有验证基于真实数据
4. **文档先行**: 先审阅再执行
5. **鲁棒设计**: 完善的错误处理

### 成功标准

- ✅ 代码对齐度 100%
- ✅ 单元测试全通过
- ✅ 冒烟测试成功率 100%
- ✅ 输出数据验证正确
- ✅ 文档体系完整

### 时间估算

**GT-Pose 实际耗时参考**:
- 代码对齐: 4-6 小时
- 单元测试: 2-3 小时
- 冒烟测试开发: 3-4 小时
- 结果验证: 2-3 小时
- **总计**: 1-2 天

---

**方法论文档完成！可用于指导后续模式开发！** 🎯
