# 🔍 官方实现调查：VIPE Keyframe策略的真相

**调查日期**: 2026-08-14  
**方法**: Ponytail - 基于实际代码和数据的深度分析  
**状态**: ❓ 发现矛盾，需要进一步验证

---

## 调查结果

### 1. 官方配置文件

**位置**: `sana-wm-data-clean/vipe_patches/sanawm_pipeline.yaml`

**内容**: 与我们的 `vipe_sanawm.yaml` **完全相同**
```yaml
slam:
  keyframe_depth: pi3xmoge
  optimize_intrinsics: true
  ba:
    fused: false
  # 注意：没有设置 filter_thresh
```

**结论**: 官方使用默认的 `filter_thresh: 2.4`（来自 `slam/default.yaml`）

---

### 2. 官方VIPE调用

**代码**: `sana_wm_data/pose/vipe_cli.py:152-155`

```python
subprocess.run(
    [cfg["vipe_bin"], "infer", video, "-o", str(vipe_out), "-p", "sanawm"],
    check=True, env=vipe_env,
)
```

**结论**: 
- ✅ 使用相同的pipeline: `sanawm`
- ✅ 没有额外的命令行参数
- ✅ 调用方式与我们完全一致

---

### 3. SpatialVID数据集的Keyframe特征

**实测数据**:
```
样本1: vipe_sparse_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
       间隔 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
       
样本2: vipe_sparse_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
       间隔 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
       
样本3: vipe_sparse_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52]
       间隔 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
```

**关键观察**:
- ✅ **所有间隔都精确是4帧**
- ✅ VIPE处理了50帧，选出13个keyframes
- ❓ 这**不可能**是motion-based filter的自然结果（motion不会这么均匀）

---

## 矛盾之处

### 矛盾1: 精确的4帧间隔

**MotionFilter的逻辑** (`vipe/slam/components/motion_filter.py`):
```python
if dense_motion_score > self.thresh:
    return True  # 添加keyframe
```

**问题**: 
- Motion是连续变化的，不可能产生精确的等间隔
- 除非：
  1. **视频极度规律**（每4帧恰好运动超过阈值）→ 不现实
  2. **有其他机制强制固定间隔**

### 矛盾2: 我们的输出是连续keyframes

**我们的结果**:
```
样本1: indices = [0,1,2,3,...,31] (连续32个)
```

**使用相同配置**:
- ✅ 相同的 `filter_thresh: 2.4` (默认)
- ✅ 相同的pipeline配置
- ✅ 相同的VIPE调用方式

**为什么结果不同？**

---

## 可能的解释

### 假设A: SpatialVID使用了不同版本的VIPE ⭐⭐⭐⭐

**证据**:
- SpatialVID数据集可能是几个月前生成的
- VIPE在那之后可能有算法更新
- `filter_thresh`的默认值可能改变过

**验证方法**:
```bash
# 检查VIPE的git历史
cd third_party/vipe
git log --all --oneline --grep="filter\|motion\|keyframe" | head -20
git log --all -p -- vipe/config/slam.yaml | grep -A 5 -B 5 "filter_thresh"
```

### 假设B: SpatialVID手动后处理了keyframes ⭐⭐⭐

**可能的流程**:
1. 运行VIPE，得到原始keyframes
2. 手动筛选为每4帧一个
3. 保存到数据集

**证据**:
- `vipe_sparse_indices` 和 `vipe_sparse_c2w` 可能是后处理结果
- 元数据中的 `vipe_frame_skip: 4` 是描述性的，不是VIPE的输入

**问题**: 
- 为什么要这么做？
- 如果是后处理，为什么轨迹仍然准确？

### 假设C: 我们的VIPE配置有隐藏差异 ⭐⭐

**可能被忽略的地方**:
1. **环境变量**: VIPE可能读取环境变量覆盖配置
2. **VIPE版本**: 我们安装的VIPE可能不是官方使用的版本
3. **运行时patch**: `vipe_patches/` 可能有额外的修改

**验证方法**:
```bash
# 检查vipe_patches
cat /mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-wm-data-clean/vipe_patches/apply_vipe_patches.sh
```

### 假设D: 视频内容差异导致MotionFilter行为不同 ⭐

**理论**:
- SpatialVID的视频可能运动非常规律
- 我们测试的3个样本可能运动更剧烈
- 导致motion filter几乎每帧都触发

**问题**:
- 为什么所有3个SpatialVID样本都是精确4帧间隔？
- 统计上不太可能

---

## 当前结论

### 已确认的事实

1. ✅ 官方配置与我们的配置**完全相同**
2. ✅ 官方VIPE调用与我们的**完全相同**
3. ✅ SpatialVID的keyframes是**精确4帧间隔**
4. ✅ 我们的keyframes是**连续的**（每1帧）
5. ❓ **无法从代码解释这个差异**

### 最可能的解释

**假设A（不同VIPE版本）** 是最合理的：
- VIPE可能在过去几个月更新了默认阈值或算法
- 或者有bug修复改变了行为
- 需要查看VIPE的git历史

### 需要进一步验证

1. **检查VIPE版本历史**
   - 对比当前VIPE与几个月前的版本
   - 查找 `filter_thresh` 或 `MotionFilter` 的变更

2. **检查vipe_patches**
   - 官方可能有额外的patch
   - 我们应用patch的方式可能有差异

3. **直接测试不同阈值**
   - 既然代码无法解释，用实验验证
   - 测试 `filter_thresh: 5.0, 10.0, 15.0` 看能否产生4帧间隔

---

## 修复建议（基于当前理解）

### 方案1: 增大filter_thresh（推荐）⭐⭐⭐⭐⭐

即使无法从代码解释SpatialVID的精确4帧间隔，我们仍然可以：
- 增大 `filter_thresh` 到 10-15
- 实验找到产生稀疏keyframes的阈值
- 验证是否能减小轨迹偏差

**理由**: 
- 连续keyframes → 短基线 → scale漂移（这个机制是明确的）
- 稀疏keyframes应该能改善（无论SpatialVID如何做到的）

### 方案2: 直接联系NVIDIA VIPE团队

询问：
- SpatialVID数据集使用的VIPE版本/配置
- 如何产生精确4帧间隔的keyframes
- 是否有未公开的配置参数

---

## 下一步行动

### 优先级1: 检查vipe_patches

```bash
cat /mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-wm-data-clean/vipe_patches/apply_vipe_patches.sh
ls -la /mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-wm-data-clean/vipe_patches/
```

查看是否有修改 `MotionFilter` 或 `filter_thresh` 的patch。

### 优先级2: 实验验证

即使无法从代码解释，也可以通过实验：
```bash
# 测试 filter_thresh: 10.0
# 看是否能产生稀疏keyframes和正确的轨迹
```

### 优先级3: VIPE版本历史

```bash
cd third_party/vipe
git log --all --oneline --since="2025-01-01" -- vipe/slam/components/motion_filter.py
git log --all --oneline --since="2025-01-01" -- vipe/config/slam.yaml
```

---

## 文件记录

1. `OFFICIAL_VIPE_INVESTIGATION.md` - 本文档
2. `FINAL_ROOT_CAUSE_AND_SOLUTION.md` - 之前的根因分析
3. `THREE_WAY_POSE_COMPARISON.md` - 三方pose对比

---

**状态**: 官方代码与我们**完全相同**，但结果**完全不同**  
**矛盾**: SpatialVID的精确4帧间隔无法从代码解释  
**下一步**: 检查vipe_patches和VIPE版本历史  
**修复方案**: 仍然推荐增大filter_thresh（实验验证）
