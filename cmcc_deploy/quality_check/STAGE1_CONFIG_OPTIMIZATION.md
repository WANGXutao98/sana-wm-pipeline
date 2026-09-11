# Stage 1 配置优化说明

**日期**: 2026-07-08  
**目标**: 提升 Stage 1 通过率，从 76% → 92%（+22%）

---

## 🎯 优化目标

### 问题1：DL3DV 没有 caption（90% flag）
- **现状**: 9,937 样本中只有 587 pass (5.9%)
- **原因**: 8,963 个样本因 `caption_len=7 < 50` 被 flag
- **根本原因**: DL3DV 数据集的 caption 字段为 `null`（数据源问题）

### 问题2：游戏数据包含 camera words（60-70% flag）
- **现状**: 
  - OmniWorld: 2,416/6,145 pass (39%)
  - sekai-game-drone: 288/931 pass (31%)
  - sekai-game-walking: 596/1,602 pass (37%)
- **原因**: 80-90% 的 flag 是因为 caption 包含 camera action words
  - "camera follows", "camera pans", "camera moves" 等
- **根本原因**: 游戏引擎生成的 caption 描述相机运动，而非场景内容

---

## ✅ 实施的解决方案

### 修改1：添加 `check_camera_words` 配置项

**文件**: `src/sana_wm_pipeline/qc/group_config.py`

```python
@dataclass(frozen=True)
class GroupConfig:
    # ...
    check_camera_words: bool = True  # 新增：是否检查 camera action words
    # ...
```

**逻辑修改**（line 104-107）:
```python
# Camera action words → flag for Qwen rewrite in Stage 3
cw = metrics.get("camera_words", [])
if cw and cfg.check_camera_words:  # 添加 check_camera_words 条件
    flag_reasons.append(f"camera_word: {cw[0]!r} (+{len(cw)-1} more)")
```

---

### 修改2：DL3DV 配置优化

**修改前**:
```python
_REAL_STRICT = GroupConfig(
    jump_threshold_m=0.5, max_jumps_flag=0, max_jumps_fail=5,
    min_caption_len=50,  # ❌ DL3DV 没有 caption
    ...
)
```

**修改后**:
```python
_REAL_STRICT = GroupConfig(
    jump_threshold_m=0.5, max_jumps_flag=0, max_jumps_fail=5,
    min_caption_len=0,  # ✅ 允许无 caption
    check_camera_words=True,
    ...
)
```

**预期效果**:
- Pass: 587 (5.9%) → **~8,500 (85%)**
- 增加: **+7,913 样本**

---

### 修改3：游戏数据配置优化

#### OmniWorld
```python
_OMNIWORLD = GroupConfig(
    jump_threshold_m=2.0, max_jumps_flag=15, max_jumps_fail=50,
    min_caption_len=10,  # ✅ 降低（从 50 → 10）
    check_camera_words=False,  # ✅ 关闭 camera words 检查
    ...
)
```

**预期效果**:
- Pass: 2,416 (39%) → **~5,500 (90%)**
- 增加: **+3,084 样本**

#### sekai-game-drone
```python
_SEKAI_DRONE = GroupConfig(
    jump_threshold_m=5.0, max_jumps_flag=20, max_jumps_fail=80,
    min_caption_len=10,  # ✅ 降低
    check_camera_words=False,  # ✅ 关闭
    ...
)
```

**预期效果**:
- Pass: 288 (31%) → **~850 (91%)**
- 增加: **+562 样本**

#### sekai-game-walking
```python
_SEKAI_GAME_WALKING = GroupConfig(
    jump_threshold_m=2.0, max_jumps_flag=15, max_jumps_fail=50,
    min_caption_len=10,  # ✅ 降低
    check_camera_words=False,  # ✅ 关闭
    ...
)
```

**预期效果**:
- Pass: 596 (37%) → **~1,450 (90%)**
- 增加: **+854 样本**

---

## 📊 预期总体效果

| Group | 当前 Pass | 优化后 Pass | 增加 |
|-------|-----------|-------------|------|
| DL3DV-ALL-2K | 587 (6%) | 8,500 (85%) | +7,913 |
| OmniWorld-Game | 2,416 (39%) | 5,500 (90%) | +3,084 |
| sekai-game-drone | 288 (31%) | 850 (91%) | +562 |
| sekai-game-walking | 596 (37%) | 1,450 (90%) | +854 |
| SpatialVID-hq | 35,042 (93%) | 35,042 (93%) | 0 |
| sekai-real-walking-hq | 18,201 (95%) | 18,201 (95%) | 0 |
| **总计** | **57,130 (76%)** | **~69,500 (92%)** | **+12,413 (+22%)** |

---

## 🚀 执行步骤

### 1. 确认修改

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 查看修改
git diff src/sana_wm_pipeline/qc/group_config.py
```

### 2. 运行重跑脚本

```bash
# 在 CMCC 机器上
cd /root/work/david_work/sana_wm_qc

# 复制脚本
cp /mnt/afs/davidwang/workspace/sana_wm_pipeline/rerun_optimized_groups.sh .

# 给予执行权限
chmod +x rerun_optimized_groups.sh

# 在 tmux 中运行
tmux new -s qc_rerun
./rerun_optimized_groups.sh

# 或后台运行
nohup ./rerun_optimized_groups.sh > qc_rerun.log 2>&1 &
```

### 3. 预计时间

- DL3DV-ALL-2K: ~30 分钟
- OmniWorld-Game: ~15 分钟
- sekai-game-drone: ~5 分钟
- sekai-game-walking: ~10 分钟

**总计**: ~60 分钟

---

## ✅ 验证结果

运行完成后：

```bash
# 对比优化前后
for group in DL3DV-ALL-2K OmniWorld-Game sekai-game-drone sekai-game-walking; do
  echo "=== $group ==="
  echo -n "Before: "; jq -r '.verdict' qc_full_output/${group}.backup_*/stage1_results.jsonl 2>/dev/null | sort | uniq -c
  echo -n "After:  "; jq -r '.verdict' qc_full_output/${group}/stage1_results.jsonl 2>/dev/null | sort | uniq -c
  echo ""
done

# 查看 flag 原因分布（应该看到 caption/camera words 大幅减少）
for group in DL3DV-ALL-2K OmniWorld-Game sekai-game-drone sekai-game-walking; do
  echo "=== $group flag reasons (after) ==="
  jq -r 'select(.verdict=="flag") | .flag_reasons[]' qc_full_output/${group}/stage1_results.jsonl | sort | uniq -c | sort -rn | head -10
  echo ""
done
```

---

## 🎯 设计理念

### 为什么这样修改？

1. **Caption 问题可以延后处理**
   - Stage 3 的 Qwen VLM 可以生成/改写 caption
   - Stage 1 不应因 caption 丢弃有价值的 pose/视频数据

2. **保持 Pose 质量把关**
   - `n_jumps`, `scale_cv` 等 pose 检查保持严格
   - Pose 质量无法在后续 Stage 修复

3. **Per-Group 配置**
   - 不同数据源有不同特点
   - 游戏数据：pose 完美，caption 技术化
   - 真实数据：pose 有噪声，caption 内容化

### Stage 3 的作用

当 Stage 3 开发完成后：
- ✅ **Qwen VLM** 为 DL3DV 生成新的 caption
- ✅ **Qwen VLM** 改写游戏数据的 camera words caption
- ✅ **UniMatch 光流** 检测运动连续性
- ✅ **DOVER** 评估视觉美学质量

---

## 📝 注意事项

1. **备份自动创建**
   - 重跑脚本会自动备份旧结果到 `${output_dir}.backup_TIMESTAMP`
   - 可以对比优化前后的效果

2. **不影响其他 group**
   - SpatialVID-hq 和 sekai-real-walking-hq 配置不变
   - 它们本身已经有很高的通过率

3. **后续工作**
   - 重跑完成后，可以继续原计划的人工审查
   - 或者直接进入 Stage 3 开发和执行

---

## 🔗 相关文档

- 原设计文档: `docs/superpowers/specs/2026-07-08-large-scale-human-review-implementation.md`
- Group 配置代码: `src/sana_wm_pipeline/qc/group_config.py`
- 重跑脚本: `rerun_optimized_groups.sh`
