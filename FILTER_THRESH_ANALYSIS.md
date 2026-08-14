# filter_thresh: 10.0 修复失败分析

**测试日期**: 2026-08-14  
**结果**: ❌ 仍然是连续keyframes

---

## 测试结果

| 样本 | Keyframe数量 | 间隔 | 轨迹比例 | 状态 |
|------|-------------|------|---------|------|
| 样本1 | 32 (连续) | 每1帧 | 3.91x | ❌ 无改善 |
| 样本2 | 35 (连续) | 每1帧 | 11.22x | ❌ 无改善 |
| 样本3 | 37 (连续) | 每1帧 | 5.90x | ❌ 无改善 |

**关键发现**: 
- Keyframe数量和间隔**完全没有变化**
- 轨迹偏差甚至**略微增大**（样本1: 2.24x → 3.91x）

---

## 可能的原因

### 原因1: 配置未生效（最可能）⭐⭐⭐⭐⭐

**Hydra配置覆盖规则**: 
```yaml
defaults:
  - /slam: default  # 先加载 slam/default.yaml

slam:
  filter_thresh: 10.0  # 可能没有正确覆盖？
```

**问题**: Hydra可能要求特定的覆盖语法，或者我们的位置不对。

**验证方法**: 
- 直接修改 `configs/slam/default.yaml` 中的默认值
- 或者查看VIPE运行日志，确认实际使用的阈值

### 原因2: MotionFilter有其他逻辑 ⭐⭐⭐

回顾 `motion_filter.py:138-140`:
```python
if (dense_motion_score > self.thresh 
    or sparse_motion_score > self.thresh * 2):
    return True
```

**可能**: 
- `sparse_motion_score` 使用 `thresh * 2` (即10.0 * 2 = 20.0)
- 仍然不够高，每帧的sparse motion都超过20.0

### 原因3: 视频本身motion极大 ⭐

这3个测试视频可能运动特别剧烈，导致即使阈值10.0也不够。

---

## 下一步尝试

### 方案A: 直接修改 slam/default.yaml ⭐⭐⭐⭐⭐

**最保险的方法**，避免配置覆盖问题：

```bash
# 修改默认配置文件
vim third_party/vipe/configs/slam/default.yaml
# 将 filter_thresh: 2.4 改为 filter_thresh: 50.0
```

**优点**: 
- 绕过Hydra的覆盖规则
- 100%确保生效

**缺点**: 
- 修改了全局默认值
- 影响所有使用VIPE的场景

### 方案B: 使用极大的阈值 ⭐⭐⭐⭐

尝试 `filter_thresh: 100.0` 或 `filter_thresh: 1000.0`

**原理**: 
- 如果10.0不够，可能需要更大的值
- 极大的阈值应该能产生明显效果

### 方案C: 禁用MotionFilter ⭐⭐⭐

查找VIPE是否有选项完全禁用motion-based filtering，改用固定间隔。

---

## 建议

**立即尝试方案A + 方案B的组合**:
1. 直接修改 `configs/slam/default.yaml`
2. 设置 `filter_thresh: 100.0` (极大值)
3. 重新运行测试

如果仍然是连续keyframes，说明问题不在 `filter_thresh`，需要重新审视VIPE的keyframe选择机制。
