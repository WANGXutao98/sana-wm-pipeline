# 🚨 关键发现：阈值参数无效，VIPE Keyframe机制需重新审视

**调查日期**: 2026-08-14  
**Ponytail模式**: 基于实际测试结果的深度分析

---

## 测试结果总结

### 三轮测试对比

| 测试 | filter_thresh | keyframe_thresh | Keyframes | 间隔 | 轨迹比例 |
|------|--------------|----------------|-----------|------|---------|
| 基线 | 2.4 (默认) | 4.0 (默认) | 32-37 | 每1帧 | 2.24-10.70x |
| 测试1 | 10.0 | 4.0 (默认) | 32-37 | 每1帧 | 3.91-11.22x |
| 测试2 | 100.0 | 100.0 | 32-37 | 每1帧 | 3.88-9.56x |

### 关键观察

**即使将阈值增大到100.0（默认值的25-40倍），结果完全没有变化**:
- ❌ Keyframe数量：完全相同（32-37个）
- ❌ 间隔：仍然是连续的（每1帧）
- ❌ 轨迹偏差：没有改善

---

## 结论

### `filter_thresh` 和 `keyframe_thresh` 不是控制因素

**证据链**:
1. ✅ 我们修改了配置文件
2. ✅ 使用了极端值（100.0）
3. ❌ 结果完全没有变化
4. ✅ 说明这些参数**根本没有影响keyframe选择**

### VIPE必定使用了其他机制

**可能性**:

#### 假设A: 代码层面的强制逻辑 ⭐⭐⭐⭐⭐

VIPE源码中可能有：
```python
# 伪代码
if some_condition:
    # 强制每帧都是keyframe，忽略阈值
    return True
```

**需要检查**:
- `MotionFilter.check()` 的完整逻辑
- 是否有其他条件强制返回True
- 是否有环境变量或全局标志

#### 假设B: Pipeline级别的覆盖 ⭐⭐⭐⭐

`DefaultAnnotationPipeline` 可能有特殊逻辑：
- 在phase 2 (第二遍SLAM)中强制所有帧为keyframes
- 我们看到的连续keyframes是phase 2的结果

**证据**: 
```python
# system.py:305 (phase 2)
self._add_keyframe(frame_idx, images, buffer_masks, frame_data_list, phase=2)
```

Phase 2可能无条件添加所有帧。

#### 假设C: 我们的VIPE版本有bug ⭐⭐⭐

可能的bug:
- 配置加载失败，阈值始终使用硬编码的默认值
- MotionFilter被bypass
- 逻辑错误导致阈值比较失效

---

## 修改记录（用于回退）

### 文件1: third_party/vipe/configs/slam/default.yaml

**原始值**:
```yaml
filter_thresh: 2.4
keyframe_thresh: 4.0
```

**当前值**:
```yaml
filter_thresh: 100.0
keyframe_thresh: 100.0
```

**回退命令**:
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
sed -i 's/filter_thresh: 100.0/filter_thresh: 2.4/' third_party/vipe/configs/slam/default.yaml
sed -i 's/keyframe_thresh: 100.0/keyframe_thresh: 4.0/' third_party/vipe/configs/slam/default.yaml
```

### 文件2: third_party/vipe/configs/pipeline/vipe_sanawm.yaml

**原始状态**: 没有 `filter_thresh` 字段

**当前状态**: 添加了 `filter_thresh: 10.0`（但无效）

**回退命令**:
```bash
# 删除第21-25行（filter_thresh相关注释和设置）
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
# 手动编辑或使用以下方式
git checkout third_party/vipe/configs/pipeline/vipe_sanawm.yaml
```

---

## 下一步行动

### 优先级1: 深入审查VIPE源码 ⭐⭐⭐⭐⭐

**目标**: 找到**真正控制keyframe选择的代码**

**检查点**:

1. **Phase 2的逻辑** (`system.py:300-310`)
```python
# SLAM Pass (2/2) - 是否无条件添加所有帧？
for frame_idx, frame_data_list in enumerate(...):
    self._add_keyframe(frame_idx, ..., phase=2)  # 没有条件判断？
```

2. **MotionFilter.check() 的实际执行**
- 添加debug日志打印 `self.thresh` 的值
- 确认是否真的被调用
- 检查返回值

3. **环境变量或全局标志**
```bash
grep -rn "FORCE.*KEYFRAME\|ALL.*KEYFRAME\|DENSE" third_party/vipe --include="*.py"
```

### 优先级2: 检查SpatialVID数据生成的真实方法 ⭐⭐⭐⭐

**可能性**: SpatialVID根本不是用我们看到的这套配置生成的

**验证方法**:
1. 查看SpatialVID论文的数据生成章节
2. 联系作者询问具体配置
3. 检查是否有未公开的脚本或参数

### 优先级3: 考虑替代方案 ⭐⭐⭐

**如果无法控制VIPE的keyframe选择**:

**方案A: 后处理稀疏化**
- 运行VIPE得到连续keyframes
- 事后每4帧抽取1个
- 重新运行BA（如果可能）

**方案B: 预处理视频**
- 降低视频帧率（50帧 → 13帧）
- 输入到VIPE
- 插值回原始帧率

**方案C: 接受偏差**
- 如果下游任务不敏感
- 使用连续keyframes的结果
- 在训练中学习这个偏差

---

## 当前状态

**问题**: 连续keyframes导致轨迹偏差3.88-9.56x  
**尝试**: 修改阈值参数 → ❌ 完全无效  
**结论**: `filter_thresh` 和 `keyframe_thresh` 不是控制因素  
**下一步**: 深入审查VIPE源码，找到真正的控制机制

---

## Ponytail原则应用

**"Read fully, then be lazy"**: 
- ✅ 我们修改了参数
- ❌ 但没有验证参数是否真的被使用
- 💡 应该先在VIPE源码中找到参数的**实际使用位置**，而不是盲目修改配置

**"Bug fix = root cause"**:
- ❌ 我们假设阈值是控制因素
- ✅ 测试证明假设错误
- 💡 需要回到源码，找到**真正的**keyframe选择逻辑

**下一步必须**: 在VIPE源码中添加debug日志，追踪实际的执行路径。
