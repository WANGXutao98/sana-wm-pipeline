# 阶段11深度调查总结（2026-08-14）

## P0：验证融合深度的物理单位 ✅ 已完成

**结论：融合深度的物理单位和数值范围正常**

- 样本1: 2.5-36.0m, 均值12.4m
- 样本2: 0.7-73.5m, 均值16.3m
- 样本3: 1.3-109.3m, 均值24.0m
- Scale: 0.7-2.4，CoV < 0.03（远低于论文阈值2.0）

✅ **P0排除：不是融合深度的单位问题**

---

## P1：调查VIPE深度处理 🔄 进行中

### 已检查的VIPE代码路径

1. **Pi3xMogeModel (`pi3xmoge.py:55-69`)**
   - 正确读取 `fused.npy` 并返回 metric_depth
   - 返回格式：`(V, H, W)` torch.Tensor
   - 添加了诊断日志（待重跑测试查看）

2. **SLAM系统使用深度 (`system.py:167-169`)**
   ```python
   disp_sens = frame_data.metric_depth[3::8, 3::8]  # 8x下采样
   disp_sens = torch.where(disp_sens > 0, disp_sens.reciprocal(), disp_sens)
   self.buffer.disps_sens[kf_idx, view_idx] = disp_sens
   ```
   - VIPE内部使用**逆深度（disparity）**
   - 直接取 `1 / metric_depth`

3. **焦距缩放逻辑 (`buffer.py:update_disps_sens`)**
   ```python
   if depth_model.depth_type == DepthType.METRIC_DEPTH:
       self.disps_sens *= (last_depth_intrinsics[0][0] / intrinsics[0][0])
   ```
   - 当内参变化时，VIPE用焦距比例缩放逆深度
   - **但实际测试显示：内参几乎不变（fx_ratio=1.0）**
   - ✅ **排除：焦距缩放不是问题根源**

4. **深度对齐逻辑 (`alignment.py:align_inv_depth_to_depth`)**
   - 这是VideoDepthAnything与prompt depth的对齐
   - SANA-WM配置：`depth_align_model: null`
   - **不应该触发此逻辑**

### 关键未解之谜

**为什么轨迹偏大2.5-10x，但修正因子不一致？**
- 样本1: 2.46x → 需要修正因子 0.407
- 样本2: 10.73x → 需要修正因子 0.093

如果是某个固定的normalize，两个样本的修正因子应该相同。

---

## 待验证假设（P2优先级）

### 假设A：VIPE读取深度时有额外处理

可能位置：
- `update_disps_sens` 中调用 `depth_model.estimate()` 时的后处理
- BA初始化时对 `disps_sens` 的normalize

**验证方法**：
1. 添加日志到 `buffer.py:update_disps_sens` 的 depth_model.estimate() 调用前后
2. 记录返回的 metric_depth 的数值范围
3. 对比预计算的 fused.npy 和VIPE实际使用的深度

### 假设B：BA优化过程调整了深度scale

VIPE的BA可能优化的变量：
- `disps` (逆深度)
- `poses` (位姿)
- `intrinsics` (内参)

**验证方法**：
1. 查看BA的优化变量定义
2. 检查是否有全局scale参数
3. 对比BA前后的 `disps_sens` 数值

### 假设C：不同样本的深度统计量导致不同的缩放

观察到：
- 样本2的深度范围最大（0.7-73.5m），偏差也最大（10.73x）
- 样本1的深度范围最小（2.5-36.0m），偏差最小（2.46x）

可能存在基于深度统计量的自动缩放？

**验证方法**：
1. 计算融合深度的统计量（mean, median, percentile）
2. 查找VIPE代码中是否有基于这些统计量的normalize
3. 尝试用统计量解释修正因子的差异

---

## 下一步行动

### 立即行动：重跑带日志的测试

1. 清理旧输出
2. 重新运行冒烟测试（已添加Pi3xMogeModel日志）
3. 查看日志确认：
   - 每个keyframe的深度范围
   - signature匹配是否正确
   - 深度值是否与预计算一致

### 后续调查（如果日志正常）

按P2优先级检查假设A、B、C

### 如果仍无法定位

考虑P3：运行官方 `sana-wm-data-clean` 的 Reference Backend 作为baseline对比

---

## 已排除的可能原因

1. ❌ 融合深度物理单位错误（P0已验证）
2. ❌ Scale未传递（阶段9已修复且生效）
3. ❌ 本地vs官方VIPE调用不同（阶段11调查1）
4. ❌ 验证脚本选错参考（阶段11调查3）
5. ❌ VIPE焦距缩放逻辑（P1已验证内参不变）

---

## 代码修改记录

### 已添加诊断日志

**文件**: `third_party/vipe/vipe/priors/depth/pi3xmoge.py:62-65`
```python
print(f"[Pi3xMogeModel] matched frame {si}/{len(self._fused)}, "
      f"depth range: {depth.min():.2f}-{depth.max():.2f}m, "
      f"mean: {depth.mean():.2f}m, shape: {depth.shape}->{(h,w)}")
```

### 待添加日志（如需要）

1. `vipe/slam/components/buffer.py:update_disps_sens` - 记录depth_model.estimate()返回值
2. `vipe/slam/components/backend.py` - 记录BA前后的disps变化

---

**最后更新**: 2026-08-14 Ponytail模式调查
**状态**: P1进行中，等待重跑测试查看深度读取日志
