# 阶段11调查报告：轨迹长度偏差根因分析

**调查日期**: 2026-08-14  
**调查模式**: Ponytail (高效诊断)  
**问题**: 轨迹长度比VIPE参考大 2.5-10.7x

---

## 执行摘要

✅ **P0完成**: 融合深度物理单位正常（0.7-109m，米制）  
✅ **P1部分完成**: 排除了焦距缩放问题（内参不变）  
❌ **根本原因**: 尚未定位，但缩小到3个假设  

**关键发现**: 不同样本需要的修正因子不一致（0.407 vs 0.093），说明**不是固定的normalize问题**，可能是**基于深度统计量的自适应缩放**。

---

## P0: 融合深度物理单位验证 ✅

### 测试数据

| 样本 | 融合深度范围(m) | Scale范围 | CoV | 轨迹比例 |
|------|----------------|----------|-----|---------|
| 样本1 | 2.5-36.0, 均值12.4 | 1.399-1.409 | 0.0023 | 2.46x ❌ |
| 样本2 | 0.7-73.5, 均值16.3 | 0.694-0.756 | 0.0265 | 10.73x ❌ |
| 样本3 | 1.3-109.3, 均值24.0 | 2.322-2.369 | 0.0057 | - |

### 结论

✅ **深度范围合理**: 0.7-109m符合室内外真实场景  
✅ **Scale正常**: 0.7-2.4在论文范围内  
✅ **CoV优秀**: < 0.03，远低于论文阈值2.0  
✅ **Scale已传递**: pose_artifact中正确记录  

**P0排除**: 不是融合深度的物理单位或数值问题。

---

## P1: VIPE深度处理调查 🔄

### 已检查的代码路径

#### 1. Pi3xMogeModel (`pi3xmoge.py`)

```python
def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
    rgb = _rgb_hwc(src.rgb)
    si = self._match(rgb)  # 16x16 RGB签名匹配
    depth = self._fused[si].astype(np.float32)  # 从预计算读取
    depth = _resize(depth, (h, w))
    return DepthEstimationResult(metric_depth=torch.from_numpy(depth)[None].float())
```

✅ **正确读取预计算的 fused.npy**  
✅ **返回格式正确**: `(V, H, W)` metric_depth  
⏳ **已添加诊断日志**: 待重跑测试查看实际输出

#### 2. VIPE SLAM深度使用 (`system.py:167-169`)

```python
disp_sens = frame_data.metric_depth[3::8, 3::8]  # 8x下采样
disp_sens = torch.where(disp_sens > 0, disp_sens.reciprocal(), disp_sens)
self.buffer.disps_sens[kf_idx, view_idx] = disp_sens
```

- VIPE内部使用**逆深度（disparity）**
- 直接从metric_depth取倒数
- 无额外normalize（在此处）

#### 3. 焦距缩放逻辑 (`buffer.py:update_disps_sens`)

```python
if depth_model.depth_type == DepthType.METRIC_DEPTH:
    self.disps_sens *= (last_depth_intrinsics[0][0] / intrinsics[0][0])
```

**实测结果**:
- 样本1: fx 692.8 → 692.8 (ratio=1.0)
- 样本2: fx 761.4 → 761.4 (ratio=1.0)

✅ **P1排除**: 内参几乎不变，焦距缩放不是问题。

#### 4. 深度对齐 (`alignment.py`)

```python
_, scale_tensor, bias_tensor = align_inv_depth_to_depth(
    video_depth_inv_depth,  # VideoDepthAnything
    prompt_result,          # 我们的metric_depth
)
```

- 这是VideoDepthAnything与prompt depth的对齐
- SANA-WM配置: `depth_align_model: null`
- ✅ **P1排除**: 不触发此逻辑

---

## 关键未解之谜

### 为什么修正因子不一致？

| 样本 | 深度范围(m) | 轨迹比例 | 需要的修正因子 |
|------|-----------|---------|--------------|
| 样本1 | 2.5-36.0 | 2.46x | 0.407 |
| 样本2 | 0.7-73.5 | 10.73x | 0.093 |

**观察**: 
- 样本2的深度范围**最大**（73.5m），偏差也**最大**（10.73x）
- 样本1的深度范围**较小**（36.0m），偏差**较小**（2.46x）

**推测**: 可能存在**基于深度统计量的自适应缩放**（如median/mean normalize）。

---

## 待验证假设（P2优先级）

### 假设A: VIPE读取深度后有额外处理

**可能位置**:
- `buffer.py:update_disps_sens` 调用 `depth_model.estimate()` 的后处理
- BA初始化时对 `disps_sens` 的normalize

**验证方法**:
1. 添加日志到 `buffer.py:update_disps_sens` (depth_model.estimate()前后)
2. 记录返回的metric_depth数值范围
3. 对比预计算的fused.npy和VIPE实际使用的深度

### 假设B: BA优化过程调整了深度scale

**VIPE BA优化变量**:
- `disps` (逆深度)
- `poses` (位姿)
- `intrinsics` (内参)

**验证方法**:
1. 查看BA的优化变量定义（backend.py）
2. 检查是否有全局scale参数
3. 对比BA前后的 `disps_sens` 数值

### 假设C: 基于深度统计量的自适应缩放 ⭐

**动机**: 解释修正因子不一致

**可能的normalize逻辑**:
```python
# 假设的normalize（待在代码中查找）
depth_median = np.median(depth)
depth_normalized = depth / depth_median * TARGET_SCALE
```

**验证方法**:
1. 计算融合深度的统计量（mean, median, P10, P90）
2. 尝试用统计量解释修正因子：
   - 样本1: median ≈ 12.4m, 修正因子 0.407 → 目标median ≈ 5.0m?
   - 样本2: median ≈ 16.3m, 修正因子 0.093 → 目标median ≈ 1.5m?
3. 搜索VIPE代码中是否有基于median/mean的normalize

---

## 下一步行动计划

### 立即行动（优先级1）

**重跑带日志的冒烟测试**

```bash
# 已添加日志到 pi3xmoge.py:62-65
# 预期看到每个keyframe的深度范围
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_spatialvid.sh
```

**查看日志确认**:
- 每个keyframe的深度范围（应该是1-100m左右）
- Signature匹配索引是否合理
- 深度值是否与预计算的fused.npy一致

### 后续调查（优先级2）

**如果日志显示深度读取正常**:

1. **假设C优先**: 计算深度统计量，尝试解释修正因子
2. **假设A**: 添加更多日志到buffer.py
3. **假设B**: 添加日志到backend.py的BA过程

**如果日志显示深度异常**:
- 直接修复Pi3xMogeModel的问题

### Baseline对比（优先级3）

**如果P2仍无法定位**:

运行官方 `sana-wm-data-clean` 的 Reference Backend:
```bash
cd sana-wm-data-clean
python3 -m sana_wm_data.camera_cli \
  video.mp4 --out /tmp/ref_out --backend reference
```

对比轨迹长度，定位问题在:
- **前端**（融合深度计算）
- **后端**（VIPE SLAM/BA）

---

## 已排除的可能原因

| 假设 | 验证方法 | 结果 |
|------|---------|------|
| 融合深度单位错误 | P0实测深度范围 | ❌ 排除（0.7-109m正常）|
| Scale未传递 | 检查artifact | ❌ 排除（阶段9已修复）|
| VIPE调用差异 | 对比代码 | ❌ 排除（完全相同）|
| 验证参考错误 | 对比两套标注 | ❌ 排除（vipe_c2w正确）|
| 焦距缩放 | 实测内参变化 | ❌ 排除（fx_ratio=1.0）|
| 深度对齐 | 检查配置 | ❌ 排除（depth_align_model=null）|

---

## 代码修改记录

### 已添加的诊断日志

**文件**: `third_party/vipe/vipe/priors/depth/pi3xmoge.py:62-65`

```python
print(f"[Pi3xMogeModel] matched frame {si}/{len(self._fused)}, "
      f"depth range: {depth.min():.2f}-{depth.max():.2f}m, "
      f"mean: {depth.mean():.2f}m, shape: {depth.shape}->{(h,w)}")
```

### 待添加日志（如需要）

1. `buffer.py:update_disps_sens` - depth_model.estimate()返回值
2. `backend.py` - BA前后disps变化
3. `buffer.py` - disps_sens的统计量（median/mean）

---

## 文件输出

1. **本报告**: `INVESTIGATION_STAGE11_SUMMARY.md`
2. **诊断脚本**: `scripts/diagnose_depth_scale.py`（焦距缩放检查）
3. **架构文档**: `SANA_WM_DATA_CLEAN_ARCHITECTURE.md`（双后端澄清）

---

**调查状态**: P0✅ P1🔄 P2⏳  
**下一步**: 重跑测试查看Pi3xMogeModel日志  
**预期突破点**: 假设C（深度统计量normalize）
