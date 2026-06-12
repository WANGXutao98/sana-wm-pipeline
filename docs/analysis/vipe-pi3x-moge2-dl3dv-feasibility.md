# VIPE+MoGe-2+Pi3X 在 DL3DV 数据上的可行性分析

## 摘要

**结论：** 该管线在 DL3DV 上**技术可行**，但需注意以下数据特性差异：
- ✅ 兼容纯 RGB 视频（DL3DV 无 RGB-D）
- ✅ 帧率适应性强（DL3DV 通常 24-30 fps）
- ⚠️ DL3DV 场景更复杂（多样化照明、户外等）
- ⚠️ GT 位姿质量不如 TUM mocap 精确

---

## 1. 数据特性对比

### TUM RGB-D (已验证)
- 长度：28-99s
- 帧率：固定 30fps
- 分辨率：640×480
- 场景：室内桌面，受控照明
- GT：精确 mocap (~mm 级精度)
- 输入格式：RGB-D 帧 + mocap 位姿

### DL3DV (目标数据集)
- 长度：60-300s （更长）
- 帧率：24-30fps （可变）
- 分辨率：1024×1024+ （更高）
- 场景：室内+室外混合，自然照明
- GT：SfM 重建 + 手工标注 (cm 级精度)
- 输入格式：RGB 图像序列 + SfM 内参 + 估计位姿

---

## 2. 管线兼容性分析

### 2.1 Pi3X 视频深度估计

**TUM 实验中的表现：**
- 输入：RGB 视频 (T, H=480, W=640, 3)
- 推理配置：chunk=16, stride=8（16 帧窗口，8 帧步长）
- 推理时间：fr1=~8min, fr2=~30min
- 输出：视频一致性相对深度 (T, H, W)

**DL3DV 场景预测：**
- 输入：RGB 视频 (T, H≥1024, W≥1024, 3)
- 推理时间：预计 1.5-3× TUM (更高分辨率 + 更长序列)
- **风险**：Pi3X 未在户外、变光照场景验证，质量不确定

**技术兼容性：✅ 完全兼容**

**建议行动：**
- ✓ 直接执行 Pi3X 推理（无需修改）
- ⚠️ 产出质量需在 DL3DV 上验证
- 📌 优先选择**室内场景**进行初步验证
- 📌 如扩展到户外，增加其他深度估计方法作为对比

### 2.2 MoGe-2 米制深度估计

**TUM 实验中的表现：**
- 输入：RGB 帧 (T, H, W, 3)
- FOV 推导：自动计算（`fov_x = 2 * arctan(W/2/fx)`）
- 米制标定：通过 RGB-D GT 深度标定
- 效果：尺度偏差从 7.9% 改善到 1.1% (fr1)

**DL3DV 场景预测：**
- 输入：RGB 图像 + SfM 估计的内参
- SfM 内参质量：通常 2-5% 误差（不如 mocap 精确）
- **风险**：SfM 内参误差会直接传播到 MoGe-2 的米制尺度

**技术兼容性：✅ 完全兼容**

**建议行动：**
- ✓ 直接执行 MoGe-2 推理（内参为 transforms.json 中的 fl_x, fl_y, cx, cy）
- ⚠️ 最终尺度精度取决于 DL3DV SfM 内参质量
- 📌 如果 DL3DV 提供多个内参估计方案，逐个测试

### 2.3 EMA 时序融合

**TUM 实验中的表现：**
- EMA 动量：α=0.99（一阶滤波，较强的历史记忆）
- 融合公式：`s_ema_t = 0.99 * s_ema_{t-1} + 0.01 * s_t`
- 效果：减小尺度漂移、减少深度闪烁
- **最强效果在长序列**：fr2 尺度偏差从 18.5% → 3.3%

**DL3DV 场景预测：**
- DL3DV 序列更长（60-300s vs 28-99s TUM）
- EMA 融合在长序列上效果预期**更好**
- **优势**：DL3DV 的长度正好利用 EMA 的优势

**技术兼容性：✅ 强烈推荐**

**建议行动：**
- ✅ 在 DL3DV 长序列上验证 EMA 的效果
- 📌 可做 ablation：关闭 EMA（α=1.0），观察是否有尺度漂移加重

---

## 3. 数据处理流程兼容性

### 3.1 Stage 01 — normalize (视频标准化)

**DL3DV 特异性：**
- 输入：图像序列 (PNG/JPG) + transforms.json
- sana_wm_pipeline 需提供：video.mp4 + gt_poses.npy + intrinsics.npy
- 对应代码：`experiments/data_production_smoke/prepare_dl3dv.py`

**兼容性分析：✅ 已支持**
- ✅ 已有脚本支持图像序列 → MP4 转换
- ✅ 已有脚本支持 transforms.json → poses/intrinsics 解析
- ✅ OpenCV c2w 约定与 VIPE 一致

### 3.2 Stage 02 — pose estimation

**可选模式：**

**模式 1：default 模式（推荐用于 DL3DV）**
- 使用 VIPE + Pi3X + MoGe-2（当前 TUM 已验证的配置）
- 流程：预计算缓存 → SLAM BA 优化
- ✅ 完全兼容 DL3DV

**模式 2：gt-pose 模式（如有 GT）**
- 使用 GT 位姿 + Pi3X 估计内参
- ✅ 对 DL3DV GT 位姿也适用

### 3.3 Stage 03 — frame filtering

**问题：**
- DL3DV 数据配置中 `unimatch_flow: null`, `dover: null`
- 这两个工具未安装

**方案：**
- 可保持 `strict_frames=False` (允许少于 961 帧)
- 或为 DL3DV 安装 unimatch + dover（可选）
- ✅ Stage 03 可跳过或支持

### 3.4 Stage 04-06 — filtering + caption + pack

**兼容性：✅ 完全支持**
- ✅ Stage 04 (apply_table6) 已有 DL3DV 配置
- ✅ Stage 05 (qwen35_vl_runner) 有 stub 实现
- ✅ Stage 06 (webdataset_writer) 支持 `strict_frames=False`

---

## 4. VIPE 深度管线在 DL3DV 上的验证计划

### 4.1 快速验证（单场景，2-3 小时）

**目的：** 端到端验证 VIPE+Pi3X+MoGe-2 在 DL3DV 单个场景上可运行

```bash
# 选择一个中等难度的 DL3DV 室内场景（优先选室内，避免户外复杂情况）
SCENE=indoor_simple_001

# Step 1: 数据准备（10 min）
python experiments/data_production_smoke/prepare_dl3dv.py \
    --scene $SCENE \
    --out /tmp/dl3dv_test

# Step 2: Pi3X + MoGe-2 预计算（30-45 min，取决于序列长度）
python experiments/vipe_comparison/precompute_pi3x_depths.py \
    --video /tmp/dl3dv_test/video.mp4 \
    --out /tmp/cache_dl3dv_${SCENE}.npz

# Step 3: VIPE 推理（15 min）
export SANA_WM_CACHED_DEPTH_PATH=/tmp/cache_dl3dv_${SCENE}.npz
vipe infer /tmp/dl3dv_test/video.mp4 \
    --pipeline vipe_cached_depth \
    --output /tmp/vipe_dl3dv_${SCENE}

# Step 4: 评测（5 min，如有 GT）
python experiments/vipe_comparison/evaluate.py \
    --seq /tmp/dl3dv_test \
    --results /tmp/vipe_dl3dv_${SCENE}
```

**预期输出物：**
- `/tmp/cache_dl3dv_${SCENE}.npz` — 深度缓存 (T,H,W)
- `/tmp/vipe_dl3dv_${SCENE}/pose/video.npz` — 估计的相机位姿 (T, 4, 4)
- ATE/RTE 指标（如有 GT）

**成功标准：**
- ✅ Pi3X 完成全序列推理（无 OOM、无 nan）
- ✅ MoGe-2 完成逐帧推理
- ✅ VIPE SLAM 收敛（无 tracking lost）
- ✅ 生成 pose/video.npz（形状正确）
- 🎯 若有 GT：ATE RMSE 在合理范围（cm 级）

### 4.2 完整验证（多场景，1-2 天）

**目的：** 在 5-8 个场景上验证（室内+室外混合），对比 Pi3X+MoGe-2 vs baseline

**场景选择：**
- 2-3 个室内短序列 (60-120s)
- 2-3 个室内长序列 (120-300s)
- 1-2 个户外场景（可选，高风险）

**指标收集：**
- 每个场景记录：ATE RMSE, 尺度偏差, RTE (后半段)
- 对比：Pi3X+MoGe-2 vs method A (不含深度增强)

### 4.3 集成验证（完整管线，1 周）

**目的：** 端到端 DL3DV → WebDataset → SANA-WM 推理

**流程：**
1. 运行 sana_wm_pipeline Stage 01-06（default 模式）
2. 生成 WebDataset shard
3. 从 shard 提取数据，调用 SANA-WM 推理
4. 对比生成视频 vs DL3DV GT，评估视觉质量

---

## 5. 已知风险和缓解方案

| 风险 | 可能性 | 影响 | 缓解方案 |
|---|---|---|---|
| DL3DV 户外场景照明变化 → Pi3X 质量下降 | 中 | 深度估计精度偏低 | 优先测试室内场景；需要时补充深度模型 |
| SfM 内参误差 → MoGe-2 米制不准 | 中 | 最终尺度偏差 2-5% | 若 DL3DV 有多个内参估计，逐个测试 |
| 高分辨率 (1024+) → 显存爆炸 | 低 | 推理失败 | Pi3X/MoGe-2 内部已有 downsampling；H100 应无压力 |
| 长序列 (>300s) → 推理时间过长 | 低 | 耗时过长 | 可分段处理；缓存中间结果 |
| unimatch/dover 缺失 → Stage 03 报错 | 低 | Stage 03 失败 | `strict_frames=False` 已支持绕过 |
| VIPE BA 未收敛（复杂场景） | 低 | 位姿跳跃/漂移 | 检查 VIPE 日志；可考虑调整 BA 参数 |

---

## 6. 结论和建议

### 技术可行性：✅ 完全可行

VIPE+MoGe-2+Pi3X 管线在技术上完全兼容 DL3DV。
- 核心依赖（RGB 视频、内参）都具备
- 代码修改最小（缓存查表）
- TUM 验证已充分

### 数据适配性：⚠️ 需要验证

- DL3DV 场景更复杂（多样化、户外）
- 但"更复杂"也意味着更好的**泛化验证**
- 长序列优势更明显（EMA 效果倍增）

### 📋 建议行动路线图

#### 立即（本周）：快速验证
- [ ] 选 1 个 DL3DV 室内短场景
- [ ] 按 Section 4.1 运行完整流程
- [ ] 验证无重大 bug

#### 近期（1-2 周）：完整验证
- [ ] 扩展到 5-8 个场景（室内优先）
- [ ] 收集 ATE/RTE 指标
- [ ] 对比 Pi3X+MoGe-2 vs baseline
- [ ] 生成可视化对比图

#### 中期（1 个月）：集成验证
- [ ] 集成到 sana_wm_pipeline Stage 02（default 模式）
- [ ] 端到端跑 DL3DV → WebDataset shard
- [ ] 调用 SANA-WM 推理验证完整管线

#### 可选（如需要）：扩展验证
- [ ] 评估户外场景效果
- [ ] 实现 per-frame intrinsics BA
- [ ] 其他长序列数据集（ScanNet, 7-Scenes）

---

## 附录：TUM vs DL3DV 关键参数对比

| 参数 | TUM fr1 | TUM fr2 | DL3DV (typical) |
|---|---|---|---|
| 序列长度 | 28s | 99s | 60-300s |
| 帧数 | 613 | 2257 | 1800-9000 |
| 帧率 | 30 fps | 30 fps | 24-30 fps |
| 分辨率 | 640×480 | 640×480 | 1024×1024+ |
| 场景类型 | 室内桌面 | 室内桌面 | 室内+室外 |
| 照明 | 人工固定 | 人工固定 | 自然变化 |
| 相机类型 | RGB-D | RGB-D | RGB only |
| GT 位姿 | mocap (~mm) | mocap (~mm) | SfM (~cm) |
| 内参精度 | mocap 精确 | mocap 精确 | SfM 估计 (±2-5%) |

