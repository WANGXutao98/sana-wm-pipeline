# Stage01-06 系统检查报告

**日期**: 2026-08-15  
**检查人**: Claude (Sonnet 4.6)  
**目标**: 检查本地6个stage实现，核查冒烟测试覆盖，对比官方sana-wm-data-clean

**Ponytail原则**: 官方实现 > 本地实现

---

## 📋 执行摘要

### 模块对应关系

| 本地实现 | 官方实现 | 对应关系 |
|---------|---------|---------|
| `stage01_ingest` | `sana_wm_data/ingest` | ✅ 存在 |
| `stage02_pose` | `sana_wm_data/pose` | ✅ 存在 |
| `stage03_3dgs_aug` | ❌ 无对应 | ⚠️ 本地独有 |
| `stage04_filter` | `sana_wm_data/filter` | ✅ 存在 |
| `stage05_caption` | ❌ 无对应 | ⚠️ 本地独有 |
| `stage06_pack` | ❌ 无对应 | ⚠️ 本地独有 |

### 冒烟测试覆盖情况

**冒烟测试脚本**: `smoke_spatialvid.sh` → `smoke_test_batch.py`

| Stage | 被测试调用 | 调用方式 |
|-------|-----------|---------|
| Stage 01 | ✅ 是 | `normalize_video()` |
| Stage 02 | ✅ 是 | `run_default()` |
| Stage 03 | ❌ 否 | - |
| Stage 04 | ❌ 否 | - |
| Stage 05 | ❌ 否 | - |
| Stage 06 | ⚠️ 部分 | 简化版`pack_shard()` |

**覆盖率**: 2/6 完整覆盖，1/6 简化覆盖

---

## 📊 详细分析

### Stage 01: Ingest (视频导入与归一化)

#### 本地实现位置
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/stage01_ingest/
├── normalize.py      # 视频归一化 (probe + normalize_video)
└── sources.py        # 数据源定义
```

#### 官方实现位置
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-wm-data-clean/sana_wm_data/ingest/
├── __init__.py
└── sekai_game.py     # 数据集特定处理
```

#### 核心功能

**本地实现 (normalize.py)**:
```python
def probe(path: Path) -> VideoInfo:
    """用ffprobe读取视频元信息"""
    # 返回: width, height, fps, duration_s, n_frames

def normalize_video(input_path: Path, output_path: Path) -> VideoInfo:
    """归一化视频到1280x720 @ 16fps"""
    # 论文§5.1 / Appendix D.1: 所有视频重采样到720p/16fps
    # 中心裁剪保持FOV
```

**官方实现**: 
- 主要是数据集特定的摄像机约定辅助函数
- 核心归一化逻辑未在官方仓库明确体现（可能在上游处理）

#### 冒烟测试覆盖

✅ **已覆盖** (`smoke_test_batch.py` 第91-93行):
```python
# Stage 1: 归一化
info = normalize_video(video_path, norm_video)
print(f"  Normalized: {info.n_frames} frames @ {info.fps}fps...")
```

#### 输入输出

| 维度 | 说明 |
|------|------|
| **输入** | 原始视频 (.mp4, 任意分辨率/帧率) |
| **处理** | ffmpeg归一化：1280x720, 16fps, 中心裁剪 |
| **输出** | normalized.mp4 (1280x720 @ 16fps) |
| **质量** | libx264, CRF 23 |

#### 对齐状态

✅ **逻辑对齐论文** (§5.1, Appendix D.1)
- 720p @ 16fps ✅
- 中心裁剪 ✅
- 与SANA-Video实践一致 ✅

⚠️ **官方实现未明确** - 归一化可能在官方pipeline的其他位置

---

### Stage 02: Pose (相机位姿估计)

#### 本地实现位置
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/stage02_pose/
├── mode_default.py     # Default模式 (Pi3X+MoGe-2+VIPE)
├── mode_gtdepth.py     # GT-depth模式
├── mode_gtpose.py      # GT-pose模式
├── depth_fusion.py     # 深度融合
├── umeyama.py          # Umeyama Sim(3)对齐
└── ...
```

#### 官方实现位置
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-wm-data-clean/sana_wm_data/pose/
├── stage.py            # 主处理逻辑 (3种模式)
├── fusion.py           # 深度融合
├── alignment.py        # Umeyama对齐
├── _real.py            # Pi3X/MoGe-2推理
├── vipe_cli.py         # VIPE CLI封装
└── ...
```

#### 核心功能

**三种模式**:

1. **Default** (互联网视频，无GT):
   - Pi3X + MoGe-2融合深度
   - VIPE SLAM
   - Metric scale来自MoGe-2

2. **GT-depth** (OmniWorld，有GT深度):
   - GT depth注入VIPE
   - MoGe-2恢复metric scale

3. **GT-pose** (Sekai/DL3DV，有GT轨迹):
   - 使用GT轨迹
   - Pi3X预测 + Umeyama对齐恢复scale

#### 冒烟测试覆盖

✅ **已覆盖** - Default模式 (`smoke_test_batch.py` 第96-98行):
```python
# Stage 2: VIPE SLAM
art = run_default(norm_video, vipe_work)
print(f"  Poses {art.poses_c2w.shape}  Intr {art.intrinsics.shape}")
```

❌ **未覆盖** - GT-depth和GT-pose模式

#### 输入输出

| 维度 | 说明 |
|------|------|
| **输入** | normalized.mp4 (1280x720 @ 16fps) |
| **处理** | Pi3X+MoGe深度融合 → VIPE SLAM → BA优化 |
| **输出** | poses_c2w (N,4,4), intrinsics (N,4), scale_per_frame (N,) |

#### 对齐状态

✅ **100%对齐官方实现** (已在前次检查中验证)
- Default模式: 代码逻辑一致 ✅
- GT-pose模式: Umeyama算法一致 ✅
- GT-depth模式: 深度融合逐行一致 ✅

**已知问题**: Default模式metric scale偏差1.67x-14.8x
- 根因: MoGe-2模型本身
- 不是代码bug ✅

---

### Stage 03: 3DGS Aug (3D高斯溅射增强)

#### 本地实现位置
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/stage03_3dgs_aug/
├── fcgs_fit.py         # FCGS拟合
├── fcgs_render.py      # FCGS渲染
├── traj_synthesis.py   # 轨迹合成
├── coverage_gate.py    # 覆盖率检查
└── difix3d_refine.py   # DiFix3D细化
```

#### 官方实现位置

❌ **官方仓库无对应模块**

**论文提及** (Appendix B.2):
> "Following HY-WorldPlay, we augment static-scene datasets such as DL3DV by rendering novel one-minute videos from pre-fitted FCGS 3D Gaussian Splats."

#### 核心功能

从论文描述推断:

1. **FCGS拟合**: 对DL3DV静态场景拟合3D高斯
2. **轨迹合成**: 生成40条候选轨迹（orbit, spiral, dolly等）
3. **渲染**: 渲染novel view视频
4. **DiFix3D细化**: 逐帧细化渲染结果
5. **质量过滤**: 覆盖率、饱和度、scene cuts等

#### 冒烟测试覆盖

❌ **未覆盖** - 冒烟测试只测试SpatialVID (真实视频)，不涉及3DGS增强

#### 输入输出

| 维度 | 说明 |
|------|------|
| **输入** | DL3DV静态场景 + FCGS重建 |
| **处理** | 合成轨迹 → FCGS渲染 → DiFix3D细化 → 过滤 |
| **输出** | 合成的60秒720p视频 + 相机轨迹 |

#### 对齐状态

⚠️ **本地独有实现**
- 官方仓库无对应代码
- 论文仅简述流程 (Appendix B.2)
- 无法验证对齐

**建议**: 
- 如果DL3DV数据重要 → 需要独立验证
- 如果只用真实视频 → Stage 03可以跳过

---

### Stage 04: Filter (质量过滤)

#### 本地实现位置
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/stage04_filter/
├── apply_table6.py         # 论文Table 6阈值
├── camera.py               # 相机过滤
├── visual_metrics.py       # VMAFF/DOVER/UniMatch
├── vlm_entity_quality.py   # Qwen3.5 VLM过滤
└── scene_cut.py            # 场景切换检测
```

#### 官方实现位置
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-wm-data-clean/sana_wm_data/filter/
├── __init__.py
└── camera.py               # 相机过滤
```

#### 核心功能

**论文Appendix B.3: Per-Dataset Quality Filter Thresholds (Table 6)**

过滤维度:
1. **VMAF Motion**: [0.5, 100]
2. **UniMatch**: [3, 100] (optical flow)
3. **DOVER**: [0.25, 1.0] (technical/aesthetic quality)
4. **Color Saturation**: [0, 180]
5. **Scene Cuts**: ≤1
6. **VLM Entity**: [0, 25] (people, vehicles, animals)
7. **VLM Quality**: [0.5, 1.5]
8. **Camera FOV**: [25°, 120°]
9. **Focal length ratio**: |fx-fy| / mean < 0.20
10. **Scale CoV**: < 2.0

#### 冒烟测试覆盖

❌ **未覆盖** - 冒烟测试不执行过滤步骤

#### 输入输出

| 维度 | 说明 |
|------|------|
| **输入** | 带pose的视频clips |
| **处理** | 计算质量指标 → Table 6阈值过滤 |
| **输出** | 通过过滤的clips (保留约70-80%) |

#### 对齐状态

⚠️ **部分对齐**
- 官方仓库只有camera.py (相机过滤)
- 本地实现覆盖论文Table 6的所有指标
- 无法验证visual_metrics/VLM部分的准确性

**建议**: 
- 对照论文Table 6验证阈值 ✅
- 独立测试VLM过滤效果

---

### Stage 05: Caption (字幕生成)

#### 本地实现位置
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/stage05_caption/
├── qwen35_vl_runner.py     # Qwen3.5-VL推理
├── prompts.py              # Caption提示词
└── postprocess.py          # 后处理
```

#### 官方实现位置

❌ **官方仓库无对应模块**

**论文未详述caption生成流程**

#### 核心功能

从代码推断:

1. **模型**: Qwen3.5-VL (论文Appendix B.3提到用于过滤)
2. **提示词**: 静态场景描述
3. **后处理**: 清理、格式化

#### 冒烟测试覆盖

❌ **未覆盖** - 冒烟测试使用固定caption:
```python
cap = f"A video from SpatialVID-hq dataset with {len(poses)} frames."
```

#### 输入输出

| 维度 | 说明 |
|------|------|
| **输入** | 视频clips |
| **处理** | Qwen3.5-VL生成描述 |
| **输出** | caption.txt |

#### 对齐状态

⚠️ **本地独有实现**
- 官方仓库无对应代码
- 论文未详述
- 无法验证对齐

**建议**: 
- 如果caption质量影响训练 → 需要验证
- 如果只是元数据 → 影响较小

---

### Stage 06: Pack (WebDataset打包)

#### 本地实现位置
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/stage06_pack/
├── webdataset_writer.py    # WDS写入器
└── schema.py               # 数据格式
```

#### 官方实现位置

❌ **官方仓库无对应模块**

**论文未详述打包格式**

#### 核心功能

**WebDataset格式** (.tar):
```
{scene_id}.mp4              # 视频
{scene_id}.poses_c2w.npy    # 相机位姿 (N,4,4)
{scene_id}.intrinsics.npy   # 内参 (N,4)
{scene_id}.scale.npy        # Scale (N,)
{scene_id}.caption.txt      # 描述
{scene_id}.meta.json        # 元数据
```

#### 冒烟测试覆盖

⚠️ **简化版覆盖** (`smoke_test_batch.py` 第25-72行):
```python
def pack_shard(...):
    """打包WebDataset shard (简化版)"""
    # 包含: mp4, poses_c2w.npy, intrinsics.npy, scale.npy, caption.txt, meta.json
```

冒烟测试使用内联版本，未测试stage06的完整实现。

#### 输入输出

| 维度 | 说明 |
|------|------|
| **输入** | 过滤后的clips + poses + captions |
| **处理** | 打包为WebDataset .tar shards |
| **输出** | {dataset}-{shard_id}.tar (训练就绪) |

#### 对齐状态

⚠️ **本地独有实现**
- 官方仓库无对应代码
- WebDataset是标准格式
- Schema需要与训练代码匹配

**建议**: 
- 验证训练代码能正确读取
- 确认字段名称、数据类型一致

---

## 🔍 关键发现

### 1. 官方实现覆盖有限

官方`sana-wm-data-clean`仓库只包含:
- ✅ `ingest/` (部分)
- ✅ `pose/` (完整)
- ✅ `filter/` (部分，只有camera.py)

**缺失**:
- ❌ 3DGS augmentation完整实现
- ❌ Caption生成流程
- ❌ WebDataset打包逻辑

**可能原因**:
1. 官方仓库是"核心算法"仓库，不包含完整pipeline
2. 这些模块可能在其他内部仓库
3. 论文未完全开源data pipeline

### 2. 冒烟测试覆盖不足

**已覆盖**:
- ✅ Stage 01: normalize_video
- ✅ Stage 02: run_default (default模式)

**未覆盖**:
- ❌ Stage 02: GT-depth, GT-pose模式
- ❌ Stage 03: 3DGS augmentation
- ❌ Stage 04: Quality filtering
- ❌ Stage 05: Caption generation
- ❌ Stage 06: WebDataset packing (仅简化版)

**覆盖率**: 33% (2/6 完整)

### 3. 本地实现的独立模块

**Stage 03, 05, 06** 是本地独有实现:
- 论文有提及（Stage 03）或使用（Stage 05的Qwen3.5）
- 但官方仓库无代码
- 无法验证对齐

### 4. Stage 02已验证对齐

**Default模式**:
- ✅ 代码逻辑100%对齐官方实现
- ✅ 深度融合算法一致
- ⚠️ Metric scale偏差来自MoGe-2模型本身（不是bug）

**GT模式**:
- ✅ GT-pose: Umeyama算法100%对齐
- ✅ GT-depth: 深度融合逐行一致

---

## 📊 对齐状态矩阵

| Stage | 官方实现 | 本地实现 | 冒烟测试 | 对齐状态 | 优先级 |
|-------|---------|---------|---------|---------|--------|
| **Stage 01** | 部分 | ✅ | ✅ | ⚠️ 逻辑对齐论文 | 🟢 低 |
| **Stage 02** | ✅ 完整 | ✅ | ✅ Default | ✅ 100%对齐 | 🟢 已验证 |
| **Stage 03** | ❌ 无 | ✅ | ❌ | ⚠️ 本地独有 | 🟡 中 |
| **Stage 04** | 部分 | ✅ | ❌ | ⚠️ 部分对齐 | 🟡 中 |
| **Stage 05** | ❌ 无 | ✅ | ❌ | ⚠️ 本地独有 | 🟠 低-中 |
| **Stage 06** | ❌ 无 | ✅ | ⚠️ 简化 | ⚠️ 本地独有 | 🟠 低-中 |

---

## 🎯 建议与行动项

### 立即行动 (P0)

1. **✅ Stage 02已验证** - Default模式100%对齐，GT模式100%对齐
   - Default的metric scale偏差来自MoGe-2 ✅
   - 需要验证MoGe-2本身的准确性

### 短期行动 (P1)

2. **扩展冒烟测试覆盖** ⚠️
   ```bash
   # 当前覆盖: Stage 01 + Stage 02 (default)
   # 建议添加: Stage 04 filter + Stage 05 caption
   ```

3. **验证Stage 04过滤阈值** ⚠️
   - 对照论文Table 6检查所有阈值
   - 测试VMAFF/DOVER/UniMatch计算
   - 验证VLM过滤效果

### 中期行动 (P2)

4. **Stage 03 (3DGS Aug)** ⚠️
   - 如果需要DL3DV数据 → 独立验证FCGS流程
   - 如果只用真实视频 → 可以跳过

5. **Stage 05 (Caption)** ⚠️
   - 验证Qwen3.5-VL生成的caption质量
   - 如果caption影响训练 → 需要优化

6. **Stage 06 (Pack)** ⚠️
   - 验证训练代码能正确读取WebDataset
   - 确认字段schema与训练匹配

### 低优先级 (P3)

7. **联系论文作者** 💬
   - 询问是否有完整的data pipeline代码
   - 确认Stage 03/05/06的官方实现位置

---

## 🔬 Ponytail原则应用

### 什么可以信任？

✅ **可以信任**:
- Stage 02 pose estimation (已验证100%对齐)
- Stage 01归一化逻辑 (符合论文§5.1)
- Stage 04的camera过滤 (官方有对应代码)

⚠️ **需要验证**:
- Stage 04的visual/VLM过滤 (官方无代码)
- Stage 03/05/06 (本地独有，无官方参考)
- MoGe-2的metric depth准确性 (根本问题)

### 如何处理无官方实现的模块？

**Ponytail策略**:
1. 对照论文描述验证逻辑 ✅
2. 端到端测试输出质量 ✅
3. 如果影响训练效果 → 独立验证
4. 如果只是工程实现 → 接受现状

---

## 📝 总结

### 核心结论

1. **Stage 02 (Pose)** ✅ **100%对齐官方实现**
   - Default/GT-pose/GT-depth三种模式全部对齐
   - Metric scale偏差来自MoGe-2模型本身

2. **Stage 01 (Ingest)** ⚠️ **逻辑对齐论文**
   - 归一化到720p@16fps符合论文
   - 官方仓库无完整对应代码

3. **Stage 03/04/05/06** ⚠️ **部分或无官方参考**
   - 官方仓库覆盖有限
   - 需要独立验证或接受现状

4. **冒烟测试** ⚠️ **覆盖不足 (33%)**
   - 只测试Stage 01 + Stage 02 default
   - 建议扩展覆盖Stage 04/05

### 下一步

**最高优先级**: 验证MoGe-2的metric depth准确性
- 这是Default模式scale偏差的根源
- 影响所有使用default模式的数据

**次优先级**: 扩展冒烟测试 + 验证过滤阈值
- 确保完整pipeline的质量控制

---

**报告完成**: 2026-08-15  
**审核状态**: 待用户审核
