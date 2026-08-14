# SANA-WM Camera 架构深度解析

> **关键发现**：官方代码存在**双后端架构**，容易造成混淆。`adapters.run_vipe_slam()` 在真实模式下**并不运行真实 VIPE SLAM**，而是使用 Pi3 pose 作为代理。

---

## 一、核心架构：两套后端

### 1.1 Reference Backend（轻量级）

**入口**：`stage.annotate_pose()` → `adapters.run_vipe_slam()`

**关键代码**（`sana_wm_data/pose/adapters.py:122-141`）：

```python
def run_vipe_slam(
    video_path: str, clip_id: str, n_frames: int, hw, depth, intrinsics0, cfg, dry_run: bool
):
    """VIPE SLAM front-end + per-frame-intrinsics BA.

    Returns ``(poses (N,4,4), intrinsics (N,V,4))``. In dry-run, returns a
    synthetic trajectory and the seed intrinsics unchanged.
    """
    if dry_run:
        poses = _synthetic_trajectory(clip_id, n_frames)
        return poses, intrinsics0
    # Real mode: full VIPE SLAM+BA is a heavy CUDA build we do not set up here.
    # We use Pi3's multi-frame-consistent pose output as the pose track (Pi3 is
    # the structure backbone SANA-WM builds on); VIPE's bundle-adjustment refine
    # is the one layer omitted. Intrinsics keep the seed (per-frame BA not run).
    from . import _real
    frames = read_frames(video_path, n_frames)
    poses, _depth = _real.pi3_infer(frames)
    return poses, intrinsics0  # ← 融合深度 fused 参数完全被忽略！
```

**行为特征**：
- ✅ Pose 来源：直接使用 **Pi3 输出**
- ❌ 融合深度：**传入但完全被忽略**（参数 `depth` 未使用）
- ❌ VIPE SLAM：**不运行**
- ❌ 内参 BA：**不运行**（返回 seed intrinsics）
- ✅ 融合深度的唯一作用：计算 `scales` 用于后续 `scale_cov` 质量过滤

### 1.2 VIPE CLI Backend（高保真）

**入口**：`vipe_cli.annotate_pose_vipe_cli()`

**关键代码**（`sana_wm_data/pose/vipe_cli.py:139-155`）：

```python
# 1. Pi3X+MoGe-2 融合深度（torch-2.8 real env）
subprocess.run(
    [sys.executable, f"{cfg['wm_root']}/scripts/precompute_fused_depth.py", 
     video, str(depth_dir)],
    check=True, env=env,
)  # → 输出 depth_dir/{fused.npy, sig.npy, scales.npy}

# 2. 真实 VIPE SLAM + 逐帧内参 BA（vipe venv）
vipe_env = dict(
    env,
    SANA_WM_FUSED_DEPTH_DIR=str(depth_dir),  # ← 融合深度通过环境变量传入
    SANA_WM_PF_DUMP=str(pf_dump),            # ← BA 优化的内参输出路径
    PYTHONPATH=f"{cfg['wm_root']}",
)
subprocess.run(
    [cfg["vipe_bin"], "infer", video, "-o", str(vipe_out), "-p", "sanawm"],
    check=True, env=vipe_env,
)  # → VIPE 通过补丁读取 fused.npy，运行 SLAM+BA
```

**行为特征**：
- ✅ Pose 来源：**真实 VIPE SLAM** 输出
- ✅ 融合深度：**真实使用**（VIPE 通过 `pi3x_moge_depth.py` 补丁读取）
- ✅ VIPE SLAM：**真实运行**（subprocess 调用 `.venv-vipe/bin/vipe`）
- ✅ 内参 BA：**真实运行**（逐帧优化 `(N,V,4)` 张量，输出 `intr_pf.npy`）
- ✅ 融合深度的作用：既用于 VIPE SLAM 输入，也用于质量过滤

---

## 二、后端选择逻辑

### 2.1 触发条件（`camera_cli.py:181-189`）

```python
use_vipe = args.backend == "vipe" or (
    args.backend == "auto" and args.mode == "default" and not args.dry_run
)
if use_vipe:
    annotate_pose_vipe_cli(record, output, models)  # ← VIPE CLI Backend
    backend = "vipe_cli"
else:
    annotate_pose(record, output, models)           # ← Reference Backend
    backend = "reference"
```

### 2.2 决策表

| 条件 | 后端 | Pose 来源 | 融合深度使用 | 内参 BA |
|------|------|----------|------------|---------|
| `--backend vipe` | **VIPE CLI** | 真实 VIPE SLAM | ✅ 使用 | ✅ 运行 |
| `--backend reference` | **Reference** | Pi3 代理 | ❌ 忽略 | ❌ 不运行 |
| `--backend auto` + `--mode default` + 非 dry-run | **VIPE CLI** | 真实 VIPE SLAM | ✅ 使用 | ✅ 运行 |
| `--backend auto` + `--mode gt_pose` | **Reference** | GT 轨迹 | ❌ 忽略 | ❌ 不运行 |
| `--backend auto` + `--dry-run` | **Reference** | 合成轨迹 | ❌ 忽略 | ❌ 不运行 |

---

## 三、三种模式的详细流程

### 3.1 Mode: `default`（互联网视频）

#### Reference Backend 流程

```python
stage.annotate_pose(mode="default", backend="reference")
  ├─ adapters.run_pi3x_depth()       # Pi3 推理 → (N,h,w) 深度（尺度模糊）
  ├─ adapters.run_moge2_depth()      # MoGe-2 推理 → (N,h,w) 度量深度
  ├─ fusion.fuse_depth_sequence()    # 融合 → fused 深度 + scales
  └─ adapters.run_vipe_slam()        # ⚠️ 实际返回 Pi3 pose，fused 被忽略
      └─ _real.pi3_infer()            # 再次调用 Pi3（重复推理！）
```

**问题**：
- Pi3 被调用**两次**（`run_pi3x_depth()` + `run_vipe_slam()` 内部）
- 融合深度计算了但**未被使用**
- Scales 仅用于后续 `scale_cov` 质量过滤

#### VIPE CLI Backend 流程

```python
vipe_cli.annotate_pose_vipe_cli(mode="default")
  ├─ subprocess: precompute_fused_depth.py
  │   ├─ _real.pi3_infer()           # Pi3 推理 → poses + depth
  │   ├─ _real.moge_metric_depth()   # MoGe-2 推理 → 度量深度
  │   ├─ fusion.fuse_depth_sequence()# 融合 → fused.npy + scales.npy
  │   └─ 保存 16x16 RGB 签名 → sig.npy
  │
  └─ subprocess: vipe infer -p sanawm
      ├─ 通过 pi3x_moge_depth.py 补丁读取 fused.npy
      ├─ VIPE SLAM（使用融合深度）
      ├─ 逐帧内参 BA → intr_pf.npy
      └─ 输出 pose/*.npz
```

**优势**：
- Pi3 仅调用**一次**
- 融合深度**真实用于 VIPE SLAM**
- 逐帧内参 BA **真实运行**

### 3.2 Mode: `gt_pose`（GT 轨迹源，如 Sekai-Game/DL3DV）

**仅支持 Reference Backend**

```python
stage.annotate_pose(mode="gt_pose")
  ├─ adapters.run_pi3x_trajectory()  # Pi3 预测相机位置 (n_scale,3)
  ├─ _load_gt_poses()                # 加载 GT 位姿 (N,4,4)，N=全帧数
  ├─ alignment.recover_metric_scale()# Umeyama Sim(3) 对齐
  │   ├─ umeyama_sim3()              # 初步对齐 → s, R, t
  │   ├─ 计算残差 → 80% 分位数阈值  # 鲁棒性过滤
  │   └─ 内点重拟合 → 最终尺度 s
  └─ poses = GT 轨迹（原始，不修改）
```

**特点**：
- GT 轨迹保持**全长**（N 帧）
- Pi3 仅用于恢复**单一度量尺度标量** `s`
- 不运行 VIPE SLAM（GT 已提供轨迹）

### 3.3 Mode: `gt_depth`（GT 深度源，如 OmniWorld）

**设计上支持，但实际未实现**

```python
vipe_cli.annotate_pose_vipe_cli(mode="gt_depth")
  └─ raise NotImplementedError(
         "gt_depth needs GT depth maps; OmniWorld-Game ships GT poses only -> use gt_pose"
     )
```

**原因**（`vipe_cli.py:112-123` 注释）：
- OmniWorld-Game 发布版仅提供 GT poses + intrinsics
- **不提供 GT 深度图**
- 实际应使用 `gt_pose` 模式

---

## 四、关键数据流对比

### 4.1 Reference Backend（default 模式）

```
视频
 ├─ Pi3 推理 #1 → depth_pi3x ──┐
 │                              ├─→ fuse_depth_sequence() → fused + scales
 ├─ MoGe-2 推理 → depth_moge ───┘                               │
 │                                                              │
 └─ Pi3 推理 #2 → poses ────────────────────────────────────→ 输出
                                                                │
    seed_intrinsics ────────────────────────────────────────→ 输出
                                                                │
    scales ──────────────────────────────────────────────────→ 质量过滤
```

**问题**：
- ❌ Pi3 重复推理（浪费计算）
- ❌ `fused` 深度计算后被丢弃
- ❌ 无 VIPE SLAM 精化
- ❌ 无内参 BA

### 4.2 VIPE CLI Backend（default 模式）

```
视频
 ├─ Pi3 推理 → depth_pi3x + poses_pi3 ──┐
 │                                       ├─→ fuse_depth → fused.npy + scales.npy
 └─ MoGe-2 推理 → depth_moge ────────────┘                    │
                                                              │
                                            16x16 RGB 签名 → sig.npy
                                                              │
    [进程边界：torch-2.8 → vipe venv]                          │
                                                              │
    VIPE SLAM ← fused.npy + sig.npy ──────────────────────────┘
      ├─ 读取融合深度（通过签名匹配 keyframe）
      ├─ SLAM 前端
      ├─ 逐帧内参 BA → intr_pf.npy (K,4)
      └─ 输出 poses (N,4,4)
```

**优势**：
- ✅ Pi3 仅推理一次
- ✅ 融合深度真实使用
- ✅ VIPE SLAM 精化轨迹
- ✅ 逐帧内参 BA 捕获焦距变化

---

## 五、关键代码证据

### 5.1 融合深度被忽略的证据

**`adapters.py:122`** 函数签名：

```python
def run_vipe_slam(
    video_path: str, clip_id: str, n_frames: int, hw, 
    depth,        # ← 融合深度参数
    intrinsics0, cfg, dry_run: bool
):
```

**`adapters.py:137-141`** 真实模式实现：

```python
# Real mode: full VIPE SLAM+BA is a heavy CUDA build we do not set up here.
from . import _real
frames = read_frames(video_path, n_frames)
poses, _depth = _real.pi3_infer(frames)  # ← 重新推理 Pi3
return poses, intrinsics0                # ← 参数 depth 从未被使用！
```

### 5.2 VIPE CLI 真实使用融合深度的证据

**`precompute_fused_depth.py:39-59`**：

```python
_poses, pi3_depth = _real.pi3_infer(frames)              # (S,h,w)
moge_depth = _real.moge_metric_depth(frames, ref_hw=pi3_depth.shape[1:])
fused, scales = fuse_depth_sequence(pi3_depth, np.abs(moge_depth), ema_momentum=0.99)

np.save(out / "fused.npy", fused.astype(np.float32))   # ← 保存融合深度
np.save(out / "sig.npy", sig)                          # ← 保存 RGB 签名
np.save(out / "scales.npy", np.asarray(scales, dtype=np.float32))
```

**`vipe_cli.py:147-148`**：

```python
vipe_env = dict(
    env,
    SANA_WM_FUSED_DEPTH_DIR=str(depth_dir),  # ← 通过环境变量传给 VIPE
    SANA_WM_PF_DUMP=str(pf_dump),
)
```

**`vipe_patches/pi3x_moge_depth.py`**（VIPE 补丁，读取融合深度）：

```python
# 补丁代码从 SANA_WM_FUSED_DEPTH_DIR 读取 fused.npy
depth_dir = os.environ.get("SANA_WM_FUSED_DEPTH_DIR")
fused = np.load(f"{depth_dir}/fused.npy")
sig = np.load(f"{depth_dir}/sig.npy")
# 通过 RGB 签名匹配 keyframe → 对应的融合深度
```

---

## 六、使用建议

### 6.1 论文复现 / 高保真结果

```bash
# 方式 1：显式指定 VIPE 后端
sana-wm-camera video.mp4 --out /tmp/out --backend vipe

# 方式 2：使用 auto（default 模式自动选择 VIPE）
sana-wm-camera video.mp4 --out /tmp/out --backend auto  # 默认值
```

**特点**：
- ✅ 真实 VIPE SLAM
- ✅ 逐帧内参 BA
- ✅ 融合深度用于 SLAM
- ⚠️ 需要安装 VIPE 环境（`setup_vipe.sh` + `apply_vipe_patches.sh`）

### 6.2 快速验证 / 轻量推理

```bash
sana-wm-camera video.mp4 --out /tmp/out --backend reference
```

**特点**：
- ✅ 快速（仅 Pi3 + MoGe-2）
- ✅ 无需 VIPE 环境
- ❌ 无 SLAM 精化
- ❌ 内参为 seed（固定值）
- ⚠️ 结果质量低于 VIPE 后端

### 6.3 GT 轨迹源（Sekai-Game/DL3DV）

```bash
sana-wm-camera video.mp4 \
  --mode gt_pose \
  --gt-poses poses.npy \
  --gt-intrinsics intrinsics.npy \
  --out /tmp/out
```

**特点**：
- ✅ 自动使用 Reference Backend（无需 VIPE）
- ✅ GT 轨迹保持原始
- ✅ Pi3 仅用于恢复度量尺度

---

## 七、常见误区

### 误区 1：`adapters.run_vipe_slam()` 运行真实 VIPE

**错误理解**：
```python
fused, scales = fuse_depth_sequence(pi3x, moge)
poses, intr = adapters.run_vipe_slam(..., fused, ...)  # 以为会用 fused
```

**实际行为**：
- Reference Backend：`fused` 被忽略，返回 Pi3 原始 pose
- 真实 VIPE 在 `vipe_cli.annotate_pose_vipe_cli()` 中

### 误区 2：融合深度没有用

**部分正确**：
- Reference Backend：融合深度仅用于计算 `scales`（质量过滤）
- VIPE CLI Backend：融合深度**真实用于 VIPE SLAM**

### 误区 3：default 模式不运行 VIPE

**错误**：
- `--backend auto` + `--mode default` → **自动选择 VIPE CLI Backend**
- 只有显式 `--backend reference` 才不运行 VIPE

---

## 八、代码质量评价

### 优点

1. **模块化设计**：Reference 和 VIPE CLI 后端分离
2. **渐进式复杂度**：可从 Reference 快速验证 → VIPE 高保真
3. **进程隔离**：VIPE 运行在独立 venv，CUDA 故障不污染
4. **文档完整**：注释明确说明 Reference Backend 的局限性

### 潜在混淆点

1. **命名误导**：`adapters.run_vipe_slam()` 命名暗示运行 VIPE，但实际是 Pi3 代理
2. **参数浪费**：`depth` 参数传入但未使用，容易误导调用者
3. **重复推理**：Reference Backend 中 Pi3 被调用两次
4. **文档不足**：README 未明确说明两套后端的差异

### 建议改进

```python
# 更清晰的命名
def run_pi3_pose_proxy(...):  # 替代 run_vipe_slam
    """Pi3 pose proxy (VIPE SLAM not run). Use vipe_cli backend for real VIPE."""
    ...

# 移除未使用的参数
def run_pi3_pose_proxy(
    video_path: str, clip_id: str, n_frames: int, 
    # depth,  # ← 移除，避免误导
    intrinsics0, cfg, dry_run: bool
):
```

---

## 九、总结

| 维度 | Reference Backend | VIPE CLI Backend |
|------|------------------|------------------|
| **Pose 来源** | Pi3 直接输出 | 真实 VIPE SLAM |
| **融合深度** | 计算但忽略 | 真实使用 |
| **内参 BA** | 不运行 | 真实运行 |
| **Pi3 调用次数** | 2 次（冗余） | 1 次 |
| **环境要求** | Pi3 + MoGe-2 | Pi3 + MoGe-2 + VIPE |
| **适用场景** | 快速验证 | 论文复现 |
| **默认触发** | gt_pose / dry-run | default 模式 |

**关键记忆点**：
- ⚠️ `adapters.run_vipe_slam()` 在真实模式下**不运行 VIPE**
- ✅ 真实 VIPE 在 `vipe_cli.annotate_pose_vipe_cli()` 中
- 🎯 `--backend auto` + `default` 模式 → 自动使用 VIPE CLI Backend

---

**文档版本**：v1.0  
**创建日期**：2026-08-14  
**适用代码版本**：sana-wm-data-clean (commit 未知，基于当前 workspace 快照)
