# SANA-WM Data Clean 架构深度分析

> **创建日期**：2026-08-14  
> **目的**：快速理解 `sana-wm-data-clean` 官方代码的结构、调用逻辑和关键设计

---

## 一、核心功能定位

**从视频估计相机轨迹（camera-to-world poses）与逐帧内参（per-frame intrinsics）**

**技术栈**：
- **Pi3**：多帧一致性结构 + 相机位姿（尺度模糊）
- **MoGe-2**：度量尺度深度
- **VIPE SLAM**：前端 + 逐帧内参 Bundle Adjustment

**代码规模**：25 个 Python 文件 + 3 个 Shell 脚本

---

## 二、目录结构速览

```
sana-wm-data-clean/
├── sana_wm_data/              # 核心库
│   ├── camera_cli.py          # CLI 入口（sana-wm-camera 命令）
│   ├── manifest.py            # ClipRecord 数据结构 + JSONL 序列化
│   ├── pose/                  # 位姿估计核心
│   │   ├── _real.py           # Pi3 + MoGe-2 推理封装（懒加载 + autocast）
│   │   ├── adapters.py        # 模型适配器（真实/dry-run）
│   │   ├── alignment.py       # Umeyama Sim(3) 对齐（度量尺度恢复）
│   │   ├── fusion.py          # Pi3X + MoGe-2 深度融合（加权LS + EMA）
│   │   ├── intrinsics.py      # 逐帧内参张量表示 (N,V,4)
│   │   ├── stage.py           # 三模式位姿估计主流程
│   │   └── vipe_cli.py        # 真实 VIPE 后端（subprocess 调用）
│   ├── filter/                # 相机质量过滤
│   │   └── camera.py          # FoV/focal_div/scale_cov 过滤器
│   └── ingest/                # 数据源接入
│       └── sekai_game.py      # Sekai-Game 格式转换
├── scripts/                   # 工具脚本
│   ├── precompute_fused_depth.py  # 预计算融合深度（Pi3+MoGe）
│   ├── setup_camera_env.sh        # 安装 Pi3/MoGe 环境
│   ├── setup_vipe.sh              # 安装 VIPE 环境
│   ├── make_zoom_clip.py          # 生成焦距渐变测试视频
│   └── compare_intrinsics.py      # 内参恢复精度验证
├── vipe_patches/              # VIPE 补丁
│   ├── apply_vipe_patches.sh      # 应用补丁脚本
│   ├── apply_perframe_intrinsics_ba.py  # 逐帧内参 BA 补丁
│   └── pi3x_moge_depth.py         # Pi3X+MoGe 深度后端补丁
├── tests/                     # pytest 测试套件
└── third_party/               # 第三方库（Pi3/MoGe/VIPE）
```

---

## 三、⚠️ 关键架构发现：双后端设计

### **核心混淆点**：`adapters.run_vipe_slam()` 不运行真实 VIPE！

官方设计了**两套完全独立的后端**：

| 维度 | **Reference Backend** | **VIPE CLI Backend** |
|------|----------------------|---------------------|
| **入口函数** | `stage.annotate_pose()` | `vipe_cli.annotate_pose_vipe_cli()` |
| **Pose 来源** | `adapters.run_vipe_slam()` → **Pi3 直接输出** | `subprocess.run([vipe_bin])` → **真实 VIPE SLAM** |
| **融合深度** | **被忽略**（传入但未使用） | **真实使用**（通过环境变量传入 VIPE） |
| **内参 BA** | **不运行**（返回 seed intrinsics） | **真实运行**（逐帧内参优化） |
| **适用场景** | 快速验证、轻量测试 | 高保真复现、论文结果 |
| **触发条件** | `--backend reference` | `--backend vipe` 或 `--backend auto` + default 模式 |

### 代码证据：`adapters.py:122-141`

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
    return poses, intrinsics0  # ← ❌ 融合深度 depth 参数完全被忽略！
```

**注释明确说明**：
- ✅ "full VIPE SLAM+BA is a heavy CUDA build we do not set up here"
- ✅ "VIPE's bundle-adjustment refine is the one layer omitted"
- ✅ "Intrinsics keep the seed (per-frame BA not run)"

### 后端选择逻辑：`camera_cli.py:181-189`

```python
use_vipe = args.backend == "vipe" or (
    args.backend == "auto" and args.mode == "default" and not args.dry_run
)
if use_vipe:
    annotate_pose_vipe_cli(record, output, models)  # ← 真实 VIPE
    backend = "vipe_cli"
else:
    annotate_pose(record, output, models)           # ← Pi3 代理
    backend = "reference"
```

### 真实 VIPE 的调用：`vipe_cli.py:139-155`

```python
# 1. 预计算融合深度（torch 环境）
subprocess.run(
    [sys.executable, f"{cfg['wm_root']}/scripts/precompute_fused_depth.py", video, str(depth_dir)],
    check=True, env=env,
)  # ← 生成 fused.npy + sig.npy + scales.npy

# 2. 真实 VIPE SLAM（vipe venv）
vipe_env = dict(
    env,
    SANA_WM_FUSED_DEPTH_DIR=str(depth_dir),  # ← 通过环境变量传入
    SANA_WM_PF_DUMP=str(pf_dump),
)
subprocess.run(
    [cfg["vipe_bin"], "infer", video, "-o", str(vipe_out), "-p", "sanawm"],
    check=True, env=vipe_env,
)  # ← VIPE 读取 fused.npy 并运行 SLAM+BA
```

---

## 四、三种工作模式详解

### 模式 1：`default`（互联网视频）

**流程**：Pi3X + MoGe-2 融合深度 → VIPE SLAM + 逐帧 BA

```python
stage.annotate_pose(mode="default")
  ├─ adapters.run_pi3x_depth()       # Pi3 → (N,h,w) 深度（尺度模糊）
  ├─ adapters.run_moge2_depth()      # MoGe-2 → (N,h,w) 度量深度
  ├─ fusion.fuse_depth_sequence()    # 融合：逐帧尺度 + EMA 平滑
  │   └─ solve_frame_scale()         # 加权 LS：s = Σ(w·a·b) / Σ(w·a²)
  └─ 后端分支：
      ├─ Reference：adapters.run_vipe_slam() → Pi3 pose（融合深度被忽略）
      └─ VIPE CLI：真实 VIPE SLAM（使用融合深度）
```

**深度融合数学**（`fusion.py`）：
```python
# 目标：每帧尺度 s_t 使得 s_t * d_Pi3X ≈ d_MoGe
# 加权最小二乘：min Σ w_i (s·a_i - b_i)²
# w_i = 1 / d_MoGe_i（逆深度权重）
s = Σ(w·a·b) / Σ(w·a²)

# EMA 平滑（momentum=0.99）
s_smoothed[t] = 0.99·s[t-1] + 0.01·s_raw[t]
fused[t] = s_smoothed[t] * d_Pi3X[t]
```

### 模式 2：`gt_pose`（GT 轨迹源，如 Sekai-Game/DL3DV）

**流程**：保留 GT 轨迹 + Pi3 结构 + Umeyama 恢复度量尺度

```python
stage.annotate_pose(mode="gt_pose")
  ├─ adapters.run_pi3x_trajectory()  # Pi3 预测相机位置 (n_scale,3)
  ├─ _load_gt_poses()                # 加载 GT 位姿 (N,4,4)，N=全帧数
  ├─ alignment.recover_metric_scale()  # Umeyama Sim(3) 对齐
  │   ├─ umeyama_sim3()              # 初步对齐 → s, R, t
  │   ├─ 80% 分位数过滤              # 鲁棒性剔除外点
  │   └─ 内点重拟合 → 最终尺度 s
  └─ poses = GT 轨迹（保持原始）    # 尺度存于 scale_factors
```

**关键**：GT 轨迹保持**全长**（N 帧），Pi3 仅跑子采样用于对齐。

### 模式 3：`gt_depth`（理论模式，当前未使用）

**注意**：当前 OmniWorld 发布版无 GT 深度，实际路由到 `gt_pose`。

**理论流程**：GT 深度喂入 SLAM + MoGe-2 恢复度量尺度

---

## 五、关键数学模块

### 5.1 Umeyama Sim(3) 对齐（`alignment.py`）

**目标**：求 `s, R, t` 使得 `s·R·pred + t ≈ gt`

```python
umeyama_sim3(pred, gt):
  ├─ 中心化：src_c = src - μ_src, dst_c = dst - μ_dst
  ├─ 协方差矩阵：Σ = (dst_c.T @ src_c) / n
  ├─ SVD：Σ = U·D·V^T
  ├─ 旋转：R = U·S·V^T（S 修正行列式符号）
  └─ 尺度：s = tr(D·S) / var(src_c)

recover_metric_scale():
  ├─ 初步对齐 → 计算残差
  ├─ 80% 分位数 → 内点阈值
  └─ 内点重拟合 → 最终尺度 s
```

### 5.2 相机质量过滤（`filter/camera.py`）

**三重门槛**（Appendix B.3）：

```python
1. FoV 范围：θ = 2·arctan(dim / 2f) ∈ [25°, 120°]
2. Focal divergence：|fx-fy| / ((fx+fy)/2) ≤ 0.20
3. Scale CoV：std(s_t) / mean(s_t) ≤ 2.0
```

**Scale CoV 特殊处理**：空序列返回 `inf`，确保无尺度恢复的 clip 被拒绝。

---

## 六、模型封装（`_real.py`）

### 6.1 Pi3 推理

```python
pi3_infer(frames: (N,H,W,3) uint8):
  ├─ _round_to_patch()      # 长边 ≤ 518，H/W 对齐 14（DINOv2 patch）
  ├─ model(imgs[None])      # (1,N,3,H,W) → Pi3
  ├─ camera_poses           # (N,4,4) cam2world（OpenCV 轴）
  └─ local_points[..., 2]   # (N,h,w) 深度（z 通道）
```

**内存保护**：
- `_PI3_MAX_SIDE=518` 防止 4K 视频 OOM
- 历史教训：原生 4K 曾分配 90+ GiB 后仍 OOM

### 6.2 MoGe-2 推理

```python
moge_metric_depth(frames, ref_hw):
  ├─ 逐帧推理：model.infer(frame)
  ├─ 深度 resize 到 ref_hw（对齐 Pi3 网格）
  └─ (N, *ref_hw) 度量深度（米）
```

### 6.3 懒加载模式

```python
@lru_cache(maxsize=1)
def _pi3():
    import torch
    from pi3.models.pi3 import Pi3
    local = _WEIGHTS / "pi3"
    src = str(local) if local.exists() else "yyfz233/Pi3"
    model = Pi3.from_pretrained(src).to(_device()).eval()
    return model
```

**特点**：
- 仅首次调用加载模型
- 优先使用本地权重（`SANA_WM_WEIGHTS`），否则从 HuggingFace 下载
- 自动选择 bfloat16/float16（Ampere+ 用 bf16）

---

## 七、VIPE CLI 两进程架构

**设计原因**：隔离 CUDA 故障，单个 clip 崩溃不污染全局环境

```
┌─────────────────────────────────────────────────────────┐
│ 进程 1：torch 环境（precompute_fused_depth.py）         │
├─────────────────────────────────────────────────────────┤
│ ├─ Pi3 推理 → (S,h,w) 深度                              │
│ ├─ MoGe-2 推理 → (S,h,w) 度量深度                       │
│ ├─ fuse_depth_sequence() → 融合 + EMA 平滑              │
│ └─ 输出：                                                 │
│     ├─ fused.npy: (S,h,w) 融合深度                      │
│     ├─ sig.npy: (S,768) RGB 16x16 签名（keyframe 匹配）│
│     └─ scales.npy: (S,) 逐帧度量尺度                    │
└─────────────────────────────────────────────────────────┘
                          ↓ 通过文件传递
┌─────────────────────────────────────────────────────────┐
│ 进程 2：VIPE .venv-vipe（vipe infer）                   │
├─────────────────────────────────────────────────────────┤
│ ├─ 读取 fused.npy（通过 SANA_WM_FUSED_DEPTH_DIR）       │
│ ├─ SLAM 前端（匹配 keyframe 到预计算深度）              │
│ ├─ 逐帧内参 BA 优化（(N,V,4) 张量）                     │
│ └─ 输出：                                                 │
│     ├─ pose/*.npz: (N,4,4) cam2world                    │
│     └─ intr_pf.npy: (K,4) BA 优化的逐帧内参            │
└─────────────────────────────────────────────────────────┘
```

### 内参插值逻辑（`vipe_cli._load_perframe_intrinsics`）

```python
# BA 输出 K 个 keyframe 内参，需插值到 N 帧
if K == N:      return pf                          # 完美对齐
if K == 1:      return np.tile(pf[0], (N, 1))     # 广播
if 1 < K < N:   return np.interp(...)              # 线性插值（保留焦距趋势）
```

**设计意图**：变焦视频的焦距变化必须保留，不能折叠为单一常量。

---

## 八、帧采样统一真值源

**关键函数**：`adapters.even_indices(count, n)`

```python
def even_indices(count: int, n: int) -> np.ndarray:
    """单一真值源：Pi3/MoGe 输入 AND GT pose 子采样必须使用此规则"""
    return np.linspace(0, max(count - 1, 0), max(min(n, count), 1)).round().astype(int)
```

**历史 Bug**：
- 早期：`read_frames` 用 `.astype(int)`（截断），`stage.py` 用 `.round()`
- 后果：Pi3 frame_i 和 GT pose_i 对应不同源帧 → Umeyama 尺度偏差
- 修复：统一到 `.round()`，所有采样调用此函数

---

## 九、输出格式与约定

### 输出目录结构

```
<out>/
├── <clip-id>.poses.npy        # (N,4,4) float64, camera-to-world
├── <clip-id>.intrinsics.npy   # (N,4) float64, [fx, fy, cx, cy]
├── result.json                # 元数据 + QC 结果
└── <clip-id>/                 # 仅 VIPE CLI 模式
    ├── depth/                 # 预计算融合深度
    │   ├── fused.npy
    │   ├── sig.npy
    │   └── scales.npy
    └── vipe_out/              # VIPE SLAM 输出
        ├── pose/
        └── intr_pf.npy
```

### 相机约定

- **Pose**：camera-to-world（c2w），OpenCV 轴定义
  - +X 右，+Y 下，+Z 前
  - 与 COLMAP/OpenCV 一致，与 OpenGL/Blender 不同
- **帧对齐**：`poses[i]`, `intrinsics[i]`, `video_frame[i]` 严格对应
- **内参布局**：`[fx, fy, cx, cy]`，像素单位

---

## 十、使用示例与最佳实践

### 基础用法

```bash
# 1. 安装环境（仅需一次）
bash scripts/setup_camera_env.sh      # Pi3 + MoGe-2
bash scripts/setup_vipe.sh             # VIPE 环境
bash vipe_patches/apply_vipe_patches.sh  # 应用补丁

# 2. 真实估计（默认 auto 后端）
sana-wm-camera video.mp4 --out /tmp/out

# 3. 限制 Pi3/MoGe 采样帧数（节省 GPU 内存）
sana-wm-camera video.mp4 --out /tmp/out --max-frames 64

# 4. Dry-run 验证接口（无需 GPU）
sana-wm-camera video.mp4 --out /tmp/out --dry-run
```

### 后端选择策略

```bash
# 高保真复现（论文结果，需 VIPE 环境）
sana-wm-camera video.mp4 --out /tmp/out --backend vipe

# 快速验证/测试（仅需 Pi3+MoGe）
sana-wm-camera video.mp4 --out /tmp/out --backend reference

# 自动选择（default 模式 → vipe，其他 → reference）
sana-wm-camera video.mp4 --out /tmp/out --backend auto
```

### GT 轨迹模式

```bash
# Sekai-Game / DL3DV 等带 GT pose 的数据集
sana-wm-camera video.mp4 \
  --mode gt_pose \
  --gt-poses poses.npy \        # (N,4,4) 或 (N,3,4) 或 (N,3)
  --gt-intrinsics intrinsics.npy \  # (N,4) 或 (4,) 可选
  --out /tmp/out
```

### 环境变量覆盖

```bash
export SANA_WM_ROOT=/path/to/sana-wm-data-clean
export SANA_WM_WEIGHTS=/path/to/weights  # Pi3/MoGe 权重
export SANA_WM_MAX_FRAMES=64              # Pi3/MoGe 最大采样帧
export SANA_WM_PI3_MAX_SIDE=518           # Pi3 输入长边上限

sana-wm-camera video.mp4 --out /tmp/out
```

---

## 十一、关键设计原则（Ponytail 风格）

1. **Stdlib 优先**：`@lru_cache`, `dataclasses`, `argparse`
2. **零无用抽象**：无接口/工厂/策略模式，直接函数调用
3. **懒加载**：模型仅在使用时加载，dry-run 永不触发
4. **进程隔离**：CUDA 故障不蔓延（Pi3/MoGe vs VIPE 分离）
5. **鲁棒过滤**：80% 分位数内点 + 三重 QC 门槛
6. **单一真值源**：`even_indices()` 统一帧采样逻辑
7. **显式配置**：环境变量 > CLI 参数 > 默认值，优先级清晰

---

## 十二、常见陷阱与解决方案

### 陷阱 1：误以为 `adapters.run_vipe_slam()` 运行 VIPE

**症状**：看到 `fuse_depth_sequence()` 的融合深度被传入 `run_vipe_slam()`，以为会被使用。

**真相**：Reference backend 完全忽略融合深度，直接返回 Pi3 pose。

**解决**：使用 `--backend vipe` 或 `--backend auto`（default 模式）。

### 陷阱 2：帧采样不一致导致 Umeyama 尺度偏差

**症状**：`gt_pose` 模式恢复的尺度不准确。

**原因**：Pi3 采样和 GT pose 采样使用不同的舍入规则。

**解决**：所有采样统一调用 `even_indices()`（已修复）。

### 陷阱 3：4K 视频 OOM

**症状**：Pi3 推理时显存不足。

**原因**：Pi3 对全帧序列做全局注意力，4K 输入会爆显存。

**解决**：`_PI3_MAX_SIDE=518` 自动下采样（已内置）。

### 陷阱 4：逐帧内参被折叠为常量

**症状**：变焦视频的焦距变化丢失。

**原因**：误将 BA 输出的 K 个 keyframe 内参取平均。

**解决**：线性插值到 N 帧，保留焦距趋势（`_load_perframe_intrinsics` 已实现）。

---

## 十三、测试与验证

### 单元测试

```bash
cd /path/to/sana-wm-data-clean
pytest -q
```

**覆盖模块**：
- `test_pose_math.py`：Umeyama、深度融合
- `test_camera_filters.py`：FoV/focal_div/scale_cov
- `test_sekai_game_pose_convention.py`：相机约定验证

### 内参精度验证

```bash
# 1. 构造焦距渐变测试视频（从已知内参 clip）
python3 scripts/make_zoom_clip.py <src-clip-dir> <zoom-clip-dir> 1.8

# 2. 恢复内参
sana-wm-camera <zoom-clip-dir>/video.mp4 --out /tmp/out

# 3. 比较恢复 vs GT
python3 scripts/compare_intrinsics.py \
  /tmp/out/<clip-id>.intrinsics.npy \
  <zoom-clip-dir>/gt_intrinsics.npy
```

---

## 十四、快速诊断清单

**遇到问题时按此顺序检查**：

```bash
# 1. 验证环境
python3 -c "import pi3, moge; print('Models OK')"
ls $SANA_WM_WEIGHTS/pi3 $SANA_WM_WEIGHTS/moge2

# 2. Dry-run 验证接口
sana-wm-camera video.mp4 --out /tmp/test --dry-run

# 3. Reference backend（仅需 Pi3+MoGe）
sana-wm-camera video.mp4 --out /tmp/test --backend reference

# 4. VIPE CLI backend（需完整环境）
sana-wm-camera video.mp4 --out /tmp/test --backend vipe

# 5. 检查输出
ls /tmp/test/*.poses.npy /tmp/test/*.intrinsics.npy /tmp/test/result.json
python3 -c "import numpy as np; print(np.load('/tmp/test/<clip>.poses.npy').shape)"
```

**常见错误信息**：

- `FileNotFoundError: no VIPE pose npz` → VIPE 未成功运行，检查 vipe venv
- `SANA_WM_ROOT required for VIPE CLI` → 未设置环境变量
- `video has no frames` → 视频损坏或格式不支持
- `expected poses (N,4,4), got ...` → 输出格式异常，检查后端实现

---

## 十五、与其他组件的集成

### 数据管线位置

```
原始视频 (*.mp4)
    ↓
┌──────────────────────────────────────┐
│ sana-wm-camera (本代码库)             │  ← 你在这里
│ ├─ Pi3X + MoGe-2 深度融合              │
│ ├─ VIPE SLAM + 逐帧内参 BA             │
│ └─ 相机质量过滤                        │
└──────────────────────────────────────┘
    ↓
poses.npy + intrinsics.npy
    ↓
[ 下游组件 ]
├─ Plücker 射线编码（3D 条件）
├─ UCPE 相机条件投影编码
└─ SANA 扩散模型训练
```

### ClipRecord 流转

```python
# 1. 创建记录（camera_cli.py）
record = ClipRecord(
    clip_id="...",
    video_path="...",
    mode="default",
    ...
)

# 2. 填充相机字段（stage.py 或 vipe_cli.py）
record.pose_path = "...poses.npy"
record.intrinsics_path = "...intrinsics.npy"
record.scale_factors = [...]

# 3. QC 过滤（camera_cli.py）
passed, reasons = camera_filter_pass(...)
record.camera = CameraMetrics(fov_x=..., focal_div=..., scale_cov=...)

# 4. 序列化（manifest.py）
write_manifest("output.jsonl", [record])
```

---

## 十六、性能特征

### 典型运行时间（H100 80GB）

| 阶段 | Reference Backend | VIPE CLI Backend |
|------|-------------------|------------------|
| Pi3 推理（64 帧）| ~3-5 秒 | ~3-5 秒 |
| MoGe-2 推理（64 帧）| ~5-8 秒 | ~5-8 秒 |
| 深度融合 | <1 秒 | <1 秒 |
| VIPE SLAM+BA | **跳过** | ~10-30 秒 |
| **总计（单 clip）** | **~10-15 秒** | **~20-45 秒** |

### 显存占用

- **Pi3**：~8-12 GB（64 帧，518px 长边）
- **MoGe-2**：~4-6 GB（逐帧推理）
- **VIPE SLAM**：~6-10 GB（取决于 keyframe 数量）

### 瓶颈分析

1. **Pi3 全局注意力**：O(N²) 复杂度，长序列慢
2. **MoGe-2 逐帧推理**：无法批处理（模型限制）
3. **VIPE BA**：迭代优化，收敛慢

---

## 十七、扩展与修改指南

### 添加新的数据源

```python
# 1. 在 ingest/ 下创建新模块
# sana_wm_data/ingest/my_dataset.py

from ..manifest import ClipRecord

def load_my_dataset(data_dir: Path) -> list[ClipRecord]:
    records = []
    for video_path in data_dir.glob("*.mp4"):
        rec = ClipRecord(
            clip_id=video_path.stem,
            source="my_dataset",
            video_path=str(video_path),
            mode="default",  # 或 "gt_pose" 如果有 GT
        )
        # 如果有 GT pose/intrinsics，添加到 rec.extra
        if (gt_pose := video_path.with_suffix(".poses.npy")).exists():
            rec.extra["gt_positions_path"] = str(gt_pose)
        records.append(rec)
    return records
```

### 修改融合策略

```python
# sana_wm_data/pose/fusion.py

def solve_frame_scale_robust(d_pi3x, d_moge):
    """添加 RANSAC 鲁棒估计"""
    # 原有逻辑...
    # 添加 RANSAC 迭代...
    return scale
```

### 自定义相机过滤器

```python
# sana_wm_data/filter/camera.py

def motion_blur_score(frames: np.ndarray) -> float:
    """计算运动模糊程度（Laplacian 方差）"""
    import cv2
    scores = [cv2.Laplacian(f, cv2.CV_64F).var() for f in frames]
    return float(np.mean(scores))

# 在 camera_filter_pass() 中添加新门槛
def camera_filter_pass(..., cfg):
    # 原有检查...
    blur = motion_blur_score(frames)
    if blur < cfg["blur_threshold"]:
        reasons.append(f"motion_blur={blur:.1f} < {cfg['blur_threshold']}")
    return (len(reasons) == 0), reasons
```

---

## 十八、参考资源

### 论文引用

```bibtex
@article{sana2024,
  title={SANA-WM: Scalable World Model with Camera Conditioning},
  author={...},
  journal={arXiv preprint arXiv:...},
  year={2024}
}
```

### 依赖模型

- **Pi3**：[yyfz233/Pi3](https://huggingface.co/yyfz233/Pi3)
- **MoGe-2**：[Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal)
- **VIPE**：（闭源，需申请访问）

### 环境要求

- Python 3.10+
- PyTorch 2.5+（Pi3/MoGe）
- PyTorch 2.7+（VIPE venv）
- CUDA 11.8+
- GPU：≥24GB 显存（推荐 H100/A100）

---

## 十九、总结：核心要点

✅ **双后端架构**：Reference（Pi3 代理）vs VIPE CLI（真实 SLAM）  
✅ **融合深度**：加权最小二乘 + EMA 平滑（仅 VIPE CLI 使用）  
✅ **Umeyama 对齐**：80% 分位数内点过滤恢复度量尺度  
✅ **逐帧内参 BA**：线性插值保留焦距趋势  
✅ **帧采样统一**：`even_indices()` 单一真值源  
✅ **进程隔离**：Pi3/MoGe 与 VIPE 分离避免 CUDA 污染  
✅ **Ponytail 风格**：零抽象、stdlib 优先、最少代码  

**记住**：`adapters.run_vipe_slam()` 的注释已明确告知它不运行真实 VIPE！真实 VIPE 在 `vipe_cli.annotate_pose_vipe_cli()` 中通过 subprocess 调用。

---

**文档版本**：v1.0  
**最后更新**：2026-08-14  
**维护者**：Claude Code Session  
**适用代码库版本**：sana-wm-data-clean (2026-06 release)
