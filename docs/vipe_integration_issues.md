# VIPE + Pi3X + MoGe-2 集成问题记录

> **实验日期**: 2026-05-28 ~ 2026-05-29  
> **实验目标**: 在 TUM fr1/desk 上对比 VIPE+unidepth-l vs VIPE+MoGe-2 位姿精度  
> **最终状态**: ✅ 实验完成，5次调试后成功运行

---

## 一、VIPE 安装问题

### 问题描述

运行 `pip install -e ".[all]"` 时进程挂起数小时，看似无进展。

### 根本原因

VIPE 的 `pyproject.toml` 将 `torch` 列在 `[build-system].requires` 中。pip 默认开启 **PEP 517 构建隔离**：在安装前，pip 会创建一个干净的隔离沙盒环境，并在沙盒里从零安装所有 build 依赖——包括 `torch`。由于沙盒里没有预编译 wheel 匹配（版本约束），pip 尝试**从源码编译 torch**，耗时数小时至数十小时。

### 解决方案

```bash
pip install --no-user --no-build-isolation -e .
```

`--no-build-isolation` 跳过 PEP 517 隔离，直接复用 conda 环境里已安装的 torch。这样只需编译 VIPE 自己的 CUDA 扩展（`vipe_ext.so`，约 3 分钟）。

### 注意事项

- VIPE 1.1.0 安装后在 `sana_wm` conda env 里永久有效，重启机器后直接 `conda activate sana_wm` 即可用 `vipe` 命令，**无需重跑** `setup_vipe.sh`。
- `setup_vipe.sh` 中的 `pip install -e ".[all]"` 这行会再次触发此问题，不要重新执行它。

---

## 二、Pi3x_moge2 深度后端 API 错误（共 5 个）

### Bug 1: Pi3X 类名错误

```python
# 错误（depth_backend_pi3x_moge2.py 原始版本）
from pi3 import Pi3

# 正确
from pi3 import Pi3X
```

`pi3` 包里的类叫 `Pi3X`，不叫 `Pi3`。触发 `ImportError: cannot import name 'Pi3' from 'pi3'`。

---

### Bug 2: MoGe-2 导入路径错误

```python
# 错误
from moge.model import MoGeModel

# 正确
from moge.model.v2 import MoGeModel
```

安装的 MoGe 2.0 将 v2 模型放在 `moge.model.v2` 子模块，顶层 `moge.model` 没有 `MoGeModel`。

---

### Bug 3: MoGeModel.from_pretrained 不接受目录

```python
# 错误 — 传入目录路径
MoGeModel.from_pretrained("/mnt/afs/davidwang/models/moge2")
# 报 IsADirectoryError: [Errno 21] Is a directory

# 正确 — 传入具体文件路径
MoGeModel.from_pretrained("/mnt/afs/davidwang/models/moge2/model.pt")
```

`MoGeModel.from_pretrained` 内部调用 `torch.load(path)`，要求传文件路径，不能传目录。Pi3X 的 `from_pretrained` 则接受目录。修复：自动检测，若传入目录则追加 `model.pt`：

```python
moge2_path = pathlib.Path(moge2_weights)
moge2_ckpt = moge2_path / "model.pt" if moge2_path.is_dir() else moge2_path
self._moge2 = MoGeModel.from_pretrained(str(moge2_ckpt)).to(self.device).eval()
```

---

### Bug 4: metric_depth 维度错误导致 VIPE buffer IndexError

**错误信息**:
```
IndexError: too many indices for tensor of dimension 2
  File ".../vipe/slam/components/buffer.py", line 265
    disp_sens = disp_sens[:, 3::8, 3::8]
```

**原因**: VIPE buffer 从 `images[frame_idx].moveaxis(1, -1)` 构造 `rgb`，shape 为 `(1, H, W, 3)`（n_views=1）。我们的 `_estimate_single` 错误地 `squeeze(0)` 后返回 `(H, W)`：

```python
# 错误 — 始终 squeeze
depth = out["depth"].squeeze(0)  # 输入是 (1,H,W,3) 时，MoGe 输出 (1,H,W)，squeeze 后变 (H,W)
return DepthEstimationResult(metric_depth=depth)
```

VIPE 随后对 metric_depth 做 `[:, 3::8, 3::8]`（需要3维），2D tensor 触发 IndexError。

**正确做法**（参考 VIPE 官方 moge 后端）：只在**自己添加了 batch dim** 时才 squeeze：

```python
if rgb.dim() == 3:          # 输入是 (H,W,3) —— 调用方没传 batch
    rgb = rgb[None]
    was_unbatched = True
else:                        # 输入是 (1,H,W,3) —— VIPE buffer 传入，已有 batch
    was_unbatched = False

out = self._moge2.infer(inp, fov_x=fov_x)
depth = out["depth"]        # (B, H, W)
if was_unbatched:
    depth = depth.squeeze(0)  # 只在自己加了 batch 时才 squeeze
return DepthEstimationResult(metric_depth=depth)
```

---

### Bug 5: depth_align_model 命名不兼容

**错误信息**:
```
AssertionError: Model name should start with 'adaptive_'
  File ".../vipe/pipeline/processors.py", line 183
    assert prefix == "adaptive", ...
```

**原因**: VIPE 的 `AdaptiveDepthProcessor` 用下划线分割 `depth_align_model` 字符串：

```python
# AdaptiveDepthProcessor.__init__ 内部
prefix, metric_model = model.split("_")  # 期望: "adaptive_unidepth-l"
```

我们原来的 yaml 配置 `depth_align_model: pi3x_moge2`，分割后 `prefix="pi3x"` 不是 `"adaptive"` → AssertionError。

**修复**：改用 `adaptive_pi3x-moge2`（连字符分隔 pi3x 和 moge2，下划线仅用于 `adaptive_` 前缀）：

```yaml
post:
  depth_align_model: adaptive_pi3x-moge2   # ← 正确
  # depth_align_model: pi3x_moge2          # ← 错误：没有 adaptive_ 前缀
```

同时在 VIPE 深度工厂注册 `pi3x-moge2`（连字符形式，通过 `-` 分割）：

```python
# vipe/priors/depth/__init__.py
elif model_name == "pi3x_moge2" or (model_name == "pi3x" and model_sub == "moge2"):
    from .pi3x_moge2 import Pi3XMoGe2DepthModel
    return Pi3XMoGe2DepthModel()
```

注：SLAM 的 `keyframe_depth: pi3x_moge2` 不经过 `AdaptiveDepthProcessor`，保持原下划线写法不变。

---

## 三、Pi3X 调用 API 错误

### Bug 6: Pi3X 没有 infer 方法，且输入维度错误

```python
# 错误
pi3x_out = self._pi3x.infer(frames_t)           # Pi3X 没有 infer 方法
d_pi3x = pi3x_out["depth"].cpu().numpy()        # 输出 dict 没有 "depth" 键
```

**Pi3X 实际 API（通过 inspect 核实）**：

```python
# 正确
# 输入: (B, N, 3, H, W), 其中 B=batch, N=帧数
# 输出 dict: {"points", "local_points", "rays", "conf", "camera_poses", "metric"}
# depth 取 local_points 的 z 分量
frames_4d = frames_t.unsqueeze(0)               # (T,3,H,W) -> (1,T,3,H,W)
pi3x_out = self._pi3x(frames_4d)                # 直接调用 forward()
d_pi3x = pi3x_out["local_points"][0, :, :, :, 2].cpu().numpy()  # (T,H,W)
```

### Bug 7: Pi3X 要求 H/W 必须是 14 的倍数

TUM 分辨率 640×480，而 Pi3X 是 ViT-based 模型（patch size=14），要求 H 和 W 都是 14 的倍数：

```
Error: Input image height 480 is not a multiple of patch height 14
```

**修复**：在传入 Pi3X 前 resize，输出后 resize 回原分辨率：

```python
H_r = (H_img // 14) * 14   # 480 -> 476
W_r = (W_img // 14) * 14   # 640 -> 630
frames_pi3x = F.interpolate(frames_t, size=(H_r, W_r), mode='bilinear', align_corners=False)
# ... Pi3X forward ...
d_pi3x = F.interpolate(d_pi3x_r, size=(H_img, W_img), ...).cpu().numpy()
```

### Bug 8: Pi3X 全序列 OOM

将 613 帧一次性输入 Pi3X 会 OOM。Pi3X 是 ViT，注意力复杂度 O(N² × patches²)：

- N=613，每帧 (630÷14)×(476÷14) = 45×34 = 1530 patches
- 总 tokens = 613 × 1530 ≈ 938,000
- 注意力矩阵 = 938000² × 4 bytes ≈ **3.5 TB**，H100 80GB 远不够

**修复**：分块处理，每块 16 帧，stride=8（50% 重叠），取平均：

```python
CHUNK, STRIDE = 16, 8
for s in range(0, T, STRIDE):
    e = min(s + CHUNK, T)
    chunk = frames_pi3x[s:e].unsqueeze(0)   # (1,16,3,H,W)
    out = self._pi3x(chunk)
    d_pi3x_accum[s:e] += out["local_points"][0,:,:,:,2].cpu().numpy()
    count[s:e] += 1
d_pi3x = d_pi3x_accum / count[:, None, None]
```

---

## 四、为什么 Pi3X 在这次实验中没有被实际调用

### 核心原因：VIPE 流式调用模式 vs Pi3X 批量需求

我们的深度后端 `Pi3XMoGe2DepthModel.estimate()` 有两条路径：

```python
def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
    if src.video_frame_list is not None:
        return self._estimate_video(src)   # ← Pi3X 批量路径
    return self._estimate_single(src)      # ← MoGe-2 单帧路径（实际被调用的）
```

VIPE 的 `AdaptiveDepthProcessor` 在深度对齐阶段的调用方式：

```python
# vipe/slam/components/buffer.py — 每次只传一帧
depth_input = DepthEstimationInput(
    rgb=self.images[frame_idx].moveaxis(1, -1).float(),   # 单帧
    intrinsics=intrinsics[0],
)
disp_sens = depth_model.estimate(depth_input).metric_depth
```

`video_frame_list` 始终为 `None`，因此 Pi3X 的批量路径从未被触发。**整个实验实际比较的是 VIPE+unidepth-l vs VIPE+MoGe-2（单帧）**，而非论文中的完整 Pi3X 序列一致性方案。

### 这是不是实验设计的问题？

不完全是。Pi3X 在 SANA-WM 论文中主要用于 SLAM **关键帧深度初始化**（`keyframe_depth`），而非深度对齐后处理（`depth_align_model`）。在 SLAM 阶段，VIPE 对每个关键帧调用 `estimate()`，此时也是单帧模式。Pi3X 的序列一致性优势需要**全视频批量**处理，这与 VIPE 的流式 SLAM 框架存在根本性冲突。

---

## 五、完整 Pi3X 集成方案（待实现）

要在 VIPE 框架内真正发挥 Pi3X 的序列一致性优势，需要实现一个**视频批量深度后处理器**。以下是两种可行方案：

### 方案 A：扩展 VIPE 添加 VideoDepthAlignProcessor（推荐）

在 VIPE 的 pipeline 后处理阶段，新增一种 processor 类型，先收集完整视频帧再批量处理：

```python
# vipe/pipeline/processors.py 新增
class VideoPi3XDepthProcessor(PostProcessor):
    """收集全视频帧 -> Pi3X 批量（分块）-> MoGe-2 尺度锚定 -> EMA 融合"""
    
    def update_iterator(self, previous_iterator, pass_idx):
        # pass 1: 收集所有帧
        frames, frame_data = [], []
        for frame in previous_iterator:
            frames.append(frame.rgb.cpu().numpy())
            frame_data.append(frame)
        
        # pass 2: Pi3X 分块推理（CHUNK=16, STRIDE=8）
        d_pi3x = run_pi3x_chunked(frames)         # (T,H,W) 序列一致深度
        d_moge  = run_moge2_perframe(frames)       # (T,H,W) 米制尺度深度
        scale   = ema_fuse(d_pi3x, d_moge, 0.99)  # (T,) EMA 尺度
        metric  = d_pi3x * scale[:, None, None]    # (T,H,W) 最终米制深度
        
        # pass 3: 逐帧输出
        for i, frame in enumerate(frame_data):
            frame.metric_depth = metric[i]
            yield frame
```

在 yaml 中启用：
```yaml
post:
  depth_align_model: video_pi3x_moge2   # 新增 processor 类型
```

### 方案 B：预计算 Pi3X 深度并注入（快速验证用）

```bash
# 1. 用 Pi3X 预计算所有帧的深度，保存为 npz
python scripts/precompute_pi3x_depth.py \
    --video experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
    --weights /mnt/afs/davidwang/models/pi3x \
    --output experiments/vipe_comparison/data/pi3x_depth.npz

# 2. 在深度后端里加载缓存，跳过在线推理
# Pi3XMoGe2DepthModel._lazy_load() 中：
cached_path = os.environ.get("SANA_WM_PI3X_DEPTH_CACHE")
if cached_path:
    self._pi3x_cache = np.load(cached_path)["depth"]  # (T,H,W)
```

---

## 六、实验结果与结论

| 指标 | A: VIPE+unidepth-l | B: VIPE+MoGe-2 | 胜者 |
|------|-------------------|--------------------|------|
| ATE RMSE ↓ (m) | 0.0272 | **0.0215** | B 胜 ↓21% |
| ATE mean ↓ (m) | 0.0232 | **0.0192** | B 胜 |
| 估计尺度 (→1.0) | **0.9877** | 1.0745 | A 更接近 1 |
| RTE 平移均值 ↓ (m) | **0.0290** | 0.0405 | A 胜 |
| RTE 旋转均值 ↓ (°) | **1.329** | 1.344 | 近似 |

**ATE 提升**（全局轨迹精度）：MoGe-2 提供更接近米制的绝对深度，Sim3 对齐后尺度误差小，全局轨迹更准。

**RTE 退化**（相对运动一致性）：MoGe-2 是单帧独立估计，帧间深度值波动大；unidepth-l 的时序一致性更好。这正是 Pi3X **序列一致性**应该解决的问题——若真正集成 Pi3X，预期 RTE 也会改善。

---

## 七、可视化文件位置

```
experiments/vipe_comparison/results/
├── plots/comparison.png           # 轨迹对比 + 逐帧ATE + ATE RMSE 柱状图
├── viz/side_by_side.mp4           # RGB | Depth-A | Depth-B 三列对比视频
├── viz/depth_diff.mp4             # |Depth_A - Depth_B| 深度差异热图视频
└── viz/depth_sample/              # 每50帧一张样本 PNG（frame_0000~0600）
```
