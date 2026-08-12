# SANA-WM Camera

用于从视频估计相机轨迹与逐帧内参。

## 输出格式

```text
<out>/
├── <clip-id>.poses.npy        # (N, 4, 4), camera-to-world
├── <clip-id>.intrinsics.npy   # (N, 4), fx, fy, cx, cy，像素单位
├── result.json                # 输入信息、shape、相机 QC 和尺度摘要
└── <clip-id>/                 # 真实 VIPE 推理的中间结果
```

`poses.npy` 使用 OpenCV 轴定义的 `c2w` 约定。视频帧、pose 和 intrinsics 必须使用
相同的采样索引。

## 安装与自检

```bash
cd /data/workspace/code/sana-wm-data-clean
python3 -m pip install -e '.[dev]'
pytest -q
```

## CPU dry-run

不安装 GPU 模型即可验证完整接口和输出格式：

```bash
sana-wm-camera input.mp4 \
  --out /tmp/camera-out \
  --dry-run
```

## 真实视频相机估计

先安装 Pi3、MoGe-2 和 VIPE 环境，并应用逐帧内参 BA 补丁：

```bash
bash scripts/setup_camera_env.sh
bash scripts/setup_vipe.sh
bash vipe_patches/apply_vipe_patches.sh
```

然后运行：

```bash
sana-wm-camera input.mp4 --out /tmp/camera-out
```

模型默认存放在 `weights/`，第三方代码存放在 `third_party/`，VIPE 虚拟环境为
`.venv-vipe/`。可以使用以下环境变量覆盖：

```bash
export SANA_WM_ROOT=/path/to/sana-wm-data-clean
export SANA_WM_WEIGHTS=/path/to/weights
export VIPE_VENV=/path/to/vipe-venv
```

限制送入 Pi3/MoGe 的最大采样帧数：

```bash
sana-wm-camera input.mp4 \
  --out /tmp/camera-out \
  --max-frames 64
```

## 使用已有 GT Pose

```bash
sana-wm-camera input.mp4 \
  --mode gt_pose \
  --gt-poses poses.npy \
  --gt-intrinsics intrinsics.npy \
  --out /tmp/camera-out
```

支持的 GT pose 格式：

- `(N, 4, 4)`：camera-to-world 矩阵；
- `(N, 3, 4)`：省略最后一行的 camera-to-world 矩阵；
- `(N, 3)`：camera center。

内参支持 `(N, 4)` 或 `(4,)`，字段顺序为 `fx, fy, cx, cy`。`gt_pose` 模式保留输入
轨迹，并通过 Pi3 与 Umeyama 对齐估计尺度。

## 后端选择

```bash
# 自动选择：default 模式使用真实 VIPE
sana-wm-camera input.mp4 --out /tmp/camera-out --backend auto

# 显式使用 VIPE
sana-wm-camera input.mp4 --out /tmp/camera-out --backend vipe

# 使用轻量 reference stage
sana-wm-camera input.mp4 --out /tmp/camera-out --backend reference
```

## 逐帧内参验证

从带已知内参的 clip 构造焦距渐变视频：

```bash
python3 scripts/make_zoom_clip.py <src-clip-dir> <zoom-clip-dir> 1.8
```

比较恢复内参与精确 GT：

```bash
python3 scripts/compare_intrinsics.py \
  <recovered.npy> \
  <zoom-clip-dir>/gt_intrinsics.npy
```

## Sekai-Game 相机约定

Sekai-Game 的原始 `extrinsic` 已是 `c2w`。按相同帧索引采样即可，不要再右乘
`diag(1,-1,-1,1)`，否则会改变正确的相机轴定义。
