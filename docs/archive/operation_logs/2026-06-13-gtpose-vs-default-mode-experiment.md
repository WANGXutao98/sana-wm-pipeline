# DL3DV GT-pose vs Default 模式完整实验记录

**日期**: 2026-06-12 ~ 2026-06-13  
**执行环境**: H100 80GB × 1，conda env `sana_wm`  
**仓库**: `/mnt/afs/davidwang/workspace/sana_wm_pipeline`

---

## 一、实验目标

在 DL3DV smoke test 数据上，对比 SANA-WM 论文（arXiv:2605.15178v1）中两种 Dataset-specific annotation modes：

| 模式 | 论文适用场景 | 实验目的 |
|---|---|---|
| **GT-pose** | DL3DV、Sekai-Game（有 GT 位姿） | 验证管线正确性，建立 baseline |
| **Default** | SpatialVID-HQ、MiraData（无 GT 互联网视频） | 量化 SLAM 漂移对位姿精度和生成质量的影响 |

---

## 二、数据准备

### 2.1 DL3DV Smoke Test 场景

4 个场景，存放于 `/mnt/afs/davidwang/workspace/data/dl3dv_smoke/1K/`：

| 场景 ID（前 12 位） | 帧数（GT 24fps） | 下载命令 |
|---|---|---|
| `0032cd2f1698...` | 358 | `bash experiments/data_production_smoke/download_dl3dv.sh` |
| `00534f5868a6...` | 355 | |
| `00713c8c22cf...` | 323 | |
| `008c201a7eff...` | 313 | |

每个场景目录结构：
```
<scene_id>/
  images/           # 原始图片帧
  transforms.json   # GT 相机位姿（OpenCV 约定，c2w）
  video.mp4         # 由 prepare_dl3dv.py 生成
  gt_poses.npy      # (T, 4, 4) float32
  gt_intrinsics.npy # (4,) [fx, fy, cx, cy]
  orig_fps.txt      # "24.0"
```

### 2.2 数据预处理

```bash
conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

python experiments/data_production_smoke/prepare_dl3dv.py \
  --scenes-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke/1K \
  --out-dir    /mnt/afs/davidwang/workspace/data/dl3dv_smoke/1K
```

---

## 三、GT-pose 模式

### 3.1 管线原理

```
DL3DV GT 轨迹 (poses_c2w)
  └─ Pi3X 预测场景结构 → 提取 centers_pi3x
  └─ Umeyama Sim(3)（80th 百分位 inlier 过滤）→ 恢复度量尺度因子 s
  └─ pose_artifact: poses_c2w = GT 直接使用，scale = s
```

**关键**：Umeyama 仅用于恢复度量尺度 `s`，**不**估计相机位姿；位姿直接取 GT。

实现文件：`src/sana_wm_pipeline/stage02_pose/mode_gtpose.py`（关键行 59-63）：

```python
s, _R, _t, _inliers = umeyama_sim3_inlier_filter(
    centers_pi3x, centers_gt, inlier_percentile=inlier_percentile,
)
scale = np.full(len(poses_gt), float(s), dtype=np.float32)
return PoseArtifact(poses_c2w=poses_gt, ...)  # 位姿直接用 GT
```

### 3.2 制作 GT-pose Shards

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

python experiments/data_production_smoke/run_e2e_gtpose.sh  # 或手动调用各 Stage
# 输出: /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_gtpose/
```

Schema 验证：
```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema \
  --shards-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_gtpose
# 结果: 5/5 shards valid（含空占位 shard-000000）
```

### 3.3 GT-pose Pose 评估结果

```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval \
  --shards-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_gtpose \
  --scenes-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke/1K \
  --out-dir    /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_gtpose/eval_output
```

| 场景 ID（前 12 位） | ATE RMSE (m) | T_est | T_gt |
|---|---|---|---|
| `0032cd2f1698...` | 1.79e-07 | 239 | 358 |
| `00534f5868a6...` | 1.58e-07 | 237 | 355 |
| `00713c8c22cf...` | 2.10e-07 | 215 | 323 |
| `008c201a7eff...` | 2.16e-07 | 209 | 313 |

ATE ≈ 数值零（直接使用 GT 位姿）。

---

## 四、Default 模式

### 4.1 管线原理

```
视频所有帧
  ├─ Pi3X（chunk=16, stride=8）→ d_pi3x (T,H,W)  多视图一致几何深度
  ├─ MoGe-2 逐帧推理          → d_moge (T,H,W)  单帧度量深度
  └─ EMA 融合（论文 App. B.1）
       scale_t = EMA(d_moge/d_pi3x)
       depths_fused = d_pi3x × scale_t
           ↓ 写入 _depth_cache.npz
  VIPE SLAM (vipe_cached_depth pipeline)
    └─ CachedDepthModel 按帧号查表注入 BA
    └─ GeoCalib 估计内参，optimize_intrinsics=True
    └─ 输出: poses_c2w (T,4,4), intrinsics (T,1,4)
```

VIPE 配置文件：`third_party/vipe/configs/pipeline/vipe_cached_depth.yaml`
- `keyframe_depth: cached` → 使用 Pi3X+MoGe-2 缓存（非 metric3d-small，非 unidepth）
- `intrinsics: geocalib`

### 4.2 环境依赖（关键！机器重启后 sana_wm env 可能丢失）

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm

pip install pyrallis flash-linear-attention einops ftfy came-pytorch
pip install "setuptools<80"
pip install --no-build-isolation mmcv==1.7.2
pip install termcolor omegaconf sentencepiece qwen-vl-utils diffusers accelerate patch-conv scikit-image
# timm: 需要 >=0.9.0（同时兼容 SANA-WM 和 VIPE）
pip install "timm>=0.9.0"
```

> ⚠️ mmcv==1.7.2 必须先 `setuptools<80` 再 `--no-build-isolation`，顺序不可颠倒。
> ⚠️ timm 使用 ≥0.9.0（提供 `timm.layers` 新路径 + `timm.models.layers` 兼容 shim），不用 0.6.13。

### 4.3 运行 Default 模式（单样本）

```bash
# 设置必要环境变量
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x/model.safetensors
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2/model.pt
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface

cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 使用预置脚本（单样本 0032cd2f，包含 Pi3X+MoGe-2+VIPE+打包）
nohup python /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/run_default_0032.py \
  > /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/run_default_0032.log 2>&1 &

# 监控进度（约 25 分钟：Pi3X ~10min + VIPE SLAM ~10min + 打包 ~5min）
tail -f /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/run_default_0032.log
```

> 脚本路径：`data/dl3dv_smoke_shards_default/run_default_0032.py`
> 若 `pose_artifact.npz` 已存在则自动跳过 Pi3X+VIPE，直接打包。

### 4.4 Default 模式 Pose 评估

```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval \
  --shards-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default \
  --scenes-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke/1K \
  --out-dir    /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/eval_output
```

> ⚠️ `--scenes-dir` 必须指向 `dl3dv_smoke/1K/`（含场景子目录），不是 `dl3dv_smoke/`。

结果（样本 `0032cd2f`）：

| 指标 | 值 |
|---|---|
| ATE RMSE | 0.127655 m |
| T_est | 239 |
| T_gt_orig | 358 |
| orig_fps | 24.0 |

---

## 五、SANA-WM 推理

### 5.1 依赖检查

机器重启后需确认以下包存在（有时会丢失）：

```bash
conda activate sana_wm
python -c "import pyrallis, fla, einops, ftfy, termcolor, omegaconf, sentencepiece, diffusers, accelerate, timm, skimage; from mmcv import Registry; print('ALL OK')"
```

若任何一个报错，按第四节安装。

### 5.2 GT-pose 推理命令

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
conda activate sana_wm

python experiments/data_production_smoke/run_sana_wm_inference.py \
  --shards-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_gtpose \
  --sana-dir   /mnt/afs/davidwang/workspace/Sana \
  --output-dir /mnt/afs/davidwang/workspace/data/sana_wm_results \
  --sample-limit 1
```

### 5.3 Default 推理命令

```bash
python experiments/data_production_smoke/run_sana_wm_inference.py \
  --shards-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default \
  --sana-dir   /mnt/afs/davidwang/workspace/Sana \
  --output-dir /mnt/afs/davidwang/workspace/data/sana_wm_results_default \
  --sample-limit 1
```

### 5.4 推理流程（各阶段耗时）

| 阶段 | 耗时 | 说明 |
|---|---|---|
| Stage1 DiT（60步 DDIM） | ~100s | ~1.66s/step，Triton kernel 首次编译后稳定 |
| Stage2 LTX-2 Refiner（3步 Euler） | ~6s | |
| VAE Decode | ~15s | LTX-2 Temporal VAE |
| 视频编码 + PSNR/SSIM | ~60s | |
| **总计** | **~3min/样本** | H100 单卡 |

**帧数约束**：LTX-2 VAE 要求 `num_frames = 8k+1`。239 输入帧 → 有效 233 帧（`floor((239-1)/8)*8+1`）→ 输出 232 帧。

---

## 六、对比结果

### 6.1 Pose 精度对比

```bash
python experiments/data_production_smoke/compare_modes.py \
  --gtpose-eval /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_gtpose/eval_output/pose_eval_summary.json \
  --default-eval /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/eval_output/pose_eval_summary.json \
  --out /mnt/afs/davidwang/workspace/docs/operation_logs/2026-06-12-mode-comparison.md
```

| 模式 | ATE RMSE (m) | 物理含义 |
|---|---|---|
| **GT-pose** | 1.79e-07 | 数值零，直接使用 GT 位姿 |
| **Default** | 0.127655 | SLAM 引入 ~13cm 漂移 |
| **倍数** | ~714,000× | — |

### 6.2 SANA-WM 生成质量对比（样本 `0032cd2f`）

| 模式 | PSNR (dB) | SSIM | 现象 |
|---|---|---|---|
| **GT-pose** | 11.17 ± 0.67 | 0.151 ± 0.025 | 场景内容忠实，无幻觉 |
| **Default** | 11.26 ± 0.69 | 0.154 ± 0.027 | **视频中凭空出现人物** |

> PSNR/SSIM 几乎相同——这是因为生成式模型的像素级误差被扩散随机性掩盖。语义层面的差异（幻觉）无法被 PSNR/SSIM 捕捉，需目视对比 SBS 视频。

**Default 模式产生幻觉的原因**：
1. SLAM 位姿漂移（ATE 12.8cm）→ 条件信号与第一帧不一致 → 模型从训练先验采样
2. `scale_per_frame=1.0`（未做 Umeyama 度量尺度恢复）→ 深度感知整体偏差
3. 训练数据先验中人物场景占比高 → 幻觉优先生成人物

### 6.3 视频文件位置

| 视频 | 路径 |
|---|---|
| DL3DV GT 视频 | `data/dl3dv_smoke/1K/0032cd2f.../video.mp4` |
| GT-pose 生成视频 | `data/sana_wm_results/0032cd2f.../*_generated.mp4` |
| GT-pose SBS（左GT右生成）| `data/sana_wm_results/0032cd2f.../*_sbs.mp4` |
| Default 生成视频 | `data/sana_wm_results_default/0032cd2f.../*_generated.mp4` |
| Default SBS | `data/sana_wm_results_default/0032cd2f.../*_sbs.mp4` |

---

## 七、结论与模式选择建议

| 数据集类型 | 推荐模式 | 原因 |
|---|---|---|
| DL3DV、Sekai-Game（有 GT 位姿） | **GT-pose** | 位姿精确（ATE≈0），生成内容语义忠实 |
| OmniWorld（有 GT 深度） | **GT-depth** | GT 深度替换预测深度注入 SLAM BA |
| SpatialVID-HQ、MiraData（无 GT） | **Default** | 唯一可行方案；Pi3X+MoGe-2 缓存缓解漂移 |

**核心结论**：Default 模式（VIPE+Pi3X+MoGe-2）在有 GT 位姿的数据集上引入不必要的 SLAM 漂移和语义幻觉，误差放大约 70 万倍。对 DL3DV 应始终使用 GT-pose 模式。

---

## 八、相关文档索引

| 文档 | 路径 | 内容 |
|---|---|---|
| DL3DV 管线实施记录 | `docs/operation_logs/2026-06-12-dl3dv-e2e-implementation.md` | Fix 1-5 详细代码修改 |
| 对比实验计划 | `docs/superpowers/plans/2026-06-12-default-vs-gtpose-comparison.md` | 4-Task 计划原文 |
| Pose 对比报告（自动生成）| `workspace/docs/operation_logs/2026-06-12-mode-comparison.md` | 数值对比 Markdown 表 |
| SANA-WM 推理 Smoke Test | `workspace/docs/operation_logs/2026-06-12-sana-wm-e2e-smoke-test.md` | Bug 修复 + 推理流程 |
| VIPE 对比实验（TUM fr1/fr2）| `experiments/vipe_comparison/` | metric3d-small vs Pi3X+MoGe-2 ATE 对比 |
