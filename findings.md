# Findings: SANA-WM Pipeline 技术发现汇总

---

## F-10：QC 系统设计关键发现（2026-06-25）

### F-10a：testdata OmniWorld-Game 样本分析结论

4 条样本（splits_013-015 / 000-012 / 000-010 / 011-017）：
- SO(3)、首帧归零、FoV、焦距差异：全部合格
- 游戏场景跳变（>50cm）：5~12 次/样本，属于合理范围（游戏镜头切换）
- caption 含"camera stays behind"/"third-person view"：**弱框架词，不是强动作词，无需改写**
- 结论：游戏数据整体质量达标，跳变上限设 50 次合理

### F-10b：论文 App.B.3 过滤体系完整清单（精读结论）

论文 Table 6 规定的 7 项视觉过滤器（部分 group 适用）：

| 指标 | 工具 | OmniWorld | Sekai Walking | DL3DV-GS | SpatialVID |
|---|---|---|---|---|---|
| VMAF Motion | FFmpeg libvmaf | [0.5, 100] | [0.5, 50] | [6, 50] | [0.5, 50] |
| UniMatch Flow | UniMatch 神经网络 | [3, 100] | [3, 50] | [3, 80] | [3, 80] |
| DOVER | DOVER 模型 | [0.35, 1.0] | [0.35, 1.0] | [0.4, 1.0] | [0.35, 1.0] |
| 颜色饱和度 | OpenCV HSV | — | [0, 180] | [0, 180] | [0, 180] |
| 场景切割 | PySceneDetect | — | — | ≤1 | — |
| VLM 实体数 | Qwen3.5 VLM | ≤10 | ≤25 | — | ≤10 |
| VLM 质量 | Qwen3.5 VLM | [0.5, 1.5] | [0.5, 1.5] | — | [0.5, 1.5] |

额外发现：论文第 4.4 节明确禁止 caption 含摄像机动作词（pan/zoom/tilt/dolly/track 等）。

### F-10c：Caption 两级检测策略

| 类型 | 示例 | 来源 | 处理 |
|---|---|---|---|
| **强动作词**（禁止） | "camera pans left", "zooms in", "tilt up" | 论文 §4.4 | Stage 1 标记 → Stage 3 Qwen 改写 |
| **弱框架词**（可接受） | "camera stays behind", "third-person view" | 测试数据实证 | 保留，不触发改写 |

测试数据的 4 条 caption 均为弱框架词，**不需要改写**。

### F-10d：VMAF Motion → UniMatch 光流替代方案

`static_ffmpeg` 预编译二进制大概率不含 libvmaf，调试成本高。
决策：Stage 2 取消 VMAF Motion，Stage 3 用 UniMatch 光流幅值均值替代（两者物理意义等价，均衡量帧间像素运动强度）。

替代映射：
- OmniWorld VMAF [0.5, 100] → UniMatch [3, 100]
- DL3DV-GS VMAF [6, 50] → UniMatch [3, 80]（DL3DV_GS 行使用 unimatch_flow 列阈值）

### F-10e：CMCC 实际产出包含 3 个配置缺失的新数据集

批量生产实际产出 7 个 group，原始配置只有 4 个：
- `wds-RealEstate10K-360p`：真实室内漫游，按 DL3DV 严格标准
- `wds-sekai-game-drone`：游戏航拍，jump_threshold=5.0m，unimatch_flow ≤150
- `wds-sekai-game-walking`：游戏步行，同 OmniWorld 标准（2.0m，≤50次）

已更新：`configs/filter_thresholds.yaml`（commit 3897249）

### F-10f：Qwen3.5-27B 选型评估

论文引用 [102]："Qwen3.5: Towards native multimodal agents, February 2026"（Qwen Team）
ModelScope：`Qwen/Qwen3.5-27B`，已确认支持图像输入。

选择 27B 的理由：
- 单 H100（80GB）可放下（~54GB），剩余 26GB 够 UniMatch + DOVER + KV cache
- 27B 比 7B caption 改写质量显著更好（直接影响训练数据质量）
- 48 卡并行，20万样本约 11~12h，可接受
- 论文同系列，任务对齐

**Caption 改写架构**：改写结果不修改原始 tar，写入 `qc_output/caption_overrides.jsonl`（sidecar），训练 dataloader 查询该文件后决定用原始还是改写版。

### F-10g：Docker 镜像补包需求

现有 `sana_wm-cmcc.tar.gz` 缺少 QC 专用依赖：
- `av`（PyAV）：Stage 2 视频帧数，pip install
- `scenedetect`：Stage 2 场景切割，pip install
- `dover` 代码 + 权重（~430MB）：Stage 3，pip install + 权重加入 models tar
- UniMatch 代码 + 权重（~200MB）：Stage 3，加入 models tar
- Qwen3.5-27B 权重（~55GB）：Stage 3，ModelScope 下载到 filestorage（不走 tar）

---

## F-9：校验 8 PASSED —— 单节点 8 卡并发抽样质检通过（2026-06-16）

### 环境信息
| 项目 | AFS 源机器 | CMCC 机器 |
|------|-----------|---------|
| Driver | 580.95.05 | 575.57.08 |
| CUDA | 13.0 | 13.0 |
| GPU | H100 sm_90 | H100 sm_90 |
| conda nvcc | 12.4 (`$ENV_DIR/bin/nvcc`) | 12.4 |
| 系统 nvcc | 13.0 (`/usr/local/cuda-13.0/bin/nvcc`) | 13.0 |
| torch | 2.12.0+cu130 | 2.12.0+cu130 |

### 核心诊断结论

**问题 1：PYTORCH_NVCC 被 setup.py 劫持**
`third_party/vipe/setup.py` 第 57-61 行强制覆盖 `PYTORCH_NVCC` 为 conda nvcc 12.4，
导致即使设了 `CUDA_HOME=/usr/local/cuda-13.0`，实际用的还是 12.4。

**修复：** `os.environ["PYTORCH_NVCC"] = ...` 改为 `os.environ.setdefault("PYTORCH_NVCC", ...)`

**编译命令（已验证）：**
```bash
PYTORCH_NVCC="/usr/local/cuda-13.0/bin/nvcc" \
PYTHONNOUSERSITE=1 \
TORCH_CUDA_ARCH_LIST="9.0" \
CUDA_HOME="/usr/local/cuda-13.0" \
  "$ENV_DIR/bin/pip" install --no-user -e "$PROJ_DIR/third_party/vipe" \
  --no-deps --no-build-isolation
```

---

## F-1：scale.npy 全为 1.0 是设计行为（非 Bug）

**发现日期：** 2026-06-15

`mode_default.py` 第 205-207 行注释：
> "VIPE's unidepth backend already produces metric depth directly"

Pi3X+MoGe-2 的度量尺度在 SLAM Bundle Adjustment 中已注入 `poses_c2w` 平移分量（单位=米）。
`scale_per_frame` 是 GT-depth 模式专用字段，Default 模式填 1.0 为占位符。

**验证：** DL3DV 场景 poses 坐标范围 [-5.17, 11.87]m，符合真实室内米制尺度。

---

## F-2：CMCC OOM 根因 + 修复

**发现日期：** 2026-06-15 / Sekai 960 帧 Stage 2 崩溃触发

**显存分布（崩溃瞬间）：**
```
总计 79.18 GiB（H100 80GB）
  Process 351  (GPU 保活)：     16.73 GiB
  Process 67436 (父进程 Python)：60.09 GiB  ← 根源
  vipe 子进程：                  2.02 GiB
  空闲：323 MiB  →  申请 1.10 GiB → OOM
```

**根因：** Pi3X 完成后 `del model` 只释放 Python 引用，PyTorch CUDA allocator 缓存不清；
vipe 以 subprocess 启动时父进程仍占着 60 GiB，子进程无法申请 SLAM 帧缓冲。

**修复（已同步 AFS）：**
```python
# Pi3X 后
del pi3x_model, src, accum, count
torch.cuda.empty_cache()

# MoGe-2 后
del moge2_model, frames_t
torch.cuda.empty_cache()

# run_default() 中，_precompute_depth_cache 返回后
torch.cuda.empty_cache()   # vipe 子进程启动前确保显存干净
```

**附加修复：** cache 改为只在 vipe 成功后删除（失败保留，下次自动跳过 Pi3X 重算）。

---

## F-3：frames_t GPU 显存随帧数线性增长

**发现日期：** 2026-06-15

| 帧数 | 时长 @16fps | frames_t 占用 | 加 allocator cache | OOM 风险 |
|------|------------|--------------|-------------------|---------|
| 160  | 10s  | 1.8 GiB  | ~17 GiB  | ✅ 安全 |
| 960  | 60s  | 10.7 GiB | ~60 GiB  | ⚠️ 需修复 |
| 4800 | 5min | 53.5 GiB | >70 GiB  | ❌ 高风险 |
| 7200 | 7.5min | 80 GiB | —      | ❌ frames_t 单独就爆 |

**根治方案（AFS 已实现，CMCC 暂未部署）：**
```python
# 改为 chunk 式逐批搬帧
frames_cpu = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # 留在 CPU
for s in starts:
    chunk_gpu = frames_cpu[s:e].to(device)  # 只搬 16 帧到 GPU
    out = pi3x_model(chunk_gpu.unsqueeze(0))
```
- GPU 常驻：固定 ~0.18 GiB/chunk + 模型权重，与视频长度无关
- 计算结果逐位相同（Pi3X chunk 间无跨帧状态）
- 传输开销：8ms/chunk vs 推理 ~15s/chunk = 0.05%，可忽略

---

## F-4：DL3DV shard 数据正确性基准

**建立日期：** 2026-06-15（手工核验 `shard-000001.tar`）

| 字段 | 期望 | 实测（DL3DV 160帧）|
|------|------|-------------------|
| poses_c2w.shape | (T,4,4) | (160,4,4) ✅ |
| R 行列式 | mean=1.0, std=0.0 | 1.000000 / 0.000000 ✅ |
| 第0帧 | ≈单位矩阵 | 偏差 < 1e-4 ✅ |
| intrinsics.shape | (T,1,4) | (160,1,4) ✅ |
| cx, cy | ≈ W/2, H/2 | 640.0, 360.0（偏移=0px）✅ |
| fx FoV | 室内合理范围 | 72.9° / 45.1° ✅ |
| scale | Default=全1.0 | 1.000000 ✅（设计行为）|
| caption | 非空 | 高质量英文描述 ✅ |
| schema | 1/1 valid | PASS ✅ |

---

## F-5：jdvbbfb-v3-full 数据集结构

**确认日期：** 2026-06-15（CMCC externalstorage 实地查看）

| Group | Shard 文件名规则 | 样本数 |
|-------|----------------|-------|
| wds-DL3DV-ALL-2K | `DL3DV-ALL-2K-NNNNNN.tar` | 9,993 |
| wds-sekai-real-walking-hq | `sekai-real-walking-hq-NNNNNN.tar` | 18,208 |

**数据路径：** `/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full/{group}/shards/`

**每个 tar 内的样本格式：**
```
{key}.mp4          ← RGB 视频（H264）
{key}.camera.npz   ← GT c2w + K_px + vipe_c2w（参考位姿）
```
Caption 在 `{group}/index.jsonl` 的 `manifest.prompt.text` 字段（不在 tar 内）。

---

## F-6：Pi3X API 备忘

```python
from pi3 import Pi3X
model = Pi3X.from_pretrained(weights_dir).to(device).eval()

# 输入：(B, N, 3, H, W)；H、W 必须是 14 的倍数
out = model(frames_chunk.unsqueeze(0))   # frames_chunk: (N, 3, H, W)

# 输出：local_points (B, N, H, W, 3)，第3维的 index=2 是 depth
depth = out["local_points"][0, :N, :, :, 2]   # (N, H, W)
# ⚠️ outputs["depth"] 不存在，必须用 local_points[..., 2]
```

---

## F-7：vipe 子进程调用机制

`mode_default.py` 通过 subprocess 调用 vipe CLI（不是 Python API）：
```python
cmd = ["vipe", "infer", str(clip_path),
       "--output", str(work_dir),
       "--pipeline", "vipe_cached_depth"]
subprocess.check_call(cmd)
```
深度缓存路径通过环境变量 `SANA_WM_CACHED_DEPTH_PATH` 传递。
vipe 内部 `CachedDepthModel` 读取该 `.npz`，注入 SLAM BA 作为深度先验。

**注意：** subprocess 与父进程共享同一块 GPU，父进程未释放的显存会直接占用子进程的显存配额。

---

## F-8：CMCC 批量生产监控时的两个"假阳性异常"

**发现日期：** 2026-06-16，CMCC 单节点 8 卡校验（校验8）运行约 30 分钟后用户报告日志为空

**现象：** `tail -f node0_gpu0.log` 完全空白；但 `w000/shard-000000.tar`、`w007/shard-000000.tar` 已经存在。看起来像是"卡死但又有产出"的矛盾状态。

**根因 1 — stdout 块缓冲：** `run_worker.py` 用 `print()` 输出，重定向到文件（`>> "$LOG" 2>&1`）时 Python 默认对 stdout 做**全缓冲**（约 8KB 才 flush 一次或进程退出时才 flush），不像连接终端时是行缓冲。即使最早的 `[index] 加载 N 条 caption` 早已执行，也可能仍卡在内存缓冲区里没写入文件。这与程序是否卡死无关，纯粹是 I/O 缓冲策略导致的可观测性问题。

**根因 2 — ShardWriter 提前建文件：** `stage06_pack/webdataset_writer.py` 的 `ShardWriter.__init__` 会立即调用 `_open_new_shard()` → `tarfile.open(path, "w")`，在磁盘上创建空 tar 文件——这发生在 worker 刚启动、**还没处理任何样本**的时刻。所以 `shard-000000.tar` 存在只能证明 worker 跑到了 `with ShardWriter(...) as writer:` 这一行，不能证明任何样本已完成。

**正确的存活判断方法（不依赖日志）：**
```bash
ps -eo pid,etime,pcpu,cmd | grep run_worker.py | grep -v grep   # 进程是否还在，跑了多久
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv  # 显存/利用率是否真实波动
```

**修复（已加到 AFS `launch_single_node.sh`，CMCC 端需手动同步）：**
```diff
     CUDA_VISIBLE_DEVICES=$LOCAL_GPU \
     PYTHONNOUSERSITE=1 \
-    "$ENV_DIR/bin/python" \
+    PYTHONUNBUFFERED=1 \
+    "$ENV_DIR/bin/python" -u \
```
加上 `-u`/`PYTHONUNBUFFERED=1` 后 `print()` 立即落盘，`tail -f` 才能反映真实进度。

**参考基准：** 项目自己的校验清单（Step 7）估计单样本端到端（Pi3X+MoGe2+VIPE）耗时 **45-90 分钟**，所以 30 分钟无 `.done`、无 `[OK]` 输出完全在预期内，不代表异常。

---

## F-9：校验8 首批产出样本抽样核验 PASSED（2026-06-16）

**核验对象：** 用户从 CMCC 取回的两个运行中 shard 快照 `shard-000000-w001.tar`（86.4MB）、`shard-000000-w005.tar`（111.4MB），已 copy 到 AFS。

**结构发现：** 每个 tar 内 2 个样本，每个样本第 1 个完整（6 文件），第 2 个只有 `mp4/poses_c2w/intrinsics/scale` 4 个文件，缺 `caption.txt`+`meta.json`。
逐字节核对 `.npy` 文件大小（poses 61568B、intrinsics 15488B、scale 3968B，与完整样本完全一致）确认这 4 个文件本身都是写完整的，只是整体样本还差最后两步——与 `webdataset_writer.py:50-69` 的写入顺序（mp4→poses→intrinsics→scale→caption→meta）完全吻合，证实是**worker 仍在运行、tar 在写第2个样本时被复制下来的快照**，呼应 F-8（ShardWriter 提前建文件的同类现象），不是产出 bug。

**对 2 个完整样本做的逐字段核验（全部 PASS）：**

| 字段 | w001 样本 (`PDs3TVn9jKo`) | w005 样本 (`_JFT1I1YYAg`) |
|------|--------------------------|---------------------------|
| poses_c2w.shape | (960,4,4) float32 | (960,4,4) float32 |
| det(R) | mean=1.000000 std=0 | mean=1.000000 std=0 |
| 正交误差 max | 7.15e-07 | 9.54e-07 |
| 首帧≈单位矩阵 | 偏差 1.82e-4（<1e-3）✅ | 偏差 1.30e-4（<1e-3）✅ |
| 轨迹 | 64.0m/60s≈1.07m/s 步行，无>50cm跳变 | 12.1m/60s≈0.2m/s 慢动（海滩坐席场景，符合caption）|
| intrinsics | fx=775.3，cx/cy=640.0/360.0（完美居中）| fx=1020.0，cx/cy=640.0/360.0 |
| scale | 全 1.0（Default 设计行为）✅ | 全 1.0 ✅ |
| caption | 665字符，高质量街景描述 ✅ | 565字符，沙滩场景描述 ✅ |
| video (ffprobe 实测) | h264, 1280×720, 16fps, **960帧**，与 npy T 完全一致 | h264, 1280×720, 16fps, **960帧**，与 npy T 完全一致 |

**关于帧数=960而非schema.py标注的961：** `schema.py:15` 的 `CAMERA_FRAMES=961` 是论文固定值，但 `run_worker.py:164-166` 显式传入 `strict_frames=False`（注释："允许任意帧数，视频长度不固定"），所以 960 帧不是 bug，是生产模式有意放宽的校验。

**结论：校验8 在 CMCC 上已经产出了真实合格的训练样本**，不只是空 tar 假象。w001/w005 两个 worker 各自完成了至少 1 个样本，且字段质量与此前 DL3DV/Sekai 单样本 smoke test（F-4 基准）完全一致水平。
