# Task Plan: SANA-WM Pipeline CMCC 部署 + 多数据集冒烟测试

**项目：** sana_wm_pipeline — 世界模型训练数据标注管线（arXiv:2605.15178）
**总目标：** 在 CMCC 机器上稳定运行 Default 模式管线，覆盖 jdvbbfb-v3-full 多个数据集 group，产出合格 WebDataset shard

**最后更新：** 2026-06-25

---

## 路径变量（每次新 shell 必须先 export）

```bash
export NEW_BASE=/root/work/david_work
export ENV_DIR="$NEW_BASE/sana_wm_env"
export PROJ_DIR="$NEW_BASE/sana_wm_pipeline"
export DATA_ROOT="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full"
```

## CMCC 快速激活

```bash
source /root/work/david_work/activate_sana_wm.sh
export VIPE_EXT_JIT=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
```

---

## 阶段完成状态

| 阶段 | 内容 | 状态 | 完成日期 |
|------|------|------|---------|
| 1 | CMCC 环境搭建（conda env + 代码 + 权重）| ✅ | 2026-06-15 |
| 2 | nvidia-vipe C++ 扩展编译 | ✅ | 2026-06-15 |
| 3 | activate_sana_wm.sh 激活脚本 | ✅ | 2026-06-15 |
| 4 | DL3DV 单样本 C.1 Smoke Test | ✅ | 2026-06-15 |
| 5 | DL3DV shard 数据正确性手工核验 | ✅ | 2026-06-15 |
| 6 | mode_default.py OOM Bug 修复 | ✅ | 2026-06-15 |
| 7 | Sekai-Real-Walking-HQ 单样本 Smoke Test | ✅ | 2026-06-15 |
| 8 | 其他 group smoke test（OmniWorld-Game / SpatialVID-hq 等）| ⏳ 待做 | — |
| 9 | 批量数据生产（jdvbbfb-v3-full 全量）| ✅ 校验1-8 全部 PASSED（单节点8卡跑通） | 2026-06-16 |
| 10 | 扩容到 6 节点×8卡（48 worker）批量生产 | ✅ 已完成（externalstorage 产出 7 个 group） | 2026-06-25 |
| 11 | QC 系统方案设计（两阶段质检 + 论文对齐）| ✅ 方案文档 v2.1 已完成，等待实施 | 2026-06-25 |
| 12 | QC 系统代码实施（metrics/group_config/stage1/stage2/stage3/report）| ⏳ 待实施 | — |
| 13 | Stage 3 GPU 评估（UniMatch + DOVER + Qwen3.5-27B）| ⏳ 待实施（需先补包重打镜像）| — |
| 14 | 人工审核（human_review 队列）+ 最终 manifest 生成 | ⏳ 待实施 | — |

---

## DL3DV C.1 Smoke Test 脚本（可重跑）

```bash
export NEW_BASE=/root/work/david_work
export ENV_DIR="$NEW_BASE/sana_wm_env"
export PROJ_DIR="$NEW_BASE/sana_wm_pipeline"
export DATA_ROOT="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full"
export OUT_BASE="$NEW_BASE/jdvbbfb_smoke"
export GROUP="wds-DL3DV-ALL-2K"
export SHARD_IDX=0

source "$NEW_BASE/activate_sana_wm.sh"
export VIPE_EXT_JIT=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
cd "$PROJ_DIR"
bash experiments/data_production_smoke/smoke.sh   # 或手动逐阶段
```

**已验证产出（2026-06-15）：**
- 场景：`DL3DV-ALL-2K_6K__10aacaa7c3200...`，160 帧 @16fps，1280×720
- poses_c2w (160,4,4)：SO(3) 行列式=1.000000，米制坐标 [-5.17, 11.87]m
- intrinsics (160,1,4)：fx=866.5，FoV=72.9°，主点偏移=0px
- scale (160,)：全为 1.0（设计行为：度量尺度已内嵌在 poses 平移分量）
- caption：高质量英文场景描述
- schema check：1/1 valid ✅

## Sekai Smoke Test 脚本

```bash
cd /root/work/david_work/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_sekai.sh
```

脚本路径（AFS）：`experiments/data_production_smoke/smoke_sekai.sh`
输出目录：`$NEW_BASE/sekai_smoke/`

**已验证产出（2026-06-15）：**
- 场景：`sekai-real-walking-hq__FP8j6WfkTY_0085528_0087328`，960 帧 @16fps，1280×720，60s
- poses_c2w (960,4,4)：SO(3) 行列式=1.000000，正交误差 max=1.07e-06，米制坐标
- 总轨迹 38.2m，均速 3.98cm/frame（0.64m/s，步行合理），无大跳变（>50cm 跳变=0）
- intrinsics (960,1,4)：fx=998.1，FoV=65.3°，主点偏移=0px（完美居中）
- scale (960,)：全为 1.0（Default 模式设计行为）
- caption：694 字符，高质量英文街景描述
- schema check：1/1 valid ✅

---

## CMCC 批量生产脚本套件（2026-06-16，已就绪待实跑）

路径：`experiments/batch_production/`（6 个文件，均已 spec + code-quality 双阶段评审通过）

| 文件 | 作用 |
|------|------|
| `config.sh` | 路径变量、环境激活、优先级数据集组 BATCH1_GROUPS=(sekai, DL3DV-ALL-2K, SpatialVID-hq) |
| `run_worker.py` | 单 GPU worker，逐 input shard 跑 normalize→pose→pack，写 `.done` 进度标记 |
| `launch_single_node.sh` | 单节点 8 卡启动，round-robin 分片分配 |
| `launch_multi_node.sh` | 多节点包装（RANK/WORLD_SIZE，含 SLURM/OMPI fallback） |
| `run_groups_sequential.sh` | 按优先级串行跑多个 group，单 group 失败不中断后续 |
| `watch_progress.sh` | 实时监控面板（shard 进度/样本 ok-fail/GPU 状态/日志尾） |

**分片分配规则：** `GLOBAL_WORKER = NODE_RANK×8 + LOCAL_GPU`；worker W 负责 input shard 下标 `{W, W+总worker数, W+2×总worker数, ...}`。每 worker 独立输出目录，无锁无竞争。

**CMCC 部署步骤：**
1. 复制 `experiments/batch_production/` 整目录到 CMCC `$PROJ_DIR/experiments/batch_production/`
2. 单节点验证：`bash experiments/batch_production/launch_single_node.sh wds-sekai-real-walking-hq`
3. 多节点验证：`RANK=$i WORLD_SIZE=5 bash experiments/batch_production/launch_multi_node.sh wds-sekai-real-walking-hq`
4. 全量批次：`bash experiments/batch_production/run_groups_sequential.sh`
5. 监控：`bash experiments/batch_production/watch_progress.sh <group>`

详细会话记录见 `progress.md` 会话 5。

### CMCC 实跑校验当前状态（2026-06-16）

校验1-8 全部 PASSED（基础环境/GPU/权重/数据目录/filestorage/config.sh/单样本冒烟/单节点8卡并发抽样质检通过，详见 `progress.md` 会话5 + `findings.md` F-8/F-9）。

- 已对 `launch_single_node.sh` 加 `-u`/`PYTHONUNBUFFERED=1` 修复实时日志可见性问题，**目前只同步到 AFS，CMCC 端尚未应用**（不影响已跑过的单节点任务，下次新启动 group 时生效）

---

## 多节点扩容（6 节点 × 8 卡 = 48 worker，2026-06-16）

用户已获得 6 节点资源（1 master + 5 worker），运维提供 `keep_all_gpu.sh`（GPU 防回收保活，遍历 hostfile SSH 拉起 `gg` 命令）+ `detect_gpu.py`（从平台 `VC_MASTER_*_HOSTS`/`VC_WORKER_*_HOSTS` 环境变量生成 hostfile）两份参考脚本。

**关键判断：** 这个 CMCC 平台是自研 "VC" 调度器，不会像 SLURM/torchrun 那样自动给每个节点注入 `RANK`/`SLURM_NODEID`/`OMPI_COMM_WORLD_RANK`（`launch_multi_node.sh` 里的环境变量 fallback 链在这个平台永远落到默认值 0/1）。必须由我们自己读 hostfile、主动 SSH 派发 rank——这正是运维 `keep_all_gpu.sh` 已经示范的模式。

**新增脚本：** `experiments/batch_production/launch_all_nodes.sh`

```bash
# 用法
bash experiments/batch_production/launch_all_nodes.sh [--batch1-only] [--check-only] <HOSTFILE>
# 示例（HOSTFILE 是 detect_gpu.py 生成的那份）
bash experiments/batch_production/launch_all_nodes.sh --batch1-only ~/work/filestorage/shangaoooooo/world_models/hostfiles
```

- rank = hostfile 中的行号（从0开始），`NUM_NODES = hostfile 行数`
- **阶段1 预检**：SSH 进每个节点检查 `ENV_DIR/bin/python` 存在、`import torch/vipe_ext/vipe/sana_wm_pipeline` 成功、GPU 数量与 hostfile `slots=` 一致、`DATA_ROOT`/`OUT_BASE` 可达——任一节点失败则整体中止，不会拉起真实任务
- **阶段2 拉起**：跳过 `launch_multi_node.sh`（其 env-var 自动探测在此平台用不到），直接对每个节点 SSH 执行 `nohup bash run_groups_sequential.sh [--batch1-only] <rank> <NUM_NODES> > driver_log 2>&1 &`；启动前先 `pgrep` 检查该节点是否已有任务在跑，避免重复启动写坏 worker 输出目录
- driver 日志统一写到共享存储 `$OUT_BASE/driver_logs/node{rank}_driver.log`，`watch_progress.sh` 不用改，本身按 `node*_gpu*.log` glob 自动汇总所有 6 节点的 worker 日志

**改动范围结论：** 只新增 `launch_all_nodes.sh` 这一个文件，`config.sh`/`launch_single_node.sh`/`run_groups_sequential.sh`/`run_worker.py`/`watch_progress.sh` 全部不用改——`launch_single_node.sh` 早已是"给定 NODE_RANK+NUM_NODES 算分片"的纯函数式设计，`run_groups_sequential.sh` 本来就接受 `NODE_RANK NUM_NODES` 位置参数，只是之前从未真正用 `NUM_NODES>1` 跑过。`launch_multi_node.sh` 保留但不在本次部署路径上，留作手动单 group 调试入口。

**待用户验证的风险点（已写入 progress.md，未自动验证）：**
1. `/root/work/david_work`（conda env+vipe_ext+代码+权重）是否 6 节点镜像一致
2. `DATA_ROOT`/`OUT_BASE` 是否 6 节点共享同一份 externalstorage 挂载
3. 运维 `gg` 保活进程会不会和真实 worker 抢显存（`gg` 每卡尝试抓 24GB，OOM 修复后 worker 峰值理论上 < 56GB，但未实测验证）
4. master→5个worker 的 root SSH 免密是否已打通（运维脚本默认成立）

**建议执行顺序：** ① 确认 `keep_all_gpu.sh` 保活在跑 → ② `launch_all_nodes.sh --check-only` 预检 → ③ `launch_all_nodes.sh --batch1-only` 跑批次1三个 group → ④ `watch_progress.sh` 监控 → ⑤ 批次1通过后去掉 `--batch1-only` 跑全量剩余 group。

---

## CMCC 坑点汇总（截至 2026-06-15，共 9 个）

| 错误 | 根因 | 修复 |
|------|------|------|
| `Unknown CUDA arch 12.0+PTX` | TORCH_CUDA_ARCH_LIST 未设置 | activate.sh 加 `9.0` |
| `cusparse.h not found` | conda nvcc 12.4 toolkit 不完整 | 换系统 nvcc 13.0 |
| `__cudaLaunch 2-arg mismatch` | setup.py 强制覆盖 PYTORCH_NVCC | `os.environ.setdefault(...)` |
| `ModuleNotFoundError: nvidia_vipe` | pip 包名 ≠ Python 模块名 | 改用 `import vipe; import vipe_ext` |
| `libtorch_python undefined symbol` | 系统 LD_LIBRARY_PATH 含 Python3.12 torch | activate.sh prepend env torch/lib |
| `VIPE_EXT_JIT=1` 触发 JIT 重编失败 | JIT=1 绕过预编译产物 | 改为 `VIPE_EXT_JIT=0` |
| `No module named 'psutil'` | rerun_sdk 隐式依赖 | `pip install psutil` |
| HF bert-base-uncased 超时 ~2 分钟 | GroundingDINO 无外网重试 | `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` |
| `No module named 'sana_wm_pipeline'` | src layout 未安装 | `pip install -e $PROJ_DIR --no-deps` |
| **CUDA OOM（960帧 Sekai）** | **Pi3X 完成后未释放显存，vipe 子进程显存不足** | **mode_default.py OOM 修复（见下）** |

---

## mode_default.py OOM 修复内容（2026-06-15）

文件：`src/sana_wm_pipeline/stage02_pose/mode_default.py`

**修复 1：Pi3X 后立即释放显存**
```python
# 改前
del pi3x_model
# 改后
del pi3x_model, src, accum, count
torch.cuda.empty_cache()
```

**修复 2：MoGe-2 后立即释放显存**
```python
# 改前
del moge2_model
# 改后
del moge2_model, frames_t
torch.cuda.empty_cache()
```

**修复 3：run_default() 中 cache 管理重构**
- 新增 cache 存在检查：若上次因 OOM 失败但 cache 已存，直接跳过 Pi3X 重算（节省 30+ 分钟）
- vipe 子进程启动前再调一次 `torch.cuda.empty_cache()`
- cache 改为只在 vipe 成功后删除（失败时保留以便 resume）

**已同步到 AFS：** ✅（CMCC 需手动 rsync 或复制粘贴）

### 备用优化（超长视频准备，暂未部署到 CMCC）

若单视频帧数超过 ~5000 帧（约 5 分钟 @16fps），`frames_t` 全量搬 GPU 会超过 80GiB。
已在 AFS 实现 chunk 式逐批搬帧（`frames_cpu[s:e].to(device)`），不影响计算结果，需要时同步到 CMCC 即可。

---

---

## QC 系统方案（阶段 11-14，截至 2026-06-25）

### 已产出 CMCC 数据（7 个 group）

| 输出目录（externalstorage） | 类型 | QC 策略 |
|---|---|---|
| final_wds-DL3DV-ALL-2K | 真实 3D 重建 | 严格（跳变 ≤5次，场景切割 ≤1）|
| final_wds-OmniWorld-Game | 游戏第三人称 | 宽松（跳变 ≤50次，2m阈值）|
| final_wds-SpatialVID-hq | 真实空间视频 | 严格（跳变 ≤5次）|
| final_wds-sekai-real-walking-hq | 真实街道步行 | 中等（跳变 ≤15次，3次标记）|
| final_wds-RealEstate10K-360p | 真实室内漫游 | 严格（跳变 ≤5次，场景切割 ≤1）|
| final_wds-sekai-game-drone | 游戏航拍 | 宽松（跳变 ≤80次，5m阈值）|
| final_wds-sekai-game-walking | 游戏步行 | 宽松（跳变 ≤50次，2m阈值）|

数据路径：`~/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/`
内部结构：`final_{group}/wds-{group}/w{worker:03d}/shard-{idx:06d}-{N:06d}.tar`

### QC 系统架构（三阶段）

```
Stage 1（CPU 全量，~2分钟）
  ├─ 文件完整性 / 数组形状 / 帧数对齐 / NaN-Inf
  ├─ SO(3) 验证 / 首帧归零 / 轨迹跳变（按 group 差异化）
  ├─ FoV[25°,120°] / 焦距差异≤0.20 / 尺度 CV≤2.0
  ├─ 颜色饱和度（OpenCV HSV，部分 group）
  └─ Caption 强动作词检测（pan/zoom/tilt/dolly/track）

Stage 2（CPU 深度，~4~10小时，仅待定+5%抽检）
  ├─ 视频帧数核验（PyAV）
  ├─ 场景切割计数（PySceneDetect，部分 group ≤1）
  ├─ 黑帧比例（>30% 拒绝）
  └─ 轨迹冻结（>50% 帧位移<0.1mm 拒绝）

Stage 3（GPU，48×H100，~6~7小时，全量必做）
  ├─ UniMatch 光流幅值（替代 VMAF Motion）
  ├─ DOVER 视频质量评分
  └─ Qwen3.5-27B VLM：实体计数 + 质量标记 + Caption 改写（捆绑推理）
```

### 确认的关键决策（2026-06-25）

| 决策点 | 结论 |
|---|---|
| Stage 3 是否做 | ✅ 必做，48 H100 约 6~7 小时 |
| Caption 摄像机词处理 | 两级检测（强/弱词）+ Qwen 捆绑改写，零额外成本 |
| 游戏数据跳变上限 | max_jumps_fail=50（原计划 100 收紧），标记阈值=15 |
| MiraData | 本批不生产，配置保留 |
| VMAF Motion | 取消，用 UniMatch 光流均值替代（规避 libvmaf 依赖）|
| Qwen 模型选型 | **Qwen3.5-27B**（支持图像输入已确认），ModelScope 下载 |
| Caption 改写架构 | **不动原始 tar**，改写写入 `qc_output/caption_overrides.jsonl` sidecar |

### QC 代码实施计划（阶段 12，待启动）

文档：`docs/superpowers/plans/2026-06-21-output-qc-system.md`（完整 TDD 计划，含代码）
评审文档：`docs/QC_REVIEW_DESIGN.md`（v2.1，多方评审版）

待实施的 6 个模块（按顺序）：
1. `src/sana_wm_pipeline/qc/metrics.py` — 基础度量函数
2. `src/sana_wm_pipeline/qc/group_config.py` — 差异化阈值注册表（含 7 个 group 配置）
3. `src/sana_wm_pipeline/qc/stage1_fast.py` — 全量 CPU 扫描
4. `src/sana_wm_pipeline/qc/stage2_deep.py` — 针对性深度检测
5. `src/sana_wm_pipeline/qc/stage3_gpu.py` — UniMatch + DOVER + Qwen3.5-27B
6. `src/sana_wm_pipeline/qc/report.py` + `scripts/run_qc.py` — 报告生成 + CLI

### Docker 镜像补包清单（Stage 12 实施前必须完成）

| 包 / 资产 | 用途 | 体积 | 安装方式 |
|---|---|---|---|
| `av`（PyAV） | Stage 2 视频帧数核验 | ~5MB | pip install av |
| `scenedetect` | Stage 2 场景切割检测 | ~10MB | pip install scenedetect |
| `dover` | Stage 3 视频质量评分 | ~30MB | pip install dover |
| DOVER 模型权重 | Stage 3 推理 | ~400MB | 加入 sana_wm-models.tar.gz |
| UniMatch 代码 + 权重 | Stage 3 光流 | ~200MB | 加入 sana_wm-models.tar.gz |
| Qwen3.5-27B 权重 | Stage 3 VLM | ~55GB | ModelScope 下载到 filestorage |

### 备份建议（实施 QC 前）

```bash
# CMCC 上执行：原始数据快照到 filestorage
rsync -av ~/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/ \
          ~/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output_backup/

# 最低限度：生成 md5 清单
find ~/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output -name "*.tar" \
  -exec md5sum {} \; > ~/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output_md5.txt
```

**QC 写操作只写 `qc_output/` 目录，严禁修改原始 tar 文件。**

---

## 数据价值说明

本管线产出的每个 shard 包含：
- `video.mp4`：真实世界归一化视频
- `poses_c2w.npy`：SLAM 估计的米制相机轨迹（Pi3X+MoGe-2 深度先验注入）
- `intrinsics.npy`：每帧内参 [fx,fy,cx,cy]
- `scale.npy`：度量尺度因子（Default 模式已内嵌在 poses，填 1.0）
- `caption.txt`：高质量场景文字描述

用途：直接作为 SANA-WM / HyWorld 等世界模型的相机控制训练数据，无需二次转换。
