# CMCC 批量生产脚本套件

## 标准输出目录规范

```
$OUT_BASE/                         # = 启动参数2「自定义输出路径」，每个数据集组独占一个
└── <group>/                       # 如 wds-sekai-real-walking-hq
    ├── w000/                      # worker 0 输出（NODE_RANK*8+LOCAL_GPU）
    │   ├── shard-000000-000000.tar   # 输入 shard 0 → 输出（part 0）
    │   ├── shard-000008-000000.tar   # 输入 shard 8（round-robin 步长=全局worker数）
    │   └── ...
    ├── w001/ ... w0NN/
    ├── progress/
    │   ├── 000000.done            # JSON: {shard_idx,n_ok,n_fail,elapsed_s}
    │   └── ...                    # 每个"已完成输入 shard"一个标记
    ├── logs/
    │   ├── node0_main_<ts>.log    # 单节点总日志
    │   └── node0_gpu0.log ...     # 每 worker 日志
    └── driver_logs/               # （在 OUT_BASE 根，非 group 下）launch_all_nodes 各节点 driver 日志
```

判完成：输入 shard `idx` 完成 ⟺ `progress/{idx:06d}.done` 存在 **且** 某个 `w*/shard-{idx:06d}-*.tar` 存在。

## 两个历史事故根因

### 问题1：输出只有 `shard-000000.tar` —— 重启截断丢数（Critical）

`webdataset_writer.py:36` `_open_new_shard()` 用 `tarfile.open(path, "w")` 打开 `w{wid}/shard-000000.tar`，**"w" 模式会截断**。`run_worker.py` 旧逻辑用**一个**贯穿所有输入 shard 的 `ShardWriter`，`shard_id` 永远从 0 起。worker 每次重启（OOM、节点被踢、`gg` 保活抢显存、两天里必然多次）：

1. `ShardWriter.__init__` → 截断 `shard-000000.tar`，**抹掉上次已写入的样本**；
2. `process_input_shard` 跳过所有有 `.done` 的输入 shard（这些样本不会重算）；
3. 净结果：被截断掉的样本永久丢失，且 `.done` 已存在不再补算。

两天 + 多次重启 ⇒ 每个 worker 只剩「最后一段未重启期间」的少量样本，且 200 样本/输出 shard 的阈值远未触达，所以恒为单个 `shard-000000.tar`。**这不是慢，是每次重启都在毁数据，续跑根本没生效。**

> `w000~w031`（仅 32 个而非 64）是问题2 的连带结果：`launch_all_nodes.sh` 预检「自动剔除坏点」把 CUDA 失效节点踢出，`NUM_NODES` 缩水、rank 重排稠密化 → 只生成存活节点对应的 worker 目录（4 节点存活 = w000~w031）。

**修复：** 输出 tar 改为按输入 shard 下标确定性命名 `shard-{input_idx:06d}-{part:06d}.tar`（每输入 shard 独立 writer，`prefix=f"shard-{idx:06d}"`），`.done` 与输出 tar 1:1。续跑判定升级为「`.done` 存在 **且** 对应 `shard-{idx:06d}-*.tar` 存在」才算完成——旧的全部 `.done`（共享命名，无 per-idx tar）会自动判为未完成并重算，**无需手动清空目录即可自愈两天的损坏产出**。

### 问题2：多节点恒有一台 CUDA 不可用 —— master 环境注入污染远程（Critical）

`launch_all_nodes.sh` 预检与拉起的 SSH 负载里：

```bash
export PATH="${PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
```

SSH 负载是**外层双引号字符串**，`${PATH:-}`/`${LD_LIBRARY_PATH:-}` 由 **master 本机** shell 先展开再发往远程 ⇒ 把 master 的 `LD_LIBRARY_PATH`（可能含系统 Python3.12 torch，正是 `undefined symbol` 根因）强行 export 到远程，污染/抢在 env libtorch 之前。该值随「你从哪台机器发起」而变，表现就是「source 不生效 / 环境没完全覆盖 / 稳定有一台跟 master 环境不合的节点挂」。

此外预检只 `source activate_sana_wm.sh`，**没 source `config.sh`**，与真实生产路径（`run_groups_sequential.sh`→`config.sh`）环境不一致；`launch_single_node.sh` 还在热路径里 `pip install -e`，多节点并发对共享 `$PROJ_DIR` 装同一个 editable 包会竞态损坏 import。

**修复：** ① 删掉两处 master env 注入行；② 远程统一 `source config.sh`（唯一真源，含 activate + alloc + offline + 后续 OUT_BASE 覆盖）后 `unset CUDA_VISIBLE_DEVICES`；③ `pip install -e` 移到预检阶段**串行**每节点跑一次，从 worker 热路径删除。

## 5 个独立任务组启动模板

每个任务组 = 一个数据集 + 一份独立 hostfile + 一个独立输出路径（彼此隔离，无 worker 目录写冲突）：

```bash
cd /root/work/david_work/sana_wm_pipeline
BP=experiments/batch_production
OUT=/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output

# 组1
bash $BP/launch_all_nodes.sh wds-sekai-real-walking-hq $OUT/final_sekai_realwalking_hq  /path/hostfile_group1
# 组2
bash $BP/launch_all_nodes.sh wds-DL3DV-ALL-2K          $OUT/final_dl3dv_all_2k          /path/hostfile_group2
# 组3
bash $BP/launch_all_nodes.sh wds-SpatialVID-hq         $OUT/final_spatialvid_hq         /path/hostfile_group3
# 组4 / 组5：同理换 <DATASET> <OUT_PATH> <HOSTFILE>

# 先只预检不拉起：加 --check-only
bash $BP/launch_all_nodes.sh --check-only wds-sekai-real-walking-hq $OUT/final_sekai_realwalking_hq /path/hostfile_group1

# 监控 / 停止
bash $BP/watch_progress.sh wds-sekai-real-walking-hq $OUT/final_sekai_realwalking_hq
bash $BP/stop_all_nodes.sh /path/hostfile_group1                      # 停单组
bash $BP/stop_all_nodes.sh /path/hostfile_group1 /path/hostfile_group2 ...  # 一键停多组
```

## 断点续跑

任意中断后**重跑同一条命令即可**：worker 按输入 shard 下标判定，
`progress/{idx}.done` + 对应 `shard-{idx}-*.tar` 同时存在才跳过，否则重算。

## 从旧版本迁移（重要）

旧脚本因「共享 shard-000000.tar 截断」bug 产出的输出**不可信**。新命名
（`shard-{idx}-*.tar`）会让旧的全部 `.done` 自动判为未完成并重算，**无需手动删目录**；
但旧的 `w*/shard-000000.tar` 残留文件建议手动清掉以免与新文件混淆：

```bash
find $OUT_BASE/<group> -name 'shard-000000.tar' -path '*/w*' -delete   # 仅删旧共享命名残留
```
