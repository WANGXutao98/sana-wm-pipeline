# 分布式批处理任务启动失败诊断与修复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 诊断并修复 CMCC 多节点集群上批处理任务在 hostfile 更新后无法启动的问题

**Architecture:** 分布式批处理系统通过 SSH 多播启动远程节点上的 worker 进程，driver 日志写入共享存储目录。问题核心是输出路径权限/挂载状态异常导致启动失败。

**Tech Stack:** Bash, SSH, Linux 文件系统, CMCC 集群环境

## 全局约束

- 所有操作在目标机器（CMCC 集群节点）上执行，非本机
- 保持与之前成功运行时的环境一致性
- 不改变核心业务逻辑，仅修复启动问题
- 所有诊断命令必须可远程执行（SSH 非交互式）

---

## 问题分析

**症状总结**：
1. 预检通过（环境/GPU/数据完整）→ 启动阶段失败
2. 错误信息：`Read-only file system` 在写入 driver_logs 时
3. 本机 `/root/work/externalstorage` 不存在 → 这是目标机器路径
4. `watch_progress.sh` 显示旧日志 → 新任务根本未启动

**根因假设**：
- H1: 输出路径 `/root/work/externalstorage/...` 在目标机器上变为只读（存储故障/挂载问题）
- H2: 输出路径不存在，`mkdir -p` 在 launch_all_nodes.sh:51 失败但被 `||` 吞掉
- H3: hostfile 更新后节点 ID 改变，但输出路径仍指向旧路径
- H4: 权限问题（用户/组变更）

**关键发现**：launch_all_nodes.sh:50 在**本机**创建 `DRIVER_LOG_DIR="$OUT_BASE/driver_logs"`，但 line 163 在**远程机器** SSH 会话中使用 `$LOG`（指向该目录）。这是脚本设计缺陷。

**诊断策略**：
1. 验证 hostfile 内容和节点可达性
2. 检查目标机器上输出路径的存在性、权限、挂载状态
3. 验证预检阶段 `mkdir -p` 是否真正成功
4. 识别启动阶段和预检阶段的环境差异

---

### Task 1: 诊断环境和路径状态

**Files:**
- Read: `experiments/batch_production/hostfile_0630/hostfile`（或用户提供的实际路径）
- Execute: SSH 远程诊断命令

**Interfaces:**
- Consumes: hostfile 路径（用户提供）
- Produces: 诊断报告（节点状态、路径状态、挂载信息）

- [ ] **Step 1: 定位并读取 hostfile**

询问用户 hostfile 的完整路径，或在目标机器上查找：

```bash
# 如果在本机找不到，需要在目标机器上执行
find /mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/batch_production -name "hostfile" -type f 2>/dev/null
```

预期：获得 hostfile 完整路径和内容

- [ ] **Step 2: 验证 hostfile 中的节点可达性**

```bash
# 从 hostfile 提取节点 ID 并测试 SSH 连接
while read line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    NODE_ID=$(awk '{print $1}' <<< "$line")
    echo "测试节点: $NODE_ID"
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@"$NODE_ID" "hostname && date" 2>&1 | head -2
done < hostfile_path
```

预期：所有节点可达，返回主机名和时间

- [ ] **Step 3: 检查目标机器上输出路径状态**

在第一个可达节点上执行完整诊断：

```bash
NODE_ID=$(awk 'NR==1 && !/^[[:space:]]*#/ {print $1}' hostfile_path)
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@"$NODE_ID" '
    OUT_PATH="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-real-walking-hq"
    
    echo "=== 路径存在性检查 ==="
    ls -ld "$OUT_PATH" 2>&1
    ls -ld "$(dirname "$OUT_PATH")" 2>&1
    ls -ld "/root/work/externalstorage" 2>&1
    
    echo -e "\n=== 挂载点检查 ==="
    df -h "$OUT_PATH" 2>&1 || df -h /root/work 2>&1
    mount | grep -E "externalstorage|/root/work" || echo "无相关挂载点"
    
    echo -e "\n=== 可写性测试 ==="
    mkdir -p "$OUT_PATH/driver_logs" 2>&1 && echo "[OK] mkdir 成功" || echo "[FAIL] mkdir 失败"
    touch "$OUT_PATH/driver_logs/test_write_$$.tmp" 2>&1 && echo "[OK] touch 成功" || echo "[FAIL] touch 失败"
    rm -f "$OUT_PATH/driver_logs/test_write_$$.tmp" 2>/dev/null
    
    echo -e "\n=== 磁盘空间检查 ==="
    df -h "$OUT_PATH" 2>&1 | tail -1
'
```

预期输出：
- 路径存在且可写 → 问题在其他地方
- 路径不存在 → 需要创建或修复挂载
- Read-only file system → 存储故障，需要修复挂载或更换输出路径

- [ ] **Step 4: 对比预检和启动阶段的执行上下文**

提取 launch_all_nodes.sh 中预检（line 63-108）和启动（line 142-166）的差异：

```bash
# 预检阶段在 line 92 执行：
mkdir -p '$OUT_BASE' || { echo '[FAIL] OUT_BASE 不可写: $OUT_BASE'; exit 1; }

# 启动阶段在 line 163 执行：
nohup bash run_groups_sequential.sh ... > '$LOG' 2>&1
# 但 $LOG 在 line 130 定义：
LOG="$DRIVER_LOG_DIR/node${i}_driver.log"
# $DRIVER_LOG_DIR 在 line 50 创建（本地机器）：
mkdir -p "$DRIVER_LOG_DIR"  # 这是在 SSH 之前！
```

发现：**关键 bug - DRIVER_LOG_DIR 在本机创建，但 $LOG 路径在远程机器上使用**

预期：识别出脚本逻辑错误

- [ ] **Step 5: 验证预检阶段的 mkdir 是否真正成功**

在目标节点上手动重现预检阶段的 mkdir：

```bash
NODE_ID=$(awk 'NR==1 && !/^[[:space:]]*#/ {print $1}' hostfile_path)
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@"$NODE_ID" "
    set -e
    source /etc/profile 2>/dev/null || true
    source ~/.bashrc 2>/dev/null || true
    export OUT_BASE='/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-real-walking-hq'
    
    echo '预检阶段 mkdir 测试:'
    mkdir -p \"\$OUT_BASE\" && echo '[OK] mkdir 成功' || echo '[FAIL] mkdir 失败'
    
    echo '启动阶段 mkdir 测试:'
    DRIVER_LOG_DIR=\"\$OUT_BASE/driver_logs\"
    mkdir -p \"\$DRIVER_LOG_DIR\" && echo '[OK] mkdir driver_logs 成功' || echo '[FAIL] mkdir driver_logs 失败'
    
    echo '写入测试:'
    echo 'test' > \"\$DRIVER_LOG_DIR/test.log\" 2>&1 && echo '[OK] 写入成功' || echo '[FAIL] 写入失败'
" 2>&1
```

预期：定位是 mkdir 失败还是后续写入失败

---

### Task 2: 根因定位和临时解决方案

**Files:**
- Analyze: Task 1 的诊断输出

**Interfaces:**
- Consumes: Task 1 的诊断报告
- Produces: 根因分析和快速修复方案（workaround）

- [ ] **Step 1: 分析 Task 1 的诊断结果**

根据 Task 1 的输出，判断问题类别：

| 场景 | 根因 | 快速修复方案 |
|------|------|--------------|
| A. 路径不存在 | 挂载点丢失或路径配置错误 | 1) 验证挂载点 2) 手动创建路径 3) 或修改输出路径到可写位置 |
| B. Read-only | 存储挂载为只读（NFS/共享存储故障） | 1) 重新挂载 rw 2) 或临时切换到本地存储 |
| C. 权限不足 | 用户/组权限变更 | `chown -R` 修复权限 |
| D. 脚本 bug | `DRIVER_LOG_DIR` 在本机创建但远程使用 | 修改 launch_all_nodes.sh 在远程创建目录 |

预期：识别出具体场景

- [ ] **Step 2: 实施场景 A 的修复（路径不存在）**

如果路径不存在，在所有节点上创建：

```bash
while read line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    NODE_ID=$(awk '{print $1}' <<< "$line")
    echo "在 $NODE_ID 上创建输出路径..."
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@"$NODE_ID" "
        mkdir -p /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-real-walking-hq/{driver_logs,progress,logs}
        chmod 755 /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-real-walking-hq
    " 2>&1
done < hostfile_path
```

- [ ] **Step 3: 实施场景 B 的修复（Read-only）**

如果是只读挂载，尝试重新挂载或切换输出路径：

```bash
# 选项1: 重新挂载为读写（需要 root 权限和挂载点信息）
# ssh root@NODE "mount -o remount,rw /mount/point"

# 选项2: 切换到本地可写路径（临时方案）
NEW_OUT_BASE="/root/work/local_output/final_wds-sekai-real-walking-hq"
# 然后用 NEW_OUT_BASE 重新运行 launch_all_nodes.sh
```

- [ ] **Step 4: 实施场景 D 的修复（脚本 bug）**

修改 `launch_all_nodes.sh` 确保在远程节点上创建 driver_logs：

在 line 142-166 的 SSH 块中，`cd '$PROJ_DIR'` 之前添加：

```bash
mkdir -p '$DRIVER_LOG_DIR'
```

或者将 line 130 的 LOG 定义改为使用 OUT_BASE（远程路径）而非本地 DRIVER_LOG_DIR

- [ ] **Step 5: 验证修复后的启动流程**

使用 `--check-only` 验证预检通过，然后小规模测试启动：

```bash
# 1. 预检
bash launch_all_nodes.sh --check-only wds-sekai-real-walking-hq \
    /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-real-walking-hq \
    hostfile_path

# 2. 仅用第一个节点测试启动
head -1 hostfile_path > /tmp/hostfile_test
bash launch_all_nodes.sh wds-sekai-real-walking-hq \
    /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-real-walking-hq \
    /tmp/hostfile_test

# 3. 等待30秒后检查进程和日志
sleep 30
ssh root@"$(head -1 hostfile_path | awk '{print $1}')" "
    pgrep -f 'run_worker.py' || echo '进程未启动'
    tail -20 /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-real-walking-hq/driver_logs/node0_driver.log 2>&1
"
```

预期：进程启动，driver 日志正常写入

---

### Task 3: 永久性修复和防御性改进

**Files:**
- Modify: `experiments/batch_production/launch_all_nodes.sh`
- Create: `experiments/batch_production/diagnose_cluster.sh`（新增诊断工具）

**Interfaces:**
- Consumes: Task 2 的根因分析
- Produces: 修复后的脚本 + 诊断工具

- [ ] **Step 1: 修复 launch_all_nodes.sh 的 DRIVER_LOG_DIR 创建逻辑**

**问题**：line 50-51 在本机创建 `DRIVER_LOG_DIR`，但 line 130 的 `$LOG` 在远程使用

**修复方案**：在远程 SSH 会话中创建目录

```bash
# 在 line 142-166 的 SSH 块中，nohup 之前添加：
mkdir -p '$DRIVER_LOG_DIR'
```

完整修改：

```bash
# 在 launch_all_nodes.sh line 161 之前插入
        mkdir -p '$DRIVER_LOG_DIR'
        
        cd '$PROJ_DIR'
```

- [ ] **Step 2: 增强预检阶段的可写性验证**

在 line 92 的 mkdir 之后添加实际写入测试：

```bash
        mkdir -p '$OUT_BASE' || { echo '[FAIL] OUT_BASE 不可写: $OUT_BASE'; exit 1; }
        # ponytail: 新增写入测试，确保不是只读挂载
        touch '$OUT_BASE/.write_test_$$' 2>/dev/null || { echo '[FAIL] OUT_BASE 只读: $OUT_BASE'; exit 1; }
        rm -f '$OUT_BASE/.write_test_$$' 2>/dev/null
```

- [ ] **Step 3: 改进错误处理 - 捕获远程 mkdir 失败**

在 line 163 的 nohup 之前添加错误检查：

```bash
        mkdir -p '$DRIVER_LOG_DIR' || { echo '[FAIL] 无法创建 driver_logs: $DRIVER_LOG_DIR'; exit 1; }
```

- [ ] **Step 4: 创建独立的集群诊断工具**

```bash
# experiments/batch_production/diagnose_cluster.sh
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

HOSTFILE="${1:?用法: $0 <HOSTFILE> [OUT_PATH]}"
OUT_PATH="${2:-$OUT_BASE}"

echo "=== 集群诊断工具 ==="
echo "OUT_PATH: $OUT_PATH"

mapfile -t NODES < <(awk '!/^[[:space:]]*#/ && NF>0 {print $1}' "$HOSTFILE")

for NODE in "${NODES[@]}"; do
    echo -e "\n--- 节点: $NODE ---"
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@"$NODE" "
        echo '主机名: '\$(hostname)
        echo '时间: '\$(date)
        
        echo -e '\n路径检查:'
        ls -ld '$OUT_PATH' 2>&1 | head -1
        
        echo -e '\n挂载点:'
        df -h '$OUT_PATH' 2>&1 | tail -1
        
        echo -e '\n可写性测试:'
        mkdir -p '$OUT_PATH/driver_logs' 2>&1 && echo '[OK] mkdir' || echo '[FAIL] mkdir'
        touch '$OUT_PATH/.test_\$\$' 2>&1 && rm -f '$OUT_PATH/.test_\$\$' && echo '[OK] write' || echo '[FAIL] write'
        
        echo -e '\nGPU 状态:'
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>&1 | head -3
        
        echo -e '\n运行中的任务:'
        pgrep -f 'run_worker.py' | wc -l | xargs echo 'worker 进程数:'
    " 2>&1 || echo "[WARN] $NODE 无法访问"
done
```

赋予执行权限：

```bash
chmod +x experiments/batch_production/diagnose_cluster.sh
```

- [ ] **Step 5: 测试修复后的完整流程**

```bash
# 1. 运行诊断工具
bash experiments/batch_production/diagnose_cluster.sh hostfile_path "$OUT_PATH"

# 2. 清理旧进程（如果有）
bash experiments/batch_production/stop_all_nodes.sh hostfile_path

# 3. 完整启动测试
bash experiments/batch_production/launch_all_nodes.sh wds-sekai-real-walking-hq \
    /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-real-walking-hq \
    hostfile_path

# 4. 监控30秒
sleep 30
bash experiments/batch_production/watch_progress.sh wds-sekai-real-walking-hq \
    /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-real-walking-hq
```

预期：所有节点正常启动，GPU 显存占用，日志实时更新

- [ ] **Step 6: 提交修复**

```bash
git add experiments/batch_production/launch_all_nodes.sh \
        experiments/batch_production/diagnose_cluster.sh
git commit -m "fix(batch): 修复 driver_logs 在远程节点创建失败的问题

- 在远程 SSH 会话中创建 DRIVER_LOG_DIR，而非在本机创建
- 预检阶段增加写入测试，提前发现只读文件系统
- 新增 diagnose_cluster.sh 工具用于快速诊断集群状态

根因：launch_all_nodes.sh:50 在本机创建目录，但 :163 在远程使用该路径
症状：Read-only file system 错误在 nohup 重定向时触发

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

## 执行检查清单

在开始执行前确认：
- [ ] 已获取 hostfile 的完整路径
- [ ] 已确认输出路径 `/root/work/externalstorage/...` 是目标机器路径，非本机
- [ ] 已准备好在目标机器上执行 SSH 命令的权限
- [ ] 已备份当前 launch_all_nodes.sh（如有修改需求）

---

## 预期结果

**Task 1**: 诊断报告，明确根因（路径不存在/只读/权限/脚本bug）
**Task 2**: 临时修复，任务能够启动运行
**Task 3**: 永久修复，防止未来复现 + 新增诊断工具

**成功标准**：
- `launch_all_nodes.sh` 执行后，所有节点的 worker 进程启动
- GPU 显存占用正常
- `watch_progress.sh` 显示今天的日志（2026-06-30）
- driver_logs 目录可正常写入

---

## 风险和注意事项

1. **存储故障风险**：如果是 NFS/共享存储故障导致只读，需要联系运维修复挂载点
2. **数据一致性**：如果切换输出路径，确保新路径在所有节点间共享
3. **权限问题**：确认 SSH 使用的用户（root）与文件系统权限一致
4. **竞态条件**：多个节点同时创建同一目录，应使用 `mkdir -p`（已是现状）

**Ponytail 原则应用**：
- 先用 SSH 单行诊断命令定位问题（stdlib）
- 不引入新依赖，只修改必要的脚本行
- 诊断工具是独立脚本，可选使用
