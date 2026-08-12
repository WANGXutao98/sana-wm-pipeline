# 鲁棒的 Tar 批量解压系统设计文档

**日期**: 2026-08-06  
**项目**: SANA-WM Stage 3 数据准备  
**目标**: 从 3TB 损坏 tar 集合中最大化恢复样本

---

## 1. 背景与目标

### 数据现状
- **总数据量**: ~3TB
- **样本总数**: 282,222 个
- **存储位置**: `/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output`
- **可用空间**: 20TB
- **样本结构**: 每个样本包含 5 个文件
  - `{sample_id}.mp4` - 视频文件
  - `{sample_id}.caption.txt` - 文本描述
  - `{sample_id}.poses_c2w.npy` - 相机外参
  - `{sample_id}.intrinsics.npy` - 相机内参
  - `{sample_id}.scale.npy` - 缩放参数

### 问题描述
- 部分 tar 文件损坏（truncated, unexpected end of data）
- 部分 tar 文件为空
- 部分 tar 文件无法打开
- **关键约束**: 损坏的 tar 中，绝大部分样本仍然可读，不能随意丢弃

### 核心目标
1. **最大化样本恢复**: 从所有 tar 文件中提取所有可读样本
2. **性能要求**: 12-24 小时内完成 3TB 数据解压
3. **断点恢复**: 支持中断后继续执行
4. **零维护**: 无需人工干预，自动处理各种损坏场景

---

## 2. 整体架构

### 核心策略
1. **激进容错提取**: 使用 `tar --ignore-failed-read` + `dd` 修复
2. **并行处理**: 16 个进程并行解压不同的 tar 文件
3. **断点恢复**: 通过标记文件跳过已完成的 tar
4. **进程隔离**: 每个 tar 独立处理，互不影响

### 目录结构
```
/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/
├── final_wds-sekai-game-drone/
│   └── wds-sekai-game-drone/
│       └── w003/
│           ├── shard-000003-000001.tar          # 原 tar 文件（保留）
│           ├── shard-000003-000001/             # 解压目录
│           │   ├── sample1.mp4
│           │   ├── sample1.caption.txt
│           │   ├── sample1.poses_c2w.npy
│           │   ├── sample1.intrinsics.npy
│           │   ├── sample1.scale.npy
│           │   └── ...
│           ├── shard-000003-000001.SUCCESS      # 成功标记
│           └── shard-000003-000002.tar
```

**设计理由**:
- 解压目录与 tar 文件同级，便于定位和追溯
- 保留原 tar 文件，以防需要重新处理
- 标记文件（`.SUCCESS` / `.FAILED`）用于断点恢复

---

## 3. 核心提取流程

### 单个 tar 文件的处理流程

```
输入: tar_file_path (e.g., /path/to/shard-000003-000001.tar)

1. 断点恢复检查
   ├─ 如果 {tar_name}.SUCCESS 存在
   │  └─ 跳过此 tar（已完成）
   └─ 如果解压目录存在但无 .SUCCESS
      └─ 删除解压目录（部分失败，需重新解压）

2. 创建解压目录
   extracted_dir = {tar_name}/  # shard-000003-000001/
   mkdir -p extracted_dir

3. 第一次尝试：标准容错提取
   tar -xf {tar_file} \
       --ignore-failed-read \        # 遇到损坏文件继续
       --warning=no-timestamp \       # 抑制时间戳警告
       -C {extracted_dir} \
       2>/dev/null                    # 抑制错误输出

   退出码 = 0 或部分成功 → 跳到步骤 5

4. 第二次尝试：dd 修复 + 容错提取
   如果步骤 3 失败：
   
   dd if={tar_file} bs=512 conv=noerror,sync 2>/dev/null | \
   tar -x \
       --ignore-zeros \               # 跳过零填充块
       -C {extracted_dir} \
       2>/dev/null

5. 验证结果
   如果 extracted_dir 包含任何文件：
      └─ touch {tar_name}.SUCCESS
   否则：
      ├─ touch {tar_name}.FAILED
      └─ rmdir {extracted_dir}      # 删除空目录

输出: 
  - 成功：解压目录 + .SUCCESS 标记
  - 失败：.FAILED 标记
```

### 关键命令说明

**tar 容错参数**:
- `--ignore-failed-read`: 遇到无法读取的文件时继续处理后续文件
- `--warning=no-timestamp`: 抑制时间戳相关警告
- `--ignore-zeros`: 忽略 tar 文件中的零块（dd 修复可能产生）

**dd 修复参数**:
- `bs=512`: tar 文件的块大小为 512 字节
- `conv=noerror`: 遇到读取错误时不停止
- `conv=sync`: 读取错误时用零填充，保持块对齐

---

## 4. 并行控制与监控

### 并行机制

**方案 A（推荐）: 使用 GNU parallel**
```bash
find {base_dir} -name "*.tar" -type f | \
  grep -v -F -f <(find {base_dir} -name "*.SUCCESS" | sed 's/.SUCCESS$/.tar/') | \
  parallel -j 16 --halt never extract_single_tar {}
```

**方案 B（备选）: 使用 xargs**
```bash
find {base_dir} -name "*.tar" -type f | \
  grep -v -F -f <(find {base_dir} -name "*.SUCCESS" | sed 's/.SUCCESS$/.tar/') | \
  xargs -P 16 -I {} bash extract_single_tar.sh {}
```

**并行度**: 16 进程
- 理由：充分利用 CPU（192 核），避免网络存储 I/O 饱和
- 可通过命令行参数调整

### 断点恢复机制

**工作队列生成**:
1. 扫描所有 tar 文件
2. 过滤掉已有 `.SUCCESS` 标记的 tar
3. 剩余的 tar 构成待处理队列

**恢复行为**:
- 重新运行脚本 → 自动跳过已完成的 tar
- 部分完成的 tar（有解压目录但无 `.SUCCESS`）→ 删除目录重新解压

### 监控机制

**主日志** (`extraction.log`):
```
开始时间: 2026-08-06 10:00:00
总 tar 数: 1234
待处理数: 456
并行度: 16
预计完成时间: 2026-08-06 22:00:00
```

**实时进度查询**:
```bash
# 查看已完成数量
watch -n 60 'find {base_dir} -name "*.SUCCESS" | wc -l'

# 查看失败数量
find {base_dir} -name "*.FAILED" | wc -l

# 预估剩余时间
# (总数 - 完成数) / (完成数 / 经过小时数)
```

**异常处理**:
- 单个进程崩溃 → 其他进程继续
- 磁盘满 → 自动停止，清理后从断点恢复
- Ctrl+C 中断 → 重新运行自动恢复

---

## 5. 错误恢复与边界情况

### 特殊场景处理

| 场景 | 检测方法 | 处理策略 |
|------|---------|---------|
| **完全损坏的 tar** | 两次尝试都失败 | 标记 `.FAILED`，不阻塞其他 tar |
| **部分提取成功** | 解压目录有文件但 tar 返回错误 | 标记 `.SUCCESS`（部分恢复优于丢弃）|
| **空 tar 文件** | tar 成功但解压目录为空 | 标记 `.FAILED`，删除空目录 |
| **文件名冲突** | 同一 tar 中重复 sample_id | tar 自动覆盖（保留最后一个）|
| **磁盘空间耗尽** | tar 命令失败，无 `.SUCCESS` | 重新运行时从失败点继续 |
| **权限问题** | 无法创建解压目录 | 记录到 `permission_errors.log`，跳过 |

### 清理策略

**保留**:
- 所有原 tar 文件（不删除，便于回溯）
- 所有解压目录（即使部分损坏，最大化保留样本）
- 所有标记文件（`.SUCCESS` / `.FAILED`）

**删除**:
- 完全空的解压目录（tar 提取失败且无任何文件）

### 鲁棒性保证

1. **幂等性**: 多次运行脚本结果一致（已完成的不重复处理）
2. **原子性**: 标记文件在所有操作完成后创建（避免部分完成误判为成功）
3. **隔离性**: 每个 tar 独立处理，失败不影响其他 tar
4. **可观测性**: 标记文件提供明确的处理状态

---

## 6. 脚本接口

### 主脚本

**命令**: `extract_all_tars.sh`

**用法**:
```bash
# 标准运行（默认并行度 16）
./extract_all_tars.sh

# 自定义并行度
./extract_all_tars.sh --parallel 8

# 预览模式（不执行，只列出待处理的 tar）
./extract_all_tars.sh --dry-run

# 强制重新解压（忽略 .SUCCESS 标记）
./extract_all_tars.sh --force

# 指定 tar 根目录（默认 /root/work/filestorage/.../jdvbbfb_output）
./extract_all_tars.sh --base-dir /custom/path
```

**输出文件**:
- `extraction.log`: 主日志（开始时间、总数、进度）
- `{tar_name}.SUCCESS`: 成功标记（空文件）
- `{tar_name}.FAILED`: 失败标记（空文件）
- `permission_errors.log`: 权限错误列表（如果有）

### 验证脚本（可选）

**命令**: `verify_extraction.sh`

**用法**:
```bash
# 统计样本完整性
./verify_extraction.sh

# 输出示例：
# 总样本数: 280,145
# 完整样本 (5 文件): 275,890 (98.5%)
# 部分样本 (1-4 文件): 4,255 (1.5%)
# 生成详细报告: sample_completeness.csv
```

---

## 7. 性能估算

### 预期指标

**处理速度**:
- 单个 tar 文件: 平均 30-60 秒（取决于大小和损坏程度）
- 并行 16 进程: 每分钟处理 16-32 个 tar
- 总时长: 12-24 小时（假设 1000-2000 个 tar 文件）

**资源消耗**:
- CPU: 中等（主要是 I/O 等待）
- 内存: 低（每个进程 < 100MB）
- 磁盘 I/O: 高（网络存储瓶颈）
- 磁盘空间: 3TB（解压后与原 tar 大小相当）

### 优化空间

**如果速度不满意**:
1. 增加并行度到 24-32（需监控网络存储性能）
2. 使用本地 SSD 作为临时解压目录，完成后移动到网络存储
3. 分批处理：先处理关键数据集（如 sekai-game-drone）

---

## 8. 使用流程

### 执行步骤

```bash
# 1. 上传脚本到 CMCC 机器
scp extract_all_tars.sh user@cmcc:/root/work/

# 2. 授予执行权限
chmod +x extract_all_tars.sh

# 3. 预览待处理的 tar 列表
./extract_all_tars.sh --dry-run | head -20

# 4. 启动解压（建议在 tmux/screen 中运行）
tmux new -s tar_extraction
./extract_all_tars.sh

# 5. 监控进度（另开终端）
watch -n 60 'find /root/work/filestorage -name "*.SUCCESS" | wc -l'

# 6. 中断恢复（如需要）
# Ctrl+C 中断，然后重新运行
./extract_all_tars.sh  # 自动跳过已完成的

# 7. 验证结果（可选）
./verify_extraction.sh
```

### 故障排查

**问题**: 进程长时间无进展
- **检查**: `ps aux | grep tar` 查看是否有卡住的进程
- **解决**: `kill` 卡住的进程，脚本会自动跳过已完成的

**问题**: 磁盘空间不足
- **检查**: `df -h /root/work/filestorage`
- **解决**: 清理其他临时文件，或暂停部分进程

**问题**: 大量 tar 标记为 FAILED
- **检查**: `cat {tar_name}.FAILED` 附近的日志
- **解决**: 可能是原 tar 文件本身完全损坏，无法恢复

---

## 9. 与 Stage 3 集成

### 代码适配

**stage3_gpu.py** 已更新为支持解压目录：

```python
# 第 65-86 行
tar_path = Path(tar_path)
extracted_dir = tar_path.parent / tar_path.stem  # shard-xxx.tar -> shard-xxx/

if extracted_dir.exists():
    # 从解压目录读取（快速）
    mp4_path = extracted_dir / f"{sample_id}.mp4"
    cap_path = extracted_dir / f"{sample_id}.caption.txt"
    mp4_bytes = mp4_path.read_bytes()
    cap_bytes = cap_path.read_bytes()
else:
    # 回退到 tar 文件（兼容性）
    with tarfile.open(tar_path, "r") as tf:
        mp4_bytes = tf.extractfile(tf.getmember(f"{sample_id}.mp4")).read()
        cap_bytes = tf.extractfile(tf.getmember(f"{sample_id}.caption.txt")).read()
```

**性能提升**:
- 从 tar 读取: 数分钟/样本（线性扫描）
- 从解压目录读取: <10ms/样本（直接文件系统）
- **加速比**: 1000x+

### 执行顺序

1. **先解压 tar 文件**（本设计）
   - 运行 `extract_all_tars.sh`
   - 预计 12-24 小时

2. **验证解压结果**（可选）
   - 运行 `verify_extraction.sh`
   - 确认样本完整性

3. **运行 Stage 3 冒烟测试**
   - 使用已解压的数据
   - 预计 18-20 分钟（901 样本）

4. **运行 Stage 3 全量处理**
   - 16 GPU 并行
   - 预计 9 小时（282K 样本）

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **网络存储性能瓶颈** | 解压时间超过 24 小时 | 降低并行度到 8-12，或分批处理 |
| **磁盘空间不足** | 解压中断 | 预留 5TB 缓冲空间，监控磁盘使用 |
| **大量 tar 完全损坏** | 样本恢复率低于预期 | 记录 `.FAILED` tar，考虑从源重新传输 |
| **权限问题** | 无法解压到目标目录 | 检查目录权限，必要时用 sudo 运行 |
| **进程意外终止** | 部分 tar 未处理 | 断点恢复机制自动处理 |

---

## 11. 总结

**设计特点**:
- ✅ **最大化样本恢复**: 激进容错策略 + dd 修复
- ✅ **高性能**: 16 并行，12-24 小时完成 3TB
- ✅ **鲁棒性**: 断点恢复、异常隔离、幂等性
- ✅ **零维护**: 自动处理各种损坏场景
- ✅ **可观测**: 标记文件、实时进度、验证脚本

**预期结果**:
- 280K+ 样本成功恢复（>98%）
- Stage 3 处理速度提升 1000 倍
- 为后续数据处理提供稳定基础

**下一步**:
- 编写 `extract_all_tars.sh` 实现脚本
- 编写 `verify_extraction.sh` 验证脚本
- 在 CMCC 机器上执行并监控
