# Stage 2 全量执行方案 - 解压数据版本

> **目的**：全量处理 Pass + Flag 样本，从解压目录直接读取文件，大幅提升速度

---

## 🎯 修改内容

### 1. 采样率调整
- **旧版本**：`SAMPLE_FRAC=0.05`（Flag 100% + Pass 5%）
- **新版本**：`SAMPLE_FRAC=1.0`（Flag 100% + Pass 100%）✅

### 2. 数据读取方式
- **旧版本**：从 tar 文件中读取（~100-200 ms/样本）
- **新版本**：从解压目录直接读取（~1-5 ms/样本）✅
- **加速比**：**20-200x**

### 3. 文件不存在处理
- **旧版本**：tar 文件损坏会导致错误
- **新版本**：文件不存在则跳过，不写入 Stage 2 结果 ✅

---

## 📂 文件清单

| 文件 | 路径 | 说明 |
|------|------|------|
| **核心逻辑** | `src/sana_wm_pipeline/qc/stage2_deep_extracted.py` | 从解压目录读取的 Stage 2 实现 |
| **执行脚本** | `scripts/run_stage2_full_extracted.sh` | Bash 批量执行脚本 |
| **使用说明** | `STAGE2_FULL_EXECUTION_GUIDE.md` | 本文档 |

---

## 🚀 使用方法

### 步骤 1：部署到 CMCC 机器

```bash
# 在本地机器（AFS）
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 打包新文件
tar czf stage2_full_extracted.tar.gz \
    src/sana_wm_pipeline/qc/stage2_deep_extracted.py \
    scripts/run_stage2_full_extracted.sh \
    STAGE2_FULL_EXECUTION_GUIDE.md

# 传输到 CMCC 机器
scp stage2_full_extracted.tar.gz user@cmcc:/root/work/david_work/
```

### 步骤 2：在 CMCC 机器上解压

```bash
cd /root/work/david_work/sana_wm_qc

# 解压文件
tar xzf ../stage2_full_extracted.tar.gz

# 验证文件
ls -l src/sana_wm_pipeline/qc/stage2_deep_extracted.py
ls -l scripts/run_stage2_full_extracted.sh

# 添加执行权限
chmod +x scripts/run_stage2_full_extracted.sh
```

### 步骤 3：单数据集测试（推荐）

```bash
# 激活环境
cd /root/work/david_work/sana_wm_qc
source /root/work/david_work/sana_wm_qc_env/bin/activate
export PYTHONPATH=/root/work/david_work/sana_wm_qc/src:$PYTHONPATH

# 测试单个数据集（用最小的数据集测试）
python << 'PYEOF'
from pathlib import Path
from sana_wm_pipeline.qc.stage2_deep_extracted import run_stage2_extracted

# 测试 wds-sekai-game-drone（最小数据集，931 样本）
s1_path = Path("/root/work/david_work/qc_output_new/wds-sekai-game-drone/stage1_results.jsonl")
s2_path = Path("/root/work/david_work/qc_output_new/wds-sekai-game-drone/stage2_results_test.jsonl")
data_root = Path("/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output")

print(f"测试配置:")
print(f"  Stage 1 输入: {s1_path}")
print(f"  Stage 2 输出: {s2_path}")
print(f"  数据根目录: {data_root}")
print(f"  采样率: 1.0 (全量)")
print()

n = run_stage2_extracted(
    s1_path, 
    s2_path, 
    data_root,
    sample_frac=1.0,  # 全量
    n_workers=8       # 测试用 8 进程
)

print(f"\n✅ 测试完成！处理了 {n} 个样本")
print(f"   结果文件: {s2_path}")
PYEOF

# 验证结果
cat /root/work/david_work/qc_output_new/wds-sekai-game-drone/stage2_results_test.jsonl | head -3
```

### 步骤 4：全量执行

```bash
# 确认测试无误后，执行全量
cd /root/work/david_work/sana_wm_qc

# 后台执行
nohup bash scripts/run_stage2_full_extracted.sh > /root/work/david_work/stage2_full_execution.log 2>&1 &

# 记录进程 ID
echo $! > /tmp/stage2_full.pid

# 查看实时日志
tail -f /root/work/david_work/stage2_full_execution.log
```

### 步骤 5：监控进度

```bash
# 方法 1：查看日志
tail -f /root/work/david_work/stage2_batch_full_*.log

# 方法 2：统计已处理样本数
find /root/work/david_work/qc_output_new -name "stage2_results_full.jsonl" -exec wc -l {} \; | awk '{s+=$1} END {print "已处理: " s " 个样本"}'

# 方法 3：查看各数据集进度
for group in wds-sekai-game-drone wds-sekai-game-walking wds-OmniWorld-Game wds-DL3DV-ALL-2K wds-sekai-real-walking-hq wds-RealEstate10K-360p wds-SpatialVID-hq; do
    file="/root/work/david_work/qc_output_new/$group/stage2_results_full.jsonl"
    if [ -f "$file" ]; then
        count=$(wc -l < "$file")
        echo "$group: $count"
    else
        echo "$group: 未开始"
    fi
done
```

---

## ⚙️ 配置参数

### 关键参数（在脚本中修改）

```bash
# 数据根目录（解压后的数据）
DATA_ROOT="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output"

# 输出目录
OUTPUT_ROOT="/root/work/david_work/qc_output_new"

# 并发进程数（根据 CPU 核心数调整）
N_WORKERS=32

# 采样率（1.0 = 全量）
SAMPLE_FRAC=1.0

# 数据集列表（可根据需要调整顺序或删除某些数据集）
DATASETS=(
    "wds-sekai-game-drone"         # 931 样本（最小，建议先测试）
    "wds-sekai-game-walking"       # 1,602
    "wds-OmniWorld-Game"           # 6,378
    "wds-DL3DV-ALL-2K"             # 9,937
    "wds-sekai-real-walking-hq"    # 20,154
    "wds-RealEstate10K-360p"       # 73,738
    "wds-SpatialVID-hq"            # 220,508（最大，耗时最长）
)
```

---

## 📊 预估性能

### 旧版本（tar 读取）
- **单样本耗时**：~100-200 ms
- **全量预估**（333,248 样本，32 进程）：
  - 333,248 × 0.15s / 32 = ~1,562 秒 ≈ **26 分钟**（理想情况）
  - 实际可能需要 **1-2 小时**（I/O 瓶颈）

### 新版本（解压目录读取）
- **单样本耗时**：~1-5 ms（文件查找） + PyAV 解码时间
- **全量预估**（333,248 样本，32 进程）：
  - 假设每样本 10ms（文件查找 + 解码）
  - 333,248 × 0.01s / 32 = ~104 秒 ≈ **2 分钟**（理想情况）
  - 实际可能需要 **5-10 分钟**（考虑 PyAV 解码时间）

### 实际测试结果（待验证）
| 数据集 | 样本数 | 预估耗时 | 实际耗时 |
|--------|--------|---------|---------|
| wds-sekai-game-drone | 931 | ~30 秒 | ？ |
| wds-sekai-game-walking | 1,602 | ~50 秒 | ？ |
| wds-OmniWorld-Game | 6,378 | ~3 分钟 | ？ |
| wds-DL3DV-ALL-2K | 9,937 | ~5 分钟 | ？ |
| wds-sekai-real-walking-hq | 20,154 | ~10 分钟 | ？ |
| wds-RealEstate10K-360p | 73,738 | ~35 分钟 | ？ |
| wds-SpatialVID-hq | 220,508 | ~1.5 小时 | ？ |
| **总计** | **333,248** | **~2.5 小时** | ？ |

---

## 🔍 关键代码变更

### 变更 1：文件查找逻辑

**旧版本**（tar 读取）：
```python
with tarfile.open(tar_path, "r") as tf:
    video_bytes = tf.extractfile(f"{sample_id}.mp4").read()
    poses = np.load(io.BytesIO(tf.extractfile(f"{sample_id}.poses_c2w.npy").read()))
```

**新版本**（解压目录读取）：
```python
# 查找样本文件
pattern = f"final_wds-*/wds-*/w*/shard-*/{sample_id}.mp4"
matches = list(data_root.glob(pattern))

if matches:
    mp4_path = matches[0]
    video_bytes = mp4_path.read_bytes()
    poses = np.load(mp4_path.with_suffix('.poses_c2w.npy'))
```

### 变更 2：跳过不存在的样本

**旧版本**：
```python
# 文件不存在会抛出 KeyError，记录到 reasons
stage2["reasons"].append("mp4_not_found")
return {"sample_id": sample_id, "stage2": stage2}
```

**新版本**：
```python
# 文件不存在直接返回 None，调用方跳过
files = find_sample_files(sample_id, data_root)
if files is None:
    return None  # 不写入 Stage 2 结果
```

### 变更 3：全量处理

**旧版本**：
```python
sample_frac = 0.05  # Flag 100% + Pass 5%
```

**新版本**：
```python
sample_frac = 1.0   # Flag 100% + Pass 100%
```

---

## 🐛 故障排查

### 问题 1：模块导入失败

**症状**：
```
[ERROR] Stage 2 解压版本模块导入失败
```

**解决**：
```bash
# 检查文件是否存在
ls -l /root/work/david_work/sana_wm_qc/src/sana_wm_pipeline/qc/stage2_deep_extracted.py

# 检查 PYTHONPATH
echo $PYTHONPATH

# 手动测试导入
python -c "from sana_wm_pipeline.qc.stage2_deep_extracted import run_stage2_extracted; print('OK')"
```

### 问题 2：数据根目录不存在

**症状**：
```
[ERROR] 数据根目录不存在
```

**解决**：
```bash
# 检查路径
ls -l /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output

# 检查解压目录
ls -l /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/final_wds-*
```

### 问题 3：找不到样本文件

**症状**：
```
[stage2] 已处理: 0, 已跳过: 1000
```

**原因**：
- 样本文件命名格式不匹配
- 解压目录结构不符合预期

**解决**：
```bash
# 检查实际文件命名
ls /root/work/filestorage/.../jdvbbfb_output/final_wds-SpatialVID-hq/wds-SpatialVID-hq/w000/shard-000000-000000/ | head -10

# 检查 sample_id 格式
grep -m 1 "sample_id" /root/work/david_work/qc_output_new/wds-SpatialVID-hq/stage1_results.jsonl
```

### 问题 4：进程卡住

**症状**：
- 长时间没有进度输出
- CPU 利用率很低

**解决**：
```bash
# 检查进程状态
ps aux | grep stage2

# 检查磁盘 I/O
iostat -x 5

# 如果卡住，终止并重启
kill $(cat /tmp/stage2_full.pid)
bash scripts/run_stage2_full_extracted.sh
```

---

## ✅ 验证清单

执行前检查：
- [ ] 文件已部署到 CMCC 机器
- [ ] 环境变量已设置（PYTHONPATH）
- [ ] 数据根目录存在且包含解压目录
- [ ] Stage 1 结果文件存在
- [ ] 已在最小数据集上测试通过

执行后验证：
- [ ] 7 个数据集的 `stage2_results_full.jsonl` 都已生成
- [ ] 样本数量符合预期（Pass + Flag）
- [ ] 日志中没有大量错误
- [ ] 跳过的样本数量合理（< 5%）

---

## 📋 输出文件

### Stage 2 结果

```
/root/work/david_work/qc_output_new/
├── wds-sekai-game-drone/
│   ├── stage1_results.jsonl                    # Stage 1 输入
│   ├── stage2_results_full.jsonl               # Stage 2 输出（新）✅
│   ├── stage2_results.jsonl                    # Stage 2 输出（旧，5% 采样）
│   └── stage2_run_full.log                     # 执行日志
├── wds-sekai-game-walking/
│   └── ...
├── wds-OmniWorld-Game/
│   └── ...
├── wds-DL3DV-ALL-2K/
│   └── ...
├── wds-sekai-real-walking-hq/
│   └── ...
├── wds-RealEstate10K-360p/
│   └── ...
└── wds-SpatialVID-hq/
    └── ...
```

### 日志文件

```
/root/work/david_work/
├── stage2_batch_full_20260807_140530.log      # 主日志
└── stage2_full_execution.log                  # nohup 日志
```

---

## 🎯 下一步

1. **测试**：在最小数据集上验证（wds-sekai-game-drone）
2. **全量执行**：确认无误后运行全部 7 个数据集
3. **生成 Stage 3 manifest**：使用 `stage2_results_full.jsonl` 作为输入
4. **进入 Stage 3**：GPU 密集型处理

---

**文档版本**：v1.0  
**创建时间**：2026-08-07  
**作者**：Claude Sonnet 4.6  
**状态**：✅ 就绪，可立即部署测试
