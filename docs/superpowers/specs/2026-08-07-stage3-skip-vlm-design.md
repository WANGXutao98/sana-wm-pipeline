# Stage 3 跳过 VLM 功能设计

> **目标**：快速验证 UniMatch + DOVER 模块，暂时跳过 Qwen VLM  
> **日期**：2026-08-07  
> **状态**：已批准

---

## 背景

### 当前问题

在 CMCC 机器上运行 Stage 3 时发现：
- ✅ UniMatch（光流检测）正常运行
- ✅ DOVER（质量评分）正常运行
- ❌ Qwen VLM 报错：`The following model_kwargs are not used by the model: ['mm_token_type_ids', 'pixel_values', 'image_grid_thw']`
- ❌ GPU 利用率为 0
- ❌ 处理速度慢 54 倍（109 秒/样本 vs 预期 2 秒/样本）

### 优先级

需要先验证 UniMatch + DOVER 是否完全正常，再处理 Qwen 问题。这样可以：
1. 确认数据读取流程正确
2. 验证 GPU 加速有效
3. 隔离 Qwen 问题
4. 不阻塞 Stage 3 其他部分的验证

---

## 设计方案

### 核心思路

在现有代码中添加 `--skip-vlm` 参数，当启用时：
- ✅ 正常执行 UniMatch 和 DOVER
- ❌ 跳过 Qwen VLM 调用
- ✅ VLM 相关字段填充 `null`
- ✅ 其他逻辑保持不变

**优点**：
- 最小改动（~15 行代码）
- 立即可用
- 修复 Qwen 后无缝切换回完整模式

---

## 架构

### 数据流

```
Stage 1+2 JSONL (--stage12-jsonl)
    ↓
run_stage3_cmcc.py --skip-vlm
    ↓
加载模型：
  ├─ UniMatch ✓
  ├─ DOVER ✓
  └─ Qwen ✓ (仍然加载，但不调用)
    ↓
处理每个样本：
  ├─ 读取视频和 caption
  ├─ UniMatch 光流检测 ✓
  ├─ DOVER 质量评分 ✓
  └─ Qwen VLM ✗ (if skip_vlm)
    ↓
写入 JSONL：
{
  "sample_id": "...",
  "stage3": {
    "unimatch_flow": 82.446,
    "dover": 0.456,
    "vlm_entity_count": null,    ← 跳过时填 null
    "vlm_quality": null,
    "caption_revised": null,
    "table6_accepted": false,     ← VLM 条件失败
    "reasons": [...]
  }
}
```

---

## 实现细节

### 修改点 1：`scripts/run_stage3_cmcc.py`

#### 1.1 参数重命名

**当前**：
```python
p.add_argument("--stage1-jsonl", type=Path, required=True, help="Stage 1+2 结果 JSONL")
```

**修改后**：
```python
p.add_argument("--stage12-jsonl", type=Path, required=True, help="Stage 1+2 联合结果 JSONL")
# 保留别名以兼容旧脚本
p.add_argument("--stage1-jsonl", dest="stage12_jsonl", type=Path, help="(已废弃，请使用 --stage12-jsonl)")
```

#### 1.2 添加 skip_vlm 参数

```python
p.add_argument("--skip-vlm", action="store_true", help="跳过 Qwen VLM 调用（仅运行 UniMatch + DOVER）")
```

#### 1.3 传递给 process_sample_stage3

```python
s3_rec = process_sample_with_index(
    sample_id=sid,
    group_name=rec.get("group", ""),
    sample_index=sample_index,
    flow_fn=flow_fn,
    dover_fn=dover_fn,
    vlm_call=vlm_call,
    table6_cfg=table6_cfg,
    has_camera_words=has_cw,
    fallback_tar_path=fallback_tar,
    skip_vlm=args.skip_vlm,  # ← 新增
)
```

### 修改点 2：`src/sana_wm_pipeline/qc/stage3_gpu.py`

#### 2.1 函数签名

**当前**：
```python
def process_sample_stage3(
    sample_id: str, tar_path: Path, group_name: str,
    flow_fn: Callable, dover_fn: Callable, vlm_call: Callable,
    table6_cfg: dict, has_camera_words: bool = False,
) -> dict:
```

**修改后**：
```python
def process_sample_stage3(
    sample_id: str, tar_path: Path, group_name: str,
    flow_fn: Callable, dover_fn: Callable, vlm_call: Callable,
    table6_cfg: dict, has_camera_words: bool = False,
    skip_vlm: bool = False,  # ← 新增
) -> dict:
```

#### 2.2 VLM 调用逻辑

**当前**（第 110-138 行）：
```python
# Check if VLM is needed for this source
need_vlm = False
if cfg.table6_source is not None:
    source_cfg = table6_cfg.get("per_source", {}).get(cfg.table6_source, {})
    need_vlm = (source_cfg.get("entity_count") or source_cfg.get("vlm_quality"))

if need_vlm:
    try:
        prompt = ENTITY_QUALITY_PROMPT
        if has_camera_words:
            prompt = prompt + _CAPTION_REWRITE_SUFFIX + f"\n\nCaption: {caption_text}"
        keyframes = [frames_rgb[i] for i in np.linspace(0, len(frames_rgb) - 1, 8).astype(int)]
        raw = vlm_call(prompt, keyframes)
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        # ... 解析结果
    except Exception as e:
        stage3["reasons"].append(f"vlm_error: {e}")
```

**修改后**：
```python
# Check if VLM is needed for this source
need_vlm = False
if cfg.table6_source is not None:
    source_cfg = table6_cfg.get("per_source", {}).get(cfg.table6_source, {})
    need_vlm = (source_cfg.get("entity_count") or source_cfg.get("vlm_quality"))

# ← 新增：检查是否跳过 VLM
if need_vlm and not skip_vlm:
    try:
        prompt = ENTITY_QUALITY_PROMPT
        if has_camera_words:
            prompt = prompt + _CAPTION_REWRITE_SUFFIX + f"\n\nCaption: {caption_text}"
        keyframes = [frames_rgb[i] for i in np.linspace(0, len(frames_rgb) - 1, 8).astype(int)]
        raw = vlm_call(prompt, keyframes)
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        # ... 解析结果
    except Exception as e:
        stage3["reasons"].append(f"vlm_error: {e}")
# ← 新增：如果跳过 VLM，无需额外处理（字段已初始化为 None）
```

---

## 错误处理

### UniMatch 或 DOVER 失败

**行为**：与当前一致
- 捕获异常
- 记录到 `stage3["reasons"]`
- 继续处理其他模块

### VLM 被跳过时的 table6 判定

**预期行为**：
- `vlm_entity_count` 和 `vlm_quality` 要求会失败
- `reasons` 中会包含：
  ```
  "vlm_entity_count=None not in [0, 10]"
  "vlm_quality=None not in [0.5, 1.5]"
  ```
- `table6_accepted` 为 `false`

**这是预期行为**，因为我们暂时跳过了 VLM。

### 输出一致性

- 无论是否跳过 VLM，输出 JSONL 格式完全相同
- 只是 VLM 相关字段为 `null`
- 便于后续对比和数据合并

---

## 边界情况

### 场景 1：has_camera_words=true 的样本

**当前逻辑**：需要 VLM 改写 caption  
**跳过时**：`caption_revised` 为 `null`  
**影响**：不影响 UniMatch 和 DOVER 的执行

### 场景 2：混合模式执行

**描述**：部分 worker 跳过 VLM，部分不跳过  
**支持**：✅ 技术上可行  
**推荐**：❌ 不推荐（易混淆）

### 场景 3：后续修复 Qwen 重新运行

**操作**：
1. 去掉 `--skip-vlm` 参数
2. 对同一批样本重新运行
3. 合并 VLM 相关字段

**数据合并脚本**（后续可提供）：
```python
# 伪代码
for sample in results_with_vlm:
    if sample_id in results_without_vlm:
        merge_vlm_fields(sample, results_without_vlm[sample_id])
```

---

## 测试策略

### 单元测试（可选）

- 验证 `skip_vlm=True` 时 VLM 不被调用
- 验证输出字段为 `null`

### 集成测试（必需）

#### 测试 1：冒烟测试（10 个样本）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3_cmcc.py \
  --stage12-jsonl /path/to/10_samples.jsonl \
  --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
  --output-dir /tmp/stage3_smoke_skip_vlm \
  --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml \
  --skip-vlm
```

**验证点**：
- ✅ 运行完成无崩溃
- ✅ GPU 利用率 > 50%
- ✅ 单样本处理时间 < 3 秒
- ✅ `unimatch_flow` 有值
- ✅ `dover` 有值
- ✅ `vlm_entity_count`, `vlm_quality`, `caption_revised` 为 `null`

#### 测试 2：小批量测试（100 个样本，2 GPU）

```bash
for worker_id in 0 1; do
    CUDA_VISIBLE_DEVICES=$worker_id nohup python scripts/run_stage3_cmcc.py \
      --stage12-jsonl /path/to/100_samples.jsonl \
      --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
      --output-dir /tmp/stage3_batch_skip_vlm \
      --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
      --unimatch-dir /root/work/david_work/models/unimatch \
      --worker-id $worker_id \
      --total-workers 2 \
      --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml \
      --skip-vlm \
      > /tmp/worker${worker_id}.log 2>&1 &
done
```

**验证点**：
- ✅ 并行执行正常
- ✅ 两个 worker 处理的样本数相近
- ✅ 无资源竞争或死锁

---

## 验收标准

### 功能性

- ✅ 添加 `--skip-vlm` 参数成功
- ✅ 参数重命名为 `--stage12-jsonl`（保留 `--stage1-jsonl` 别名）
- ✅ UniMatch 正常运行（有 `unimatch_flow` 值）
- ✅ DOVER 正常运行（有 `dover` 值，且在 [0, 1] 范围）
- ✅ VLM 字段为 `null`（当 `--skip-vlm` 时）
- ✅ 输出 JSONL 格式正确

### 性能

- ✅ GPU 利用率 > 50%（说明模型在 GPU 上运行）
- ✅ 单样本处理时间 < 3 秒（目标 ~2 秒）
- ✅ 相比当前速度提升 > 30x（从 109 秒降到 3 秒）

### 稳定性

- ✅ 无崩溃或死锁
- ✅ 错误信息清晰（如果有）
- ✅ 48 GPU 并行稳定

---

## 实施计划

### 阶段 1：代码修改（15 分钟）

1. 修改 `scripts/run_stage3_cmcc.py`
   - 重命名参数
   - 添加 `--skip-vlm`
   - 传递参数

2. 修改 `src/sana_wm_pipeline/qc/stage3_gpu.py`
   - 添加 `skip_vlm` 参数
   - 修改 VLM 调用逻辑

### 阶段 2：测试验证（10 分钟）

1. 冒烟测试（10 个样本）
2. 检查输出格式
3. 验证性能

### 阶段 3：部署（5 分钟）

1. 拷贝修改后的文件到 CMCC
2. 运行完整测试

**总计**：30 分钟

---

## 后续工作

### 短期（修复 Qwen）

1. 调查 Qwen VLM 报错原因
2. 修复后去掉 `--skip-vlm`，完整运行 Stage 3
3. 合并带 VLM 和不带 VLM 的结果

### 长期（可选优化）

1. **条件加载**：`--skip-vlm` 时不加载 Qwen 模型（节省 17 GB GPU 内存）
2. **分离脚本**：如果经常需要分阶段运行，考虑创建独立的 Stage 3a/3b 脚本

---

## 回滚计划

如果测试失败：
- 代码改动很小，直接 `git revert`
- 或使用 Git stash 临时保存
- 保留旧版 `run_stage3_cmcc.py` 作为 `.backup`

---

## 附录

### 命令对比

**旧命令**：
```bash
python scripts/run_stage3_cmcc.py \
  --stage1-jsonl /path/to/manifest.jsonl \
  --output-dir /path/to/output \
  --qwen-dir /path/to/Qwen3.5-9B \
  --unimatch-dir /path/to/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml
```

**新命令（推荐）**：
```bash
python scripts/run_stage3_cmcc.py \
  --stage12-jsonl /path/to/manifest.jsonl \
  --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
  --output-dir /path/to/output \
  --qwen-dir /path/to/Qwen3.5-9B \
  --unimatch-dir /path/to/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml \
  --skip-vlm
```

### 输出示例

**跳过 VLM 时**：
```json
{
  "sample_id": "sekai-game-drone_00400110001_0006450_0006750",
  "stage3": {
    "unimatch_flow": 82.446,
    "dover": 0.456,
    "vlm_entity_count": null,
    "vlm_quality": null,
    "table6_accepted": false,
    "caption_revised": null,
    "reasons": [
      "vlm_entity_count=None not in [0, 10]",
      "vlm_quality=None not in [0.5, 1.5]"
    ]
  }
}
```

**完整运行时**：
```json
{
  "sample_id": "sekai-game-drone_00400110001_0006450_0006750",
  "stage3": {
    "unimatch_flow": 82.446,
    "dover": 0.456,
    "vlm_entity_count": 3,
    "vlm_quality": 0.8,
    "table6_accepted": true,
    "caption_revised": "A park with trees and people walking",
    "reasons": []
  }
}
```

---

**设计版本**：v1.0  
**批准日期**：2026-08-07  
**实施预计**：30 分钟
