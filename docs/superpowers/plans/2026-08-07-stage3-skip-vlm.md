# Stage 3 跳过 VLM 功能实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 添加 `--skip-vlm` 参数以快速验证 UniMatch + DOVER，暂时跳过 Qwen VLM

**Architecture:** 在现有 Stage 3 代码中添加条件开关，当 `--skip-vlm` 启用时跳过 VLM 调用但保持其他逻辑不变

**Tech Stack:** Python 3.8+, argparse, pathlib

## Global Constraints

- Python 版本: ≥ 3.8
- 保持向后兼容: `--stage1-jsonl` 作为 `--stage12-jsonl` 的别名
- 输出格式不变: JSONL，VLM 字段为 `null` 时仍需包含
- 代码风格: 遵循现有代码格式（4 空格缩进，类型注解）

---

## File Structure

### 修改文件

**1. `scripts/run_stage3_cmcc.py`** (主执行脚本)
- 添加命令行参数: `--skip-vlm`, `--stage12-jsonl`
- 传递 `skip_vlm` 到 `process_sample_with_index()`

**2. `src/sana_wm_pipeline/qc/stage3_gpu.py`** (核心处理逻辑)
- 修改 `process_sample_stage3()` 函数签名，添加 `skip_vlm` 参数
- 在 VLM 调用前检查 `skip_vlm` 标志

### 创建文件

无新文件创建（仅修改现有文件）

---

## Task 1: 修改 run_stage3_cmcc.py 参数定义

**Files:**
- Modify: `scripts/run_stage3_cmcc.py:139-151`

**Interfaces:**
- Consumes: 无（入口点）
- Produces: 
  - `args.stage12_jsonl: Path` - Stage 1+2 结果 JSONL 路径
  - `args.skip_vlm: bool` - 是否跳过 VLM

- [ ] **Step 1: 添加参数重命名代码**

在 `run_stage3_cmcc.py` 的 `main()` 函数中，找到参数定义部分（约第 141 行），修改为：

```python
p.add_argument("--stage12-jsonl", type=Path, required=False, help="Stage 1+2 联合结果 JSONL")
p.add_argument("--stage1-jsonl", dest="stage12_jsonl", type=Path, required=False, help="(已废弃，请使用 --stage12-jsonl)")
```

- [ ] **Step 2: 添加 skip-vlm 参数**

在上述参数定义后，添加：

```python
p.add_argument("--skip-vlm", action="store_true", help="跳过 Qwen VLM 调用（仅运行 UniMatch + DOVER）")
```

- [ ] **Step 3: 添加参数验证**

在 `args = p.parse_args()` 之后，添加：

```python
# 确保至少提供了一个输入参数
if args.stage12_jsonl is None:
    p.error("--stage12-jsonl (或 --stage1-jsonl) 是必需的")
```

- [ ] **Step 4: 验证语法**

运行：
```bash
python -m py_compile scripts/run_stage3_cmcc.py
```

预期：无输出（编译成功）

- [ ] **Step 5: 测试参数解析**

运行：
```bash
python scripts/run_stage3_cmcc.py --help | grep -E "(stage12-jsonl|skip-vlm)"
```

预期：显示新参数的帮助信息

- [ ] **Step 6: 提交**

```bash
git add scripts/run_stage3_cmcc.py
git commit -m "feat(stage3): add --skip-vlm and rename --stage1-jsonl to --stage12-jsonl

- Add --skip-vlm parameter to skip Qwen VLM execution
- Rename --stage1-jsonl to --stage12-jsonl for clarity
- Keep --stage1-jsonl as deprecated alias for backward compatibility

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: 修改 stage3_gpu.py 函数签名

**Files:**
- Modify: `src/sana_wm_pipeline/qc/stage3_gpu.py:45-54`

**Interfaces:**
- Consumes:
  - 现有 `process_sample_stage3()` 函数的所有参数
- Produces:
  - `process_sample_stage3(skip_vlm: bool = False)` - 新增 skip_vlm 参数

- [ ] **Step 1: 修改函数签名**

在 `src/sana_wm_pipeline/qc/stage3_gpu.py` 第 45 行，修改函数签名：

```python
def process_sample_stage3(
    sample_id: str,
    tar_path: Path,
    group_name: str,
    flow_fn: Callable,
    dover_fn: Callable,
    vlm_call: Callable,
    table6_cfg: dict,
    has_camera_words: bool = False,
    skip_vlm: bool = False,  # ← 新增参数
) -> dict[str, Any]:
```

- [ ] **Step 2: 验证语法**

运行：
```bash
python -m py_compile src/sana_wm_pipeline/qc/stage3_gpu.py
```

预期：无输出（编译成功）

- [ ] **Step 3: 提交**

```bash
git add src/sana_wm_pipeline/qc/stage3_gpu.py
git commit -m "feat(stage3): add skip_vlm parameter to process_sample_stage3

- Add optional skip_vlm parameter (default False)
- Prepare for conditional VLM execution

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 修改 VLM 调用逻辑

**Files:**
- Modify: `src/sana_wm_pipeline/qc/stage3_gpu.py:110-139`

**Interfaces:**
- Consumes:
  - `skip_vlm: bool` from Task 2
  - 现有 VLM 调用逻辑
- Produces:
  - 条件执行的 VLM 逻辑（skip_vlm=True 时跳过）

- [ ] **Step 1: 找到 VLM 调用代码块**

在 `src/sana_wm_pipeline/qc/stage3_gpu.py` 约第 110-120 行，找到以下代码：

```python
# Check if VLM is needed for this source
need_vlm = False
if cfg.table6_source is not None:
    source_cfg = table6_cfg.get("per_source", {}).get(cfg.table6_source, {})
    need_vlm = (source_cfg.get("entity_count") or source_cfg.get("vlm_quality"))

if need_vlm:
    try:
        # ... VLM 调用逻辑
```

- [ ] **Step 2: 添加 skip_vlm 检查**

修改 `if need_vlm:` 为：

```python
if need_vlm and not skip_vlm:
    try:
        # ... VLM 调用逻辑保持不变
```

完整代码块应该是：

```python
# Check if VLM is needed for this source
need_vlm = False
if cfg.table6_source is not None:
    source_cfg = table6_cfg.get("per_source", {}).get(cfg.table6_source, {})
    need_vlm = (source_cfg.get("entity_count") or source_cfg.get("vlm_quality"))

if need_vlm and not skip_vlm:  # ← 添加 skip_vlm 检查
    try:
        prompt = ENTITY_QUALITY_PROMPT
        if has_camera_words:
            prompt = prompt + _CAPTION_REWRITE_SUFFIX + f"\n\nCaption: {caption_text}"
        keyframes = [frames_rgb[i] for i in np.linspace(0, len(frames_rgb) - 1, 8).astype(int)]
        raw = vlm_call(prompt, keyframes)
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        entity_count = (
            int(parsed.get("people", 0))
            + int(parsed.get("vehicles", 0))
            + int(parsed.get("animals", 0))
        )
        stage3["vlm_entity_count"] = entity_count
        stage3["vlm_quality"] = float(parsed.get("quality", -1.0))
        if has_camera_words and "caption_revised" in parsed:
            stage3["caption_revised"] = str(parsed["caption_revised"])
    except Exception as e:
        stage3["reasons"].append(f"vlm_error: {e}")
```

- [ ] **Step 3: 验证语法**

运行：
```bash
python -m py_compile src/sana_wm_pipeline/qc/stage3_gpu.py
```

预期：无输出（编译成功）

- [ ] **Step 4: 验证逻辑**

检查代码逻辑：
- `skip_vlm=False` 时，VLM 正常执行
- `skip_vlm=True` 时，跳过 VLM 块
- VLM 字段保持 `None`（已在第 58-62 行初始化）

- [ ] **Step 5: 提交**

```bash
git add src/sana_wm_pipeline/qc/stage3_gpu.py
git commit -m "feat(stage3): implement skip_vlm logic in process_sample_stage3

- Skip VLM execution when skip_vlm=True
- VLM fields remain None when skipped
- UniMatch and DOVER continue to execute normally

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 传递 skip_vlm 到处理函数

**Files:**
- Modify: `scripts/run_stage3_cmcc.py:73-110` (process_sample_with_index)
- Modify: `scripts/run_stage3_cmcc.py:207-217` (main loop)

**Interfaces:**
- Consumes:
  - `args.skip_vlm: bool` from Task 1
  - `process_sample_stage3(skip_vlm)` from Task 2-3
- Produces:
  - 完整的端到端 skip_vlm 数据流

- [ ] **Step 1: 修改 process_sample_with_index 函数签名**

在 `scripts/run_stage3_cmcc.py` 约第 73 行，修改函数签名：

```python
def process_sample_with_index(
    sample_id: str,
    group_name: str,
    sample_index: dict,
    flow_fn,
    dover_fn,
    vlm_call,
    table6_cfg: dict,
    has_camera_words: bool = False,
    fallback_tar_path: Path = None,
    skip_vlm: bool = False,  # ← 新增参数
) -> dict:
```

- [ ] **Step 2: 传递 skip_vlm 到 process_sample_stage3 (索引分支)**

在约第 97-108 行，找到索引分支的 `process_sample_stage3` 调用，添加 `skip_vlm` 参数：

```python
return process_sample_stage3(
    sample_id=sample_id,
    tar_path=fake_tar_path,
    group_name=group_name,
    flow_fn=flow_fn,
    dover_fn=dover_fn,
    vlm_call=vlm_call,
    table6_cfg=table6_cfg,
    has_camera_words=has_camera_words,
    skip_vlm=skip_vlm,  # ← 新增
)
```

- [ ] **Step 3: 传递 skip_vlm 到 process_sample_stage3 (回退分支)**

在约第 114-123 行，找到回退分支的 `process_sample_stage3` 调用，添加 `skip_vlm` 参数：

```python
return process_sample_stage3(
    sample_id=sample_id,
    tar_path=fallback_tar_path,
    group_name=group_name,
    flow_fn=flow_fn,
    dover_fn=dover_fn,
    vlm_call=vlm_call,
    table6_cfg=table6_cfg,
    has_camera_words=has_camera_words,
    skip_vlm=skip_vlm,  # ← 新增
)
```

- [ ] **Step 4: 在主循环中传递 args.skip_vlm**

在约第 207-217 行，找到主循环中的 `process_sample_with_index` 调用，添加 `skip_vlm` 参数：

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

- [ ] **Step 5: 验证语法**

运行：
```bash
python -m py_compile scripts/run_stage3_cmcc.py
```

预期：无输出（编译成功）

- [ ] **Step 6: 提交**

```bash
git add scripts/run_stage3_cmcc.py
git commit -m "feat(stage3): wire skip_vlm parameter through call chain

- Pass skip_vlm from command line args to process_sample_stage3
- Update process_sample_with_index to accept and forward skip_vlm
- Complete end-to-end skip_vlm data flow

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: 集成测试 - 冒烟测试

**Files:**
- Test: 在 CMCC 机器上运行实际测试

**Interfaces:**
- Consumes: 完整实现的 skip_vlm 功能
- Produces: 验证结果和性能数据

- [ ] **Step 1: 准备测试数据**

在 CMCC 机器上创建小测试集（10 个样本）：

```bash
head -10 /root/work/david_work/qc_output_new/smoke_test_manifest.jsonl > /tmp/test_10_samples.jsonl
```

- [ ] **Step 2: 运行 skip_vlm 测试**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3_cmcc.py \
  --stage12-jsonl /tmp/test_10_samples.jsonl \
  --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
  --output-dir /tmp/stage3_skip_vlm_test \
  --qwen-dir /root/work/david_work/models/Qwen3.5-9B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml \
  --skip-vlm
```

预期：
- 运行完成无崩溃
- 处理 10 个样本
- 总耗时 < 30 秒（~3 秒/样本）

- [ ] **Step 3: 验证输出格式**

```bash
cat /tmp/stage3_skip_vlm_test/stage3_worker000.jsonl | head -1 | jq .
```

预期：
- JSON 格式正确
- 包含 `stage3` 字段
- `unimatch_flow` 有值
- `dover` 有值
- `vlm_entity_count` 为 `null`
- `vlm_quality` 为 `null`
- `caption_revised` 为 `null`

- [ ] **Step 4: 验证性能**

```bash
# 检查处理速度
total_lines=$(wc -l < /tmp/stage3_skip_vlm_test/stage3_worker000.jsonl)
echo "处理样本数: $total_lines"

# 检查 GPU 使用情况（在测试期间运行）
nvidia-smi
```

预期：
- 样本数 = 10
- GPU 利用率 > 50%

- [ ] **Step 5: 验证 VLM 字段为 null**

```bash
cat /tmp/stage3_skip_vlm_test/stage3_worker000.jsonl | jq '.stage3 | {vlm_entity_count, vlm_quality, caption_revised}'
```

预期：所有 10 个样本的这三个字段都是 `null`

- [ ] **Step 6: 记录测试结果**

创建测试报告：

```bash
cat > /tmp/stage3_skip_vlm_test/TEST_REPORT.md << 'EOF'
# Stage 3 Skip VLM 冒烟测试报告

## 测试环境
- 日期: $(date)
- 机器: CMCC H100
- GPU ID: 0

## 测试结果
- ✅ 运行完成无崩溃
- ✅ 处理 10 个样本
- ✅ 输出格式正确
- ✅ VLM 字段为 null
- ✅ UniMatch 和 DOVER 有值
- ✅ GPU 利用率正常

## 性能数据
- 总耗时: [填写实际值] 秒
- 单样本耗时: [填写实际值] 秒
- GPU 利用率: [填写实际值] %

## 结论
skip_vlm 功能正常工作
EOF
```

---

## Self-Review Checklist

### Spec Coverage
- ✅ 添加 `--skip-vlm` 参数 (Task 1)
- ✅ 参数重命名 `--stage1-jsonl` → `--stage12-jsonl` (Task 1)
- ✅ 修改 `process_sample_stage3` 函数签名 (Task 2)
- ✅ 实现 VLM 跳过逻辑 (Task 3)
- ✅ 端到端参数传递 (Task 4)
- ✅ 集成测试 (Task 5)

### Placeholder Scan
- ✅ 无 TBD/TODO
- ✅ 所有代码块完整
- ✅ 测试命令具体明确

### Type Consistency
- ✅ `skip_vlm: bool` 在所有函数签名中一致
- ✅ `stage12_jsonl: Path` 类型一致

### Implementation Completeness
- ✅ 所有修改点都有具体代码
- ✅ 所有测试都有验证步骤
- ✅ 提交信息清晰完整

---

**Plan Version:** v1.0  
**Created:** 2026-08-07  
**Estimated Time:** 30 minutes (15 min code + 10 min test + 5 min deploy)
