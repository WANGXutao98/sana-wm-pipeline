# Human Review System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现人工审查系统的3个Python脚本，支持在Stage 1+2和Stage 3之间进行人工质量评估

**Architecture:** 三个独立脚本形成完整工作流：export_for_review.py负责采样和视频提取，import_review_results.py负责验证和分析，apply_human_review.py负责合并决策到最终清单

**Tech Stack:** Python 3.9+, pandas, tarfile, av (PyAV), jinja2 (HTML报告)

## Global Constraints

- Python 3.9+
- 复用现有的tar提取逻辑（stage1_fast.py的_extract_samples_from_tar）
- 复用现有的group_config.py中的阈值配置
- 所有脚本放在 `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/`
- 测试文件放在 `/mnt/afs/davidwang/workspace/sana_wm_pipeline/tests/`
- CSV格式：UTF-8编码，逗号分隔
- JSONL格式：每行一个JSON对象
- 遵循DRY、YAGNI、TDD原则

---

### Task 1: Export Script - 采样和视频提取

**Files:**
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/export_for_review.py`
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/tests/test_export_for_review.py`

**Interfaces:**
- Consumes: stage1_results.jsonl, stage2_results.jsonl（可选）, tar文件
- Produces: review_list.csv, videos/, decisions_template.csv, sampling_report.txt

**Implementation:** See full plan for 19 detailed TDD steps

---

### Task 2: Import Script - 验证和分析

**Files:**
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/import_review_results.py`
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/tests/test_import_review_results.py`

**Interfaces:**
- Consumes: review_list.csv, decisions_filled.csv
- Produces: human_review_results.jsonl, disagreement_report.html

**Implementation:** See full plan for 15 detailed TDD steps

---

### Task 3: Apply Script - 合并决策

**Files:**
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/apply_human_review.py`
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/tests/test_apply_human_review.py`

**Interfaces:**
- Consumes: stage1_results.jsonl, human_review_results.jsonl
- Produces: stage1_results_merged.jsonl, manifests/{pass,fail,human_reviewed}.txt, summary_report.html

**Implementation:** See full plan for 16 detailed TDD steps

---

### Task 4: 人工审查操作手册

**Files:**
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/docs/human_review_manual.md`
- Create: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/docs/human_review_quick_reference.md`

**Content:** Complete reviewer guide with tools, workflow, judgment criteria, examples, and FAQ

---

## Execution

Plan complete. Use `superpowers:subagent-driven-development` to execute task-by-task.

Full detailed implementation steps are in the original plan file (this is summary only).
