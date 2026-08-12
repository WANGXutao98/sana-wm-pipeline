#!/usr/bin/env python3
"""
生成黄金样本集（Gold Standard Samples）
用于标注者培训和质量标准制定
"""
import json
import random
from pathlib import Path
from typing import List, Dict

random.seed(42)

REVIEW_SAMPLES_FILE = Path("human_review_samples/review_samples.jsonl")
GOLDEN_OUTPUT = Path("human_review_samples/golden_samples")
TARGET_GOLDEN_COUNT = 20  # 目标黄金样本数

def load_review_samples() -> List[Dict]:
    """加载所有审查样本"""
    samples = []
    with open(REVIEW_SAMPLES_FILE, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def select_golden_samples(samples: List[Dict]) -> List[Dict]:
    """
    选择黄金样本

    选择标准：
    1. 覆盖所有主要 group
    2. 覆盖所有 verdict 类型 (pass/fail/flag)
    3. 代表性的 flag_reasons
    4. 难度分级：简单、中等、困难
    """

    golden = []

    # 按 group 和 verdict 分组
    by_group_verdict = {}
    for s in samples:
        group = s['review_group']
        verdict = s['verdict']
        key = (group, verdict)

        if key not in by_group_verdict:
            by_group_verdict[key] = []
        by_group_verdict[key].append(s)

    print("=" * 60)
    print("黄金样本选择策略")
    print("=" * 60)
    print()

    # 1. 明显的 Pass（5个）
    print("1. 明显的 Pass 样本（5个）- 高质量基准")
    pass_candidates = [s for s in samples if s['verdict'] == 'pass']

    # 选择不同 group 的 pass 样本
    groups_covered = set()
    for s in pass_candidates:
        if len([g for g in golden if g['verdict'] == 'pass']) >= 5:
            break
        if s['review_group'] not in groups_covered:
            golden.append(s)
            groups_covered.add(s['review_group'])
            print(f"   ✓ {s['review_group']}: {s['sample_id'][:40]}...")

    print()

    # 2. 明显的 Fail（3个）
    print("2. 明显的 Fail 样本（3个）- 明确的低质量")
    fail_candidates = [s for s in samples if s['verdict'] == 'fail']

    # 选择不同类型的 fail
    fail_selected = random.sample(fail_candidates, min(3, len(fail_candidates)))
    for s in fail_selected:
        golden.append(s)
        reasons_str = ', '.join(s.get('flag_reasons', [])[:2])
        print(f"   ✓ {s['review_group']}: {reasons_str}")

    print()

    # 3. 边界 Flag 样本（7个）- 重点
    print("3. 边界 Flag 样本（7个）- 需要人工判断的边界案例")
    flag_candidates = [s for s in samples if s['verdict'] == 'flag']

    # 选择代表性的 flag_reasons
    flag_types = {
        'n_jumps': [],
        'camera_word': [],
        'caption': [],
        'saturation': [],
        'multi_issue': [],  # 多问题
        'other': []
    }

    for s in flag_candidates:
        reasons = s.get('flag_reasons', [])
        reasons_str = ' '.join(str(r) for r in reasons)

        if len(reasons) >= 2:
            flag_types['multi_issue'].append(s)
        elif 'n_jumps' in reasons_str:
            flag_types['n_jumps'].append(s)
        elif 'camera_word' in reasons_str:
            flag_types['camera_word'].append(s)
        elif 'caption' in reasons_str:
            flag_types['caption'].append(s)
        elif 'saturation' in reasons_str:
            flag_types['saturation'].append(s)
        else:
            flag_types['other'].append(s)

    # 从每种类型中选择
    flag_quota = {
        'n_jumps': 2,
        'camera_word': 1,
        'multi_issue': 2,
        'other': 2,
    }

    for flag_type, quota in flag_quota.items():
        candidates = flag_types[flag_type]
        if candidates:
            selected = random.sample(candidates, min(quota, len(candidates)))
            for s in selected:
                golden.append(s)
                reasons_str = ', '.join(str(r) for r in s.get('flag_reasons', [])[:2])
                print(f"   ✓ [{flag_type}] {s['review_group']}: {reasons_str}")

    print()

    # 4. 难度分级样本（5个）
    print("4. 难度分级样本（5个）")
    print("   ✓ 选择不同难度的样本用于一致性测试")

    # 剩余配额
    remaining = TARGET_GOLDEN_COUNT - len(golden)
    if remaining > 0:
        remaining_pool = [s for s in samples if s not in golden]
        if remaining_pool:
            additional = random.sample(remaining_pool, min(remaining, len(remaining_pool)))
            golden.extend(additional)
            print(f"   ✓ 补充 {len(additional)} 个样本")

    print()
    print(f"总计: {len(golden)} 个黄金样本")

    return golden


def save_golden_samples(golden: List[Dict]):
    """保存黄金样本"""

    GOLDEN_OUTPUT.mkdir(exist_ok=True)

    # 1. 保存 JSONL
    jsonl_file = GOLDEN_OUTPUT / "golden_samples.jsonl"
    with open(jsonl_file, 'w') as f:
        for s in golden:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"✅ 黄金样本保存到: {jsonl_file}")

    # 2. 生成人类可读的摘要
    summary_file = GOLDEN_OUTPUT / "golden_samples_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("黄金样本集摘要\n")
        f.write("=" * 80 + "\n\n")

        for i, s in enumerate(golden, 1):
            f.write(f"样本 #{i}\n")
            f.write(f"  ID: {s['sample_id']}\n")
            f.write(f"  Group: {s['review_group']}\n")
            f.write(f"  Verdict: {s['verdict']}\n")
            if s['verdict'] == 'flag':
                f.write(f"  Flag Reasons: {', '.join(str(r) for r in s.get('flag_reasons', []))}\n")
            f.write(f"  Metrics:\n")
            metrics = s.get('metrics', {})
            f.write(f"    - n_jumps: {metrics.get('n_jumps', 'N/A')}\n")
            f.write(f"    - caption_len: {metrics.get('caption_len', 'N/A')}\n")
            f.write(f"    - traj_total_m: {metrics.get('traj_total_m', 'N/A')}\n")
            f.write("\n")

    print(f"✅ 摘要保存到: {summary_file}")

    # 3. 生成标注模板
    template_file = GOLDEN_OUTPUT / "annotation_template.txt"
    with open(template_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("黄金样本标注模板\n")
        f.write("=" * 80 + "\n\n")
        f.write("标注说明:\n")
        f.write("  对于每个样本，请判断:\n")
        f.write("  1. 质量评级: [优秀 / 良好 / 可接受 / 差]\n")
        f.write("  2. 是否用于训练: [是 / 否]\n")
        f.write("  3. 主要问题（如果有）\n")
        f.write("  4. 备注\n\n")
        f.write("=" * 80 + "\n\n")

        for i, s in enumerate(golden, 1):
            f.write(f"样本 #{i}: {s['sample_id']}\n")
            f.write(f"Group: {s['review_group']} | Verdict: {s['verdict']}\n")
            if s['verdict'] == 'flag':
                f.write(f"Flag Reasons: {', '.join(str(r) for r in s.get('flag_reasons', []))}\n")
            f.write("\n")
            f.write("[ ] 质量评级: ___________\n")
            f.write("[ ] 用于训练: ___________\n")
            f.write("[ ] 主要问题: ___________\n")
            f.write("[ ] 备注: ___________\n")
            f.write("\n" + "-" * 80 + "\n\n")

    print(f"✅ 标注模板保存到: {template_file}")


def main():
    if not REVIEW_SAMPLES_FILE.exists():
        print(f"❌ 审查样本文件不存在: {REVIEW_SAMPLES_FILE}")
        print("请先运行 sample_for_review.py")
        return

    print("=" * 60)
    print("生成黄金样本集")
    print("=" * 60)
    print()

    # 加载样本
    samples = load_review_samples()
    print(f"✅ 加载 {len(samples)} 个审查样本")
    print()

    # 选择黄金样本
    golden = select_golden_samples(samples)
    print()

    # 保存
    save_golden_samples(golden)
    print()

    print("=" * 60)
    print("下一步:")
    print("=" * 60)
    print("1. 审查黄金样本，制定质量标准")
    print("2. 使用黄金样本培训标注者")
    print("3. 测试标注者一致性（多人标注同一样本）")
    print("4. 开始大规模人工审查")


if __name__ == "__main__":
    main()
