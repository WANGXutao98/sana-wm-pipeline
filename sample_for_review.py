#!/usr/bin/env python3
"""
从 Stage 1 结果中采样样本用于人工审查
基于 sampling_plan.json
"""
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# 设置随机种子，确保可重复
random.seed(42)

QC_OUTPUT_BASE = Path("qc_full_output")
OUTPUT_DIR = Path("human_review_samples")
SAMPLING_PLAN_FILE = "sampling_plan.json"

def load_samples(group: str) -> List[Dict]:
    """加载某个 group 的所有样本"""
    jsonl_file = QC_OUTPUT_BASE / group / "stage1_results.jsonl"

    if not jsonl_file.exists():
        print(f"⚠️  {group}: {jsonl_file} 不存在")
        return []

    samples = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    return samples


def sample_flag(samples: List[Dict], n: int) -> List[Dict]:
    """
    从 flag 样本中采样
    优先级：边界样本 > 多问题样本 > 随机
    """
    flag_samples = [s for s in samples if s['verdict'] == 'flag']

    if len(flag_samples) <= n:
        return flag_samples

    # 分类
    boundary = []  # 边界样本（接近阈值）
    multi_issue = []  # 多问题样本
    single_issue = []  # 单问题样本

    for s in flag_samples:
        reasons = s.get('flag_reasons', [])

        # 检查是否是边界样本（包含阈值相关的 reason）
        is_boundary = any(
            'n_jumps' in str(r) or 'caption_len' in str(r) or 'saturation' in str(r)
            for r in reasons
        )

        if len(reasons) >= 2:
            multi_issue.append(s)
        elif is_boundary:
            boundary.append(s)
        else:
            single_issue.append(s)

    # 按优先级采样
    selected = []

    # 1. 边界样本（40%）
    n_boundary = min(len(boundary), int(n * 0.4))
    selected.extend(random.sample(boundary, n_boundary))

    # 2. 多问题样本（30%）
    n_multi = min(len(multi_issue), int(n * 0.3))
    selected.extend(random.sample(multi_issue, n_multi))

    # 3. 剩余从单问题中随机采样
    remaining = n - len(selected)
    if remaining > 0 and single_issue:
        n_single = min(len(single_issue), remaining)
        selected.extend(random.sample(single_issue, n_single))

    # 如果还不够，从剩余的随机补充
    if len(selected) < n:
        pool = [s for s in flag_samples if s not in selected]
        remaining = n - len(selected)
        if pool:
            selected.extend(random.sample(pool, min(len(pool), remaining)))

    return selected[:n]


def sample_pass(samples: List[Dict], n: int) -> List[Dict]:
    """从 pass 样本中随机采样"""
    pass_samples = [s for s in samples if s['verdict'] == 'pass']

    if len(pass_samples) <= n:
        return pass_samples

    return random.sample(pass_samples, n)


def sample_fail(samples: List[Dict], n: int) -> List[Dict]:
    """
    从 fail 样本中采样边界样本
    优先选择接近 max_jumps_fail 阈值的
    """
    fail_samples = [s for s in samples if s['verdict'] == 'fail']

    if len(fail_samples) <= n:
        return fail_samples

    # 尝试找边界样本（n_jumps 接近阈值）
    boundary_fail = []
    other_fail = []

    for s in fail_samples:
        reasons = s.get('flag_reasons', [])
        has_njumps = any('n_jumps' in str(r) and 'max_jumps_fail' in str(r) for r in reasons)

        if has_njumps:
            boundary_fail.append(s)
        else:
            other_fail.append(s)

    # 优先采样边界样本
    selected = []
    if boundary_fail:
        n_boundary = min(len(boundary_fail), int(n * 0.7))
        selected.extend(random.sample(boundary_fail, n_boundary))

    # 剩余随机采样
    remaining = n - len(selected)
    if remaining > 0 and other_fail:
        selected.extend(random.sample(other_fail, min(len(other_fail), remaining)))

    return selected[:n]


def main():
    # 加载采样方案
    if not Path(SAMPLING_PLAN_FILE).exists():
        print(f"❌ 采样方案文件不存在: {SAMPLING_PLAN_FILE}")
        print("请先运行 generate_sampling_plan.py")
        return

    with open(SAMPLING_PLAN_FILE, 'r') as f:
        sampling_plan = json.load(f)

    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("开始采样...")
    print("=" * 60)
    print()

    all_sampled = []
    stats = defaultdict(lambda: {"flag": 0, "pass": 0, "fail": 0, "total": 0})

    for group, plan in sampling_plan.items():
        print(f"处理 {group}...")

        # 加载样本
        samples = load_samples(group)
        if not samples:
            print(f"  ⚠️  跳过（无数据）")
            print()
            continue

        quota = plan['samples']

        # 采样
        sampled_flag = sample_flag(samples, quota['flag'])
        sampled_pass = sample_pass(samples, quota['pass'])
        sampled_fail = sample_fail(samples, quota['fail'])

        # 合并并添加元数据
        for s in sampled_flag:
            s['review_category'] = 'flag'
            s['review_group'] = group
            all_sampled.append(s)

        for s in sampled_pass:
            s['review_category'] = 'pass'
            s['review_group'] = group
            all_sampled.append(s)

        for s in sampled_fail:
            s['review_category'] = 'fail'
            s['review_group'] = group
            all_sampled.append(s)

        # 统计
        stats[group]['flag'] = len(sampled_flag)
        stats[group]['pass'] = len(sampled_pass)
        stats[group]['fail'] = len(sampled_fail)
        stats[group]['total'] = len(sampled_flag) + len(sampled_pass) + len(sampled_fail)

        print(f"  ✅ 采样 {stats[group]['total']} 个样本")
        print(f"     Flag: {len(sampled_flag)}, Pass: {len(sampled_pass)}, Fail: {len(sampled_fail)}")
        print()

    # 打乱顺序（避免审查时的顺序偏差）
    random.shuffle(all_sampled)

    # 保存
    output_file = OUTPUT_DIR / "review_samples.jsonl"
    with open(output_file, 'w') as f:
        for sample in all_sampled:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print("=" * 60)
    print(f"✅ 采样完成！共 {len(all_sampled)} 个样本")
    print(f"保存到: {output_file}")
    print()

    # 打印统计
    print("采样统计:")
    print("-" * 60)
    for group, s in stats.items():
        print(f"{group:<30} {s['total']:>6} (Flag:{s['flag']:>4}, Pass:{s['pass']:>4}, Fail:{s['fail']:>4})")
    print("-" * 60)
    total_flag = sum(s['flag'] for s in stats.values())
    total_pass = sum(s['pass'] for s in stats.values())
    total_fail = sum(s['fail'] for s in stats.values())
    total = sum(s['total'] for s in stats.values())
    print(f"{'总计':<30} {total:>6} (Flag:{total_flag:>4}, Pass:{total_pass:>4}, Fail:{total_fail:>4})")
    print()

    # 保存统计
    stats_file = OUTPUT_DIR / "sampling_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(dict(stats), f, indent=2)

    print(f"统计信息保存到: {stats_file}")
    print()

    print("=" * 60)
    print("下一步:")
    print("=" * 60)
    print("1. 生成黄金样本（15-20个）用于标注者培训")
    print("2. 准备审查界面")
    print("3. 开始人工审查")


if __name__ == "__main__":
    main()
