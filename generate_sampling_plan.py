#!/usr/bin/env python3
"""
生成人工审查采样方案
基于 Stage 1+2 优化后的结果
"""
import json
from pathlib import Path
from collections import defaultdict

# 配置
QC_OUTPUT_BASE = Path("qc_full_output")
TARGET_SAMPLES = 6500  # 目标审查样本数

# 各 group 的实际数据（从结果中）
GROUP_STATS = {
    "DL3DV-ALL-2K": {"pass": 6462, "fail": 668, "flag": 2807, "priority": "high"},
    "RealEstate10K-360p": {"pass": 65769, "fail": 4720, "flag": 449, "priority": "medium"},
    "OmniWorld-Game": {"pass": 5983, "fail": 126, "flag": 36, "priority": "high"},
    "sekai-game-drone": {"pass": 898, "fail": 26, "flag": 7, "priority": "medium"},
    "sekai-game-walking": {"pass": 1421, "fail": 170, "flag": 11, "priority": "medium"},
    "SpatialVID-hq": {"pass": 35042, "fail": 1948, "flag": 632, "priority": "low"},
    "sekai-real-walking-hq": {"pass": 18201, "fail": 915, "flag": 122, "priority": "low"},
}

def calculate_sampling_quota():
    """计算各 group 的采样配额"""

    total_flag = sum(s["flag"] for s in GROUP_STATS.values())
    total_samples = sum(s["pass"] + s["fail"] + s["flag"] for s in GROUP_STATS.values())

    print("=" * 60)
    print("人工审查采样方案")
    print("=" * 60)
    print()
    print(f"总样本数: {total_samples:,}")
    print(f"Pass: {sum(s['pass'] for s in GROUP_STATS.values()):,}")
    print(f"Fail: {sum(s['fail'] for s in GROUP_STATS.values()):,}")
    print(f"Flag: {total_flag:,}")
    print()
    print(f"目标审查样本数: {TARGET_SAMPLES:,}")
    print()

    # 采样策略
    sampling_plan = {}

    for group, stats in GROUP_STATS.items():
        plan = {
            "total": stats["pass"] + stats["fail"] + stats["flag"],
            "pass": stats["pass"],
            "fail": stats["fail"],
            "flag": stats["flag"],
        }

        # 根据 flag 数量和优先级分配配额
        if stats["priority"] == "high":
            # 高优先级：多采样
            if stats["flag"] > 1000:
                # 大量 flag: 采样 2000
                flag_sample = min(2000, stats["flag"])
            elif stats["flag"] > 100:
                # 中等 flag: 采样 800
                flag_sample = min(800, stats["flag"])
            else:
                # 少量 flag: 全部审查 + 补充 pass
                flag_sample = stats["flag"]
        elif stats["priority"] == "medium":
            # 中等优先级
            if stats["flag"] > 100:
                flag_sample = min(400, stats["flag"])
            else:
                flag_sample = stats["flag"]
        else:  # low
            # 低优先级：已经高 pass 率，少量采样验证
            flag_sample = min(500, stats["flag"])

        # Pass 样本采样（验证管线准确性）
        if stats["pass"] > 10000:
            # 大量 pass: 采样 500
            pass_sample = 500
        elif stats["pass"] > 1000:
            # 中等 pass: 采样 300
            pass_sample = 300
        else:
            # 少量 pass: 采样 20%
            pass_sample = int(stats["pass"] * 0.2)

        # 如果 flag 很少，增加 pass 采样来补充
        if flag_sample < 100:
            pass_sample = max(pass_sample, 400)

        # Fail 样本采样（边界样本，接近阈值的）
        if stats["fail"] > 100:
            fail_sample = 100
        else:
            fail_sample = min(50, stats["fail"])

        plan["samples"] = {
            "flag": flag_sample,
            "pass": pass_sample,
            "fail": fail_sample,
            "total": flag_sample + pass_sample + fail_sample,
        }

        sampling_plan[group] = plan

    # 打印采样方案
    print("=" * 80)
    print(f"{'Group':<30} {'Total':<10} {'Pass':<10} {'Flag':<10} {'采样':<10}")
    print("=" * 80)

    total_to_review = 0
    for group, plan in sampling_plan.items():
        samples = plan["samples"]
        total_to_review += samples["total"]
        print(f"{group:<30} {plan['total']:<10,} {plan['pass']:<10,} {plan['flag']:<10,} {samples['total']:<10,}")
        print(f"  └─ 采样明细: flag={samples['flag']}, pass={samples['pass']}, fail={samples['fail']}")

    print("=" * 80)
    print(f"{'总计':<30} {'':<10} {'':<10} {'':<10} {total_to_review:<10,}")
    print()

    if total_to_review > TARGET_SAMPLES:
        print(f"⚠️  采样总数 {total_to_review:,} 超过目标 {TARGET_SAMPLES:,}")
        print(f"建议调整配额或分批审查")
    else:
        print(f"✅ 采样总数 {total_to_review:,} 符合目标 {TARGET_SAMPLES:,}")

    print()

    # 保存方案
    output_file = "sampling_plan.json"
    with open(output_file, "w") as f:
        json.dump(sampling_plan, f, indent=2)

    print(f"✅ 采样方案已保存到: {output_file}")
    print()

    return sampling_plan


def generate_sampling_strategy_details():
    """生成详细的采样策略说明"""

    print("=" * 60)
    print("详细采样策略")
    print("=" * 60)
    print()

    strategies = {
        "Flag 样本 (60%)": [
            "• 边界样本：flag_reasons 包含阈值类 (n_jumps, caption_len 等)",
            "• 多问题样本：len(flag_reasons) >= 2",
            "• 随机 flag 样本：验证管线的 flag 准确性",
        ],
        "Pass 样本 (30%)": [
            "• 随机抽取 pass 样本",
            "• 目的：验证管线的 pass 准确性（召回率）",
            "• 发现管线的盲区",
        ],
        "Fail 样本 (10%)": [
            "• 边界 fail：接近 max_jumps_fail 阈值的样本",
            "• 目的：验证 fail 阈值是否合理",
        ],
    }

    for category, items in strategies.items():
        print(f"{category}:")
        for item in items:
            print(f"  {item}")
        print()

    print("采样方法:")
    print("  1. 分层采样：按 group 和 verdict 分层")
    print("  2. 优先级采样：边界样本 > 多问题样本 > 随机样本")
    print("  3. 去重：确保不重复采样")
    print()


if __name__ == "__main__":
    # 计算采样配额
    sampling_plan = calculate_sampling_quota()

    # 生成详细策略
    generate_sampling_strategy_details()

    print()
    print("=" * 60)
    print("下一步：")
    print("=" * 60)
    print("1. 审查采样方案，确认配额分配")
    print("2. 运行采样脚本生成审查批次")
    print("3. 生成黄金样本（15-20个）")
    print("4. 开始人工审查")
