#!/usr/bin/env python3
"""Compare GT-pose vs Default pose-annotation modes on DL3DV data.

Reads two pose_eval_summary.json files (produced by verify_and_eval.py),
finds common sample_ids, and writes a Markdown comparison report.

Usage:
  python compare_modes.py \
    --gtpose-eval  /path/to/dl3dv_smoke_shards_gtpose/eval_output/pose_eval_summary.json \
    --default-eval /path/to/dl3dv_smoke_shards_default/eval_output/pose_eval_summary.json \
    --out          docs/operation_logs/2026-06-12-mode-comparison.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_results(path: Path) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    return {r["sample_id"]: r for r in data}


def fmt_ate(ate: float | None) -> str:
    if ate is None:
        return "N/A"
    if ate < 1e-4:
        return f"{ate:.2e}"
    return f"{ate:.6f}"


def generate_report(
    gtpose_path: Path,
    default_path: Path,
    out_path: Path,
) -> None:
    gtpose = load_results(gtpose_path)
    default = load_results(default_path)

    common_ids = sorted(set(gtpose) & set(default))
    all_gtpose = sorted(gtpose)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []
    lines.append(f"# GT-pose vs Default 模式对比报告")
    lines.append(f"")
    lines.append(f"> 生成时间：{now}")
    lines.append(f"")
    lines.append(f"## 实验设置")
    lines.append(f"")
    lines.append(f"| 项目 | 说明 |")
    lines.append(f"|---|---|")
    lines.append(f"| 数据集 | DL3DV（Smoke Test，4 scenes） |")
    lines.append(f"| 评测指标 | ATE RMSE（绝对轨迹误差，align=True, correct_scale=True） |")
    lines.append(f"| GT-pose 样本数 | {len(all_gtpose)} |")
    lines.append(f"| Default 样本数 | {len(default)} |")
    lines.append(f"| 公共样本数（用于对比） | {len(common_ids)} |")
    lines.append(f"| GT-pose eval JSON | `{gtpose_path}` |")
    lines.append(f"| Default eval JSON | `{default_path}` |")
    lines.append(f"")
    lines.append(f"## 模式说明")
    lines.append(f"")
    lines.append(f"### GT-pose 模式（论文 §4 Dataset-specific annotation modes）")
    lines.append(f"")
    lines.append(f"- **适用数据集**：DL3DV、Sekai-Game（具有 GT 相机轨迹）")
    lines.append(f"- **流程**：Pi3X 预测结构 → Umeyama Sim(3)（80th 百分位 inlier 过滤）从 GT 轨迹恢复度量尺度 → 直接使用 GT 位姿 `poses_c2w`")
    lines.append(f"- **关键**：Umeyama 仅用于恢复度量尺度因子 `s`，**不**用于估计相机位姿；位姿来自 GT")
    lines.append(f"- **期望 ATE**：接近数值零（取决于 GT 精度）")
    lines.append(f"")
    lines.append(f"### Default 模式（论文 §4 Dataset-specific annotation modes）")
    lines.append(f"")
    lines.append(f"- **适用数据集**：SpatialVID-HQ、MiraData（无 GT 位姿的互联网视频）")
    lines.append(f"- **流程**：Pi3X+MoGe-2 EMA 融合深度缓存 → VIPE SLAM（`vipe_cached_depth` pipeline，CachedDepthModel 注入 BA） → 每帧位姿 + 内参")
    lines.append(f"- **已知局限**：SLAM 累积漂移，对有 GT 位姿的场景非最优")
    lines.append(f"")
    lines.append(f"## 对比结果（公共样本）")
    lines.append(f"")

    # Table header
    lines.append(f"| 样本 ID（前 12 位） | GT-pose ATE RMSE (m) | Default ATE RMSE (m) | 倍数（Default/GT-pose） |")
    lines.append(f"|---|---|---|---|")

    ratios = []
    for sid in common_ids:
        g = gtpose[sid]["ate_rmse"]
        d = default[sid]["ate_rmse"]
        ratio = d / g if g and g > 0 else float("inf")
        ratios.append(ratio)
        lines.append(
            f"| `{sid[:12]}...` | {fmt_ate(g)} | {fmt_ate(d)} | {ratio:.0f}× |"
        )

    lines.append(f"")
    lines.append(f"## GT-pose 全样本结果（{len(all_gtpose)} 场景）")
    lines.append(f"")
    lines.append(f"| 样本 ID（前 12 位） | ATE RMSE (m) | T_est | T_gt_orig | orig_fps |")
    lines.append(f"|---|---|---|---|---|")
    for sid in all_gtpose:
        r = gtpose[sid]
        lines.append(
            f"| `{sid[:12]}...` | {fmt_ate(r['ate_rmse'])} "
            f"| {r['T_est']} | {r['T_gt_orig']} | {r['orig_fps']:.1f} |"
        )

    lines.append(f"")
    lines.append(f"## 结论")
    lines.append(f"")

    if common_ids:
        ratio_mean = sum(ratios) / len(ratios)
        lines.append(f"在 DL3DV 数据集上（有精确 GT 位姿），两种模式的 ATE RMSE 差异达 **{ratio_mean:.0f}×**：")
        lines.append(f"")
        lines.append(f"| 模式 | ATE RMSE 量级 | 物理含义 |")
        lines.append(f"|---|---|---|")
        g_ex = gtpose[common_ids[0]]["ate_rmse"]
        d_ex = default[common_ids[0]]["ate_rmse"]
        lines.append(f"| GT-pose | {fmt_ate(g_ex)} m | 数值零（直接使用 GT 位姿）|")
        lines.append(f"| Default | {fmt_ate(d_ex)} m | SLAM 引入 ~{d_ex*100:.0f} cm 漂移 |")

    lines.append(f"")
    lines.append(f"### 模式选择建议")
    lines.append(f"")
    lines.append(f"| 数据集类型 | 推荐模式 | 原因 |")
    lines.append(f"|---|---|---|")
    lines.append(f"| DL3DV、Sekai-Game（有 GT 位姿） | **GT-pose** | 直接使用 GT，ATE≈0，无漂移 |")
    lines.append(f"| OmniWorld（有 GT 深度） | **GT-depth** | GT 深度替换预测深度注入 SLAM BA |")
    lines.append(f"| SpatialVID-HQ、MiraData（无 GT） | **Default** | 唯一可行方案，SLAM+Pi3X+MoGe-2 |")
    lines.append(f"")
    lines.append(f"**结论**：Default 模式（VIPE+Pi3X+MoGe-2）并非在所有数据集上都能达到最好效果。对于有 GT 位姿的数据集（DL3DV、Sekai-Game），GT-pose 模式是正确选择，Default 模式引入不必要的 SLAM 漂移，在此实验中误差放大约 {ratio_mean:.0f}×。")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*报告由 `experiments/data_production_smoke/compare_modes.py` 自动生成*")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Report saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare GT-pose vs Default mode ATE.")
    parser.add_argument("--gtpose-eval", type=Path, required=True)
    parser.add_argument("--default-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for p in (args.gtpose_eval, args.default_eval):
        if not p.exists():
            parser.error(f"File not found: {p}")

    generate_report(args.gtpose_eval, args.default_eval, args.out)


if __name__ == "__main__":
    main()
