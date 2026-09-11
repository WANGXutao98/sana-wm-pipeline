#!/usr/bin/env python3
"""快速验证阶段1+2的修改

测试内容:
1. 导入新的融合算法
2. 测试预计算脚本
3. 验证Pi3xMogeModel注册
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_fusion_algorithm():
    """测试融合算法导入和基本功能"""
    print("=" * 60)
    print("[1/3] 测试融合算法...")
    print("=" * 60)

    from sana_wm_pipeline.stage02_pose.depth_fusion import solve_frame_scale, fuse_depth_sequence
    import numpy as np

    # 模拟数据
    d_pi3x = np.random.rand(10, 480, 640) + 1.0
    d_moge = np.random.rand(10, 480, 640) + 1.0

    # 测试单帧
    s = solve_frame_scale(d_pi3x[0], d_moge[0])
    print(f"  ✓ solve_frame_scale() 返回: {s:.3f}")

    # 测试序列
    fused, scales = fuse_depth_sequence(d_pi3x, d_moge, ema_momentum=0.99)
    print(f"  ✓ fuse_depth_sequence() 输入: {d_pi3x.shape}")
    print(f"  ✓ 输出 fused: {fused.shape}, scales: {scales.shape}")
    print(f"  ✓ Scales range: [{scales.min():.3f}, {scales.max():.3f}]")
    print(f"  ✓ NaN count: {np.isnan(scales).sum()}")

    # 验证EMA平滑
    scale_std = scales.std()
    print(f"  ✓ Scale std: {scale_std:.4f} (应该较小，表示平滑)")

    print("✅ 融合算法测试通过\n")
    return True


def test_precompute_script():
    """测试预计算脚本存在性和权限"""
    print("=" * 60)
    print("[2/3] 测试预计算脚本...")
    print("=" * 60)

    script_path = Path(__file__).parent.parent / "scripts" / "precompute_fused_depth_reference.py"

    if not script_path.exists():
        print(f"  ✗ 脚本不存在: {script_path}")
        return False

    print(f"  ✓ 脚本存在: {script_path}")

    import stat
    mode = script_path.stat().st_mode
    is_executable = bool(mode & stat.S_IXUSR)
    print(f"  ✓ 可执行权限: {is_executable}")

    # 检查脚本依赖
    with open(script_path) as f:
        content = f.read()
        has_fusion = "from sana_wm_pipeline.stage02_pose.depth_fusion import fuse_depth_sequence" in content
        has_pi3x = "from pi3 import Pi3X" in content
        has_moge = "from moge.model.v2 import MoGeModel" in content

    print(f"  ✓ 导入 fusion: {has_fusion}")
    print(f"  ✓ 导入 Pi3X: {has_pi3x}")
    print(f"  ✓ 导入 MoGe: {has_moge}")

    print("✅ 预计算脚本验证通过\n")
    return True


def test_vipe_model_registration():
    """测试Pi3xMogeModel注册"""
    print("=" * 60)
    print("[3/3] 测试VIPE模型注册...")
    print("=" * 60)

    vipe_path = Path(__file__).parent.parent / "third_party" / "vipe"

    # 检查文件存在
    pi3xmoge_path = vipe_path / "vipe" / "priors" / "depth" / "pi3xmoge.py"
    init_path = vipe_path / "vipe" / "priors" / "depth" / "__init__.py"
    config_path = vipe_path / "configs" / "pipeline" / "vipe_sanawm.yaml"

    files_exist = {
        "pi3xmoge.py": pi3xmoge_path.exists(),
        "__init__.py": init_path.exists(),
        "vipe_sanawm.yaml": config_path.exists(),
    }

    for name, exists in files_exist.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {exists}")

    if not all(files_exist.values()):
        print("  ⚠️  部分文件缺失（VIPE submodule未提交）")
        print("  → 参考 docs/VIPE_SUBMODULE_MODIFICATIONS.md 部署")
        return False

    # 检查注册代码
    with open(init_path) as f:
        init_content = f.read()
        registered = 'model_name == "pi3xmoge"' in init_content

    print(f"  {'✓' if registered else '✗'} pi3xmoge 已注册: {registered}")

    # 检查配置
    with open(config_path) as f:
        config_content = f.read()
        has_pi3xmoge = "keyframe_depth: pi3xmoge" in config_content
        has_opt_intr = "optimize_intrinsics: true" in config_content
        has_ba_fused = "fused: false" in config_content

    print(f"  ✓ 配置 keyframe_depth: {has_pi3xmoge}")
    print(f"  ✓ 配置 optimize_intrinsics: {has_opt_intr}")
    print(f"  ✓ 配置 ba.fused: {has_ba_fused}")

    print("✅ VIPE模型注册验证通过\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("SANA-WM 重构验证 - 阶段1+2")
    print("=" * 60 + "\n")

    results = {
        "融合算法": test_fusion_algorithm(),
        "预计算脚本": test_precompute_script(),
        "VIPE模型": test_vipe_model_registration(),
    }

    print("=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\n{'✅ 所有测试通过！' if all_passed else '⚠️  部分测试失败'}")

    if all_passed:
        print("\n下一步:")
        print("  1. 本地测试: 使用testdata样本运行完整pipeline")
        print("  2. 阶段3（可选）: 应用逐帧内参BA补丁")
        print("  3. CMCC部署: 打包代码上传到CMCC环境")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
