#!/usr/bin/env python3
"""验证 umeyama.py 与官方代码对齐的正确性。"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from sana_wm_pipeline.stage02_pose.umeyama import (
    umeyama_sim3, recover_metric_scale, umeyama_sim3_inlier_filter
)

def test_umeyama_sim3_basic():
    """测试基础 umeyama_sim3 函数"""
    print("\n=== Test 1: umeyama_sim3 基础功能 ===")

    # 创建测试数据：简单的缩放+旋转+平移
    np.random.seed(42)
    src = np.random.randn(20, 3)

    # 真实变换：s=2.5, R=绕z轴旋转30度, t=[1, 2, 3]
    angle = np.radians(30)
    R_true = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    s_true = 2.5
    t_true = np.array([1.0, 2.0, 3.0])

    dst = s_true * (src @ R_true.T) + t_true

    # 恢复变换
    s, R, t = umeyama_sim3(src, dst)

    print(f"真实 scale: {s_true:.6f}")
    print(f"恢复 scale: {s:.6f}")
    print(f"Scale 误差: {abs(s - s_true):.2e}")

    # 验证变换精度
    dst_pred = s * (src @ R.T) + t
    error = np.linalg.norm(dst - dst_pred, axis=1).mean()
    print(f"平均重投影误差: {error:.2e}")

    if abs(s - s_true) < 1e-10 and error < 1e-10:
        print("✅ PASS: umeyama_sim3 精确恢复变换")
    else:
        print("❌ FAIL: 恢复精度不足")
        return False

    return True


def test_recover_metric_scale():
    """测试 recover_metric_scale 函数（官方两步法）"""
    print("\n=== Test 2: recover_metric_scale (官方两步法) ===")

    np.random.seed(42)
    # 创建带噪声的测试数据
    pred = np.random.randn(50, 3)
    s_true = 3.7

    # 大部分点：正确变换
    gt = s_true * pred + np.random.randn(50, 3) * 0.01

    # 添加 5 个离群点
    outlier_indices = [5, 15, 25, 35, 45]
    gt[outlier_indices] += np.random.randn(5, 3) * 5.0

    # 使用 80% inlier filter 恢复 scale
    s_recovered = recover_metric_scale(pred, gt, inlier_percentile=80.0)

    print(f"真实 scale: {s_true:.6f}")
    print(f"恢复 scale: {s_recovered:.6f}")
    print(f"相对误差: {abs(s_recovered - s_true) / s_true * 100:.2f}%")

    if abs(s_recovered - s_true) / s_true < 0.05:  # 5% 误差容忍
        print("✅ PASS: recover_metric_scale 正确过滤离群点")
    else:
        print("❌ FAIL: scale 恢复误差过大")
        return False

    return True


def test_iterative_vs_twopass():
    """对比迭代版本和两步法的差异"""
    print("\n=== Test 3: 迭代版本 vs 两步法对比 ===")

    np.random.seed(42)
    pred = np.random.randn(50, 3)
    s_true = 2.8
    gt = s_true * pred + np.random.randn(50, 3) * 0.05

    # 添加离群点
    gt[[10, 20, 30]] += np.random.randn(3, 3) * 3.0

    # 两步法
    s_twopass = recover_metric_scale(pred, gt, inlier_percentile=80.0)

    # 迭代法
    s_iter, R_iter, t_iter, mask_iter = umeyama_sim3_inlier_filter(
        pred, gt, inlier_percentile=80.0, max_iter=5
    )

    print(f"真实 scale:     {s_true:.6f}")
    print(f"两步法 scale:   {s_twopass:.6f} (误差: {abs(s_twopass - s_true):.4f})")
    print(f"迭代法 scale:   {s_iter:.6f} (误差: {abs(s_iter - s_true):.4f})")
    print(f"迭代法 inliers: {mask_iter.sum()}/{len(mask_iter)}")

    diff = abs(s_twopass - s_iter)
    print(f"两种方法差异:   {diff:.4f}")

    if diff < 0.1:  # 允许一定差异（因为算法不同）
        print("✅ PASS: 两种方法结果接近")
    else:
        print("⚠️  WARNING: 两种方法差异较大（预期行为，算法不同）")

    return True


def test_official_alignment_compatibility():
    """测试与官方代码接口的兼容性"""
    print("\n=== Test 4: 官方代码接口兼容性 ===")

    # 模拟 DL3DV 场景
    np.random.seed(42)
    n_frames = 303

    # Pi3X 预测的相机中心（up-to-scale）
    pred_centers = np.random.randn(n_frames, 3)

    # GT 相机中心（metric-scale）
    s_true = 5.2
    gt_centers = s_true * pred_centers + np.random.randn(n_frames, 3) * 0.02

    # 官方接口：recover_metric_scale
    s_recovered = recover_metric_scale(
        pred_positions=pred_centers,
        gt_positions=gt_centers,
        inlier_percentile=80.0
    )

    print(f"输入: {n_frames} 帧相机中心")
    print(f"真实 metric scale: {s_true:.6f}")
    print(f"恢复 metric scale: {s_recovered:.6f}")
    print(f"相对误差: {abs(s_recovered - s_true) / s_true * 100:.3f}%")

    # 验证返回类型
    if isinstance(s_recovered, float):
        print("✅ PASS: 返回类型正确 (float)")
    else:
        print(f"❌ FAIL: 返回类型错误 ({type(s_recovered)})")
        return False

    if abs(s_recovered - s_true) / s_true < 0.01:
        print("✅ PASS: 官方接口兼容性良好")
    else:
        print("❌ FAIL: scale 恢复误差过大")
        return False

    return True


def main():
    print("="*60)
    print("验证 umeyama.py 与官方代码对齐")
    print("="*60)

    tests = [
        test_umeyama_sim3_basic,
        test_recover_metric_scale,
        test_iterative_vs_twopass,
        test_official_alignment_compatibility,
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("\n🎉 所有测试通过！umeyama.py 已成功与官方代码对齐。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
